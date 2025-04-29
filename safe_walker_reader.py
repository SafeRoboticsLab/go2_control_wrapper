import os
import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from SEP import SafetyEnforcer as safety_enforcer

VEC_PATH = "/tmp/state_vec_77.npy"

def try_load_vec77():
    try:
        vec = np.load(VEC_PATH)
        if vec.shape == (77,):
            return vec
    except Exception:
        pass
    return np.zeros((77,), dtype=np.float32)

def main():
    ChannelFactoryInitialize(0)
    client = SportClient()
    client.SetTimeout(10.0)
    client.Init()
    client.StandUp()
    time.sleep(1)
    client.BalanceStand()

    sep = safety_enforcer(epsilon=0.5, shield_type="value", parent_dir="")
    dt = 0.05

    yaw0 = None
    prev_yaw = None

    while True:
        vec77 = try_load_vec77()
        vec76, yaw = vec77[:-1], vec77[-1]

        # Initialize yaw offset
        if yaw0 is None:
            yaw0 = yaw
            prev_yaw = yaw

        yaw_error = np.arctan2(np.sin(yaw0 - yaw), np.cos(yaw0 - yaw))
        yaw_rate = (yaw - prev_yaw) / dt

        Kp, Kd = 2.0, 0.5
        vyaw = Kp * yaw_error - Kd * yaw_rate
        vyaw = np.clip(vyaw, -1.0, 1.0)
        prev_yaw = yaw

        vx = 0.5  # constant forward speed
        ctrl = np.array([vx, vyaw], dtype=np.float32)

        input_vec = np.concatenate([ctrl, vec76])
        safe_action = sep.get_action(input_vec, ctrl)

        client.Move(*safe_action.tolist())
        print(f"vx={safe_action[0]:.2f}, vyaw={safe_action[1]:.2f}, raw_yaw={yaw:.2f}, error={yaw_error:.2f}, Q={sep.prev_q:.3f}")

        time.sleep(dt)

if __name__ == '__main__':
    main()
