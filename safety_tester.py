import time
import sys
import math
import numpy as np
from collections import deque

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

from SEP import SafetyEnforcer as safety_enforcer

DECAY_TIME = 3.0
ROBOT_RADIUS = 0.85
INTENSITY_THRESHOLD = 220
VEC_PATH = "/tmp/state_vec_76.npy"

def try_load_vec76():
    try:
        vec = np.load(VEC_PATH)
        if vec.shape == (76,):
            return vec
    except Exception:
        pass
    return np.zeros((76,), dtype=np.float32)


class SportModeSafeWalker:
    def __init__(self):
        self.dt = 0.001
        self.t = 0


        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.safety_enforcer = safety_enforcer(
            epsilon=0.5,
            shield_type="value",
            parent_dir=""
        )

        self.robot_state = unitree_go_msg_dds__SportModeState_()
        self.robot_state_ready = False

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]
        self.prev_yaw = self.yaw0
        print(self.yaw0, "Init")

    def StandAndStart(self):
        self.client.StandUp()
        time.sleep(1)
        self.client.BalanceStand()
        print("Standing complete.")

    def safe_forward_control(self, robot_state: SportModeState_):
        vx_desired = 0.0 # forward velocity
        
        vyaw_desired = 0.0
        ctrl_input = np.array([vx_desired, vyaw_desired], dtype=np.float32)
        raw_action = ctrl_input

        state_vec76 = try_load_vec76()
        full_state = np.concatenate([ctrl_input, state_vec76])
        safe_action = self.safety_enforcer.get_action(full_state, raw_action)
        assert len(safe_action) == 2 
        safe_action = np.array([safe_action[0], 0.0, safe_action[1]], dtype=np.float32)

        #Robot has a separate vy for strating but currently unused
        print(safe_action.tolist(), "list")

        self.client.Move(*safe_action.tolist())
        print(f"[Shielded: {self.safety_enforcer.get_shielding_status()} | Q: {self.safety_enforcer.prev_q:.3f}]")

    def update(self):
        if self.robot_state_ready:
            self.t += self.dt
            self.safe_forward_control(self.robot_state)

walker = None

def HighStateHandler(msg: SportModeState_):
    global walker
    if walker is not None:
        walker.robot_state = msg
        walker.robot_state_ready = True

def main(args=None):
    global walker

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    walker = SportModeSafeWalker()
    walker.GetInitState(walker.robot_state)
    walker.StandAndStart()

    while True:
        walker.update()
        time.sleep(walker.dt)

if __name__ == '__main__':
    main()
