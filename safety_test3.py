import time
import sys
import math
import numpy as np
from collections import deque
import os

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

from SEP import SafetyEnforcer as safety_enforcer

DECAY_TIME = 4.0
ROBOT_RADIUS = 0.8
INTENSITY_THRESHOLD = 220
VEC_PATH = "/tmp/state_vec_72.npy"
VIZ_PATH = "/tmp/state_vec_viz.npy"

def try_load_vec72():
    try:
        vec = np.load(VEC_PATH)
        if vec.shape == (72,):
            return vec
    except Exception:
        pass
    return np.zeros((72,), dtype=np.float32)


def scale_and_shift(x, old_range, new_range):
    ratio = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    x_new = (x - old_range[0]) * ratio + new_range[0]
    return x_new

def get_control(action):
    # clip before scaling
    if action[-1] == 1:
        xdot_range_model = [0.2, 0.1]
        xdot_range = [.2, .1]
    else:
        xdot_range_model = [.2, .5]
        xdot_range = [0.2, 0.5]
    action = np.clip(action, -1, 1)
    v = scale_and_shift(action[0], [-1, 1], xdot_range_model)
    v = np.clip(v, xdot_range[0], xdot_range[1])
    w = np.clip(action[1], -1, 1)
    return v, w

def get_safe_extreme(lidar, yaw_value) -> float:
    """
    Computes the escape angle pointing directly away from the nearest obstacle.

    Returns:
        float: Angle difference between current heading and escape direction ∈ (–π, π].
    """
    # Get LiDAR readings (relative XY coordinates of 18 closest obstacles)

    lidar_xy = lidar.reshape(18, 2)

    # Find the closest obstacle
    dists = np.linalg.norm(lidar_xy, axis=1)
    closest_idx = np.argmin(dists)
    closest_vec = lidar_xy[closest_idx]

    # Compute angle of closest vector
    angle_safe = np.arctan2(closest_vec[1], closest_vec[0])
    # Return signed angle difference ∈ (–π, π]
    clearance = np.arctan2(np.sin(yaw_value - angle_safe), np.cos(yaw_value - angle_safe))

    return np.array([np.cos(clearance), np.sin(clearance)])

class SportModeSafeWalker:
    def __init__(self):
        self.dt = 1
        self.t = 0
        self.yaw0 = 0
        self.prev_yaw = 0
        self.curr_heading = None
        self.prev_heading = None


        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self.ctrl_list = deque(maxlen=1)
        self.ctrl_list.append((0.0, 0.0)) # Append vx, vyaw

        self.safety_enforcer = safety_enforcer(
            epsilon=1.0,
            shield_type="value",
            parent_dir=""
        )

        self.viz_valid = True


    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]
        self.prev_yaw = self.yaw0
        print(self.yaw0,"Init is set as offset")
        self.offset = self.yaw0     

    def StandAndStart(self):
        self.client.StandUp()
        time.sleep(1)
        self.client.BalanceStand()
        print("Standing complete.")

    def safe_forward_control(self, robot_state: SportModeState_):
        vx_desired = 0.00# forward velocity
        yaw = robot_state.imu_state.rpy[2]
        yaw_error = math.atan2(math.sin(self.yaw0 - yaw), math.cos(self.yaw0 - yaw))

        Kp_yaw = 2.0
        Kd_yaw = 0.5
        yaw_rate = (yaw - self.prev_yaw) / self.dt
        vyaw_desired = Kp_yaw * yaw_error - Kd_yaw * yaw_rate
        vyaw_desired = 0.12
        self.prev_yaw = yaw
        
        # Unpack current command for observation
        ctrl_input = np.array([*self.ctrl_list[0]], dtype=np.float32)
        assert ctrl_input.shape == (2,), f"Expected shape (2,), got {ctrl_input.shape}"
        
        
        # Set the desired velocity and yaw rate: next action
        raw_action = np.array([vx_desired, vyaw_desired], dtype=np.float32)

        # Load lidar representation and nearest obstacle heading information
        state_vec72 = try_load_vec72()

        self.prev_heading = self.curr_heading
        calib_yaw = yaw - self.offset #calibrated angle helps us with clearance
        self.curr_heading = get_safe_extreme(state_vec72[0:36], calib_yaw)
        if self.prev_heading is None: self.prev_heading = self.curr_heading
        
        # Define full observation state as a concatenation of the current command, curent and past headings and the state vector
        full_state = np.concatenate([ctrl_input, self.curr_heading, self.prev_heading, state_vec72])
        # assert 78
        # Monitoring, Intervention, and Fallback: Safety Filter with current observation and action
        safe_action = self.safety_enforcer.get_action(full_state, raw_action)
        assert len(safe_action) == 2 
        safe_action = get_control(safe_action)
        
        # Imput vy information (0.0) for compatibility with the robot sdk
        # Robot training happens in u,v frame u along right axis and v along forward axis both local to robot
        # However sdk has x in forward direction and y in right ward direction
        # (vx,vy,vyaw){sdk} --> (v, -u, w){robot training frame}


        # Nevermind success model has correct alignmnet trains in y-direction so we can use first entry for vx
        safe_action = (.8) *  np.array([safe_action[0], 0.0, safe_action[1]], dtype=np.float32)
        
        # If visualization is enabled, save the state vector
        # Use atomic file writing to avoid data corruption
        if self.viz_valid:
            viz_tmp_path = VIZ_PATH + ".tmp"
            # Plotting in SF pipeline definitely needs calibrated yaw
            viz_state = np.concatenate([np.array([calib_yaw]), raw_action, full_state], axis=0)
            with open(viz_tmp_path, 'wb') as f:
                np.save(f, viz_state)
                f.flush()
                os.fsync(f.fileno())
            os.replace(viz_tmp_path, VIZ_PATH)

        #Robot has a separate vy for strating but currently unused
        print(yaw, "yaw", calib_yaw, "calib_yaw", yaw_error, "error", yaw_rate, "rate", vyaw_desired,"vyaw")
        self.client.Move(*safe_action.tolist())
        print("safe_action", safe_action)
        # Clear the previous command and append the current command for observation
        self.ctrl_list.clear()
        self.ctrl_list.append((safe_action[0], safe_action[2]))
        print(f"[Shielded: {self.safety_enforcer.get_shielding_status()} | Q: {self.safety_enforcer.prev_q:.3f}]")

    def update(self, robot_state: SportModeState_):
        self.t += self.dt
        self.safe_forward_control(robot_state)

# === Robot state subscription ===
robot_state = unitree_go_msg_dds__SportModeState_()
 
def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg

def main(args=None):


    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    walker = SportModeSafeWalker()
    walker.GetInitState(robot_state)
    walker.StandAndStart()

    while True:
        walker.update(robot_state)
        time.sleep(walker.dt)
        if walker.safety_enforcer.prev_q <= .15: raise KeyboardInterrupt
if __name__ == '__main__':
    main()
