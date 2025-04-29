import time
import sys
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

from SEP import SafetyEnforcer as safety_enforcer

DECAY_TIME = 3.0
ROBOT_RADIUS = 0.85
INTENSITY_THRESHOLD = 220


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


class SportModeSafeWalker(Node):
    def __init__(self):
        super().__init__('safe_walker_node')

        self.dt = 0.01
        self.t = 0
        self.yaw0 = 0
        self.prev_yaw = 0
        self.px0 = 0
        self.py0 = 0

        self.lidar_max_range = 6.0
        self.obs_sequence_len = 2
        self.angle_sequence_len = 2
        self.obs_sequence = deque(maxlen=2)
        self.angle_sequence = deque(maxlen=2)
        self.latest_lidar_xy = np.zeros((18, 2), dtype=np.float32)

        self.buffer = []
        self.robot_pos = None
        self.robot_yaw = 0.0

        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.safety_enforcer = safety_enforcer(
            epsilon=0.5,
            shield_type="value",
            parent_dir=""
        )

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.create_subscription(PointCloud2, "/utlidar/cloud", self.lidar_callback, 10)
        self.create_subscription(PoseStamped, "/odom_pose", self.pose_callback, 10)

        self.robot_state = unitree_go_msg_dds__SportModeState_()
        self.robot_state_ready = False

    def pose_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.robot_pos = np.array([pos.x, pos.y])
        self.robot_yaw = yaw_from_quaternion(ori.x, ori.y, ori.z, ori.w)

    def lidar_callback(self, msg: PointCloud2):
        if self.robot_pos is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        points = np.array([
            [p[0], p[1], p[3]] for p in pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        ])

        xy = points[:, :2]
        intensity = points[:, 2]

        shifted = xy - self.robot_pos
        c, s = np.cos(-self.robot_yaw), np.sin(-self.robot_yaw)
        rot_matrix = np.array([[c, -s], [s, c]])
        aligned = shifted @ rot_matrix.T
        aligned_points = np.hstack((aligned, intensity[:, None]))

        self.buffer.append((now, aligned_points))
        self.buffer = [(t, pts) for (t, pts) in self.buffer if now - t <= DECAY_TIME]

        all_points = np.vstack([pts for (_, pts) in self.buffer])
        xy = all_points[:, :2]
        intensity = all_points[:, 2]
        dists = np.linalg.norm(xy, axis=1)

        mask = (dists > ROBOT_RADIUS) | ((dists <= ROBOT_RADIUS) & (intensity > INTENSITY_THRESHOLD))
        filtered_points = xy[mask]

        if len(filtered_points) >= 18:
            closest = np.argsort(np.linalg.norm(filtered_points, axis=1))[:18]
            self.latest_lidar_xy = filtered_points[closest].astype(np.float32)
        else:
            self.latest_lidar_xy = np.zeros((18, 2), dtype=np.float32)

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]
        self.prev_yaw = self.yaw0
        print(self.yaw0, self.robot_yaw)

    def StandAndStart(self):
        self.client.StandUp()
        time.sleep(1)
        self.client.BalanceStand()
        print("Standing complete.")

    def encode_state(self, control: np.ndarray, lidar_obs: np.ndarray) -> np.ndarray:
        lidar_xy = lidar_obs.reshape(18, 2)
        dists = np.linalg.norm(lidar_xy, axis=1)
        closest_idx = np.argmin(dists)
        closest_vec = lidar_xy[closest_idx]
        escape_vec = -closest_vec
        escape_angle = np.arctan2(escape_vec[1], escape_vec[0])
        angle_diff = np.arctan2(math.sin(self.prev_yaw - escape_angle), math.cos(self.prev_yaw - escape_angle))

        angle_obs = np.array([np.cos(angle_diff), np.sin(angle_diff)])
        self.angle_sequence.appendleft(angle_obs)
        while len(self.angle_sequence) < self.angle_sequence_len:
            self.angle_sequence.append(angle_obs)
        angle_seq = np.concatenate(list(self.angle_sequence), axis=0)

        flat_lidar = lidar_obs.flatten()
        self.obs_sequence.appendleft(flat_lidar)
        while len(self.obs_sequence) < self.obs_sequence_len:
            self.obs_sequence.append(flat_lidar)
        lidar_seq = np.concatenate(list(self.obs_sequence), axis=0)

        return np.concatenate([control[:2], angle_seq, lidar_seq], axis=0)

    def safe_forward_control(self, robot_state: SportModeState_):
        vx_desired = 0.5
        #No strafing
        #vy_desired = 0.0

        yaw = robot_state.imu_state.rpy[2]
        yaw_error = math.atan2(math.sin(self.yaw0 - yaw), math.cos(self.yaw0 - yaw))

        Kp_yaw = 2.0
        Kd_yaw = 0.5
        yaw_rate = (yaw - self.prev_yaw) / self.dt
        vyaw_desired = Kp_yaw * yaw_error - Kd_yaw * yaw_rate
        vyaw_desired = max(min(vyaw_desired, 1.0), -1.0)
        self.prev_yaw = yaw

        ctrl_input = np.array([vx_desired, vyaw_desired], dtype=np.float32)
        #raw_action = np.array([vx_desired, vy_desired, vyaw_desired], dtype=np.float32)
        raw_action = ctrl_input

        state = self.encode_state(ctrl_input, self.latest_lidar_xy)
        safe_action = self.safety_enforcer.get_action(state, raw_action)

        self.client.Move(*safe_action.tolist())
        print(f"[Shielded: {self.safety_enforcer.get_shielding_status()} | Q: {self.safety_enforcer.prev_q:.3f}]")

    def timer_callback(self):
        if self.robot_state_ready:
            self.t += self.dt
            self.safe_forward_control(self.robot_state)


class SafeWalkerApp:
    def __init__(self):
        self.robot_state = None
        self.robot_state_ready = False

        # DDS must be initialized before anything else
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])
        else:
            ChannelFactoryInitialize(0)

        # Subscribe to Go2's sportmodestate
        self.sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sub.Init(self.high_state_handler, 10)

    def high_state_handler(self, msg: SportModeState_):
        self.robot_state = msg
        self.robot_state_ready = True

    def run(self):
        # Wait until the first robot_state message is received
        print("⏳ Waiting for initial robot state...")
        timeout = time.time() + 5.0
        while not self.robot_state_ready:
            if time.time() > timeout:
                print("❌ Timed out waiting for robot state.")
                return
            time.sleep(0.05)

        # Now safe to launch ROS 2 node
        rclpy.init()
        walker = SportModeSafeWalker()
        walker.GetInitState(self.robot_state)
        walker.StandAndStart()
        rclpy.spin(walker)

        walker.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    app = SafeWalkerApp()
    app.run()
