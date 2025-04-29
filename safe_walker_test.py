import time
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

from unitree_sdk2py.go2.sport.sport_client import SportClient
from SEP import SafetyEnforcer as safety_enforcer  # your safety class
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

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

        self.yaw0 = None
        self.prev_yaw = 0.0

        self.lidar_max_range = 6.0
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

        self.create_subscription(PointCloud2, "/utlidar/cloud", self.lidar_callback, 10)
        self.create_subscription(PoseStamped, "/odom_pose", self.pose_callback, 10)
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.ready = False

    def pose_callback(self, msg: PoseStamped):
        pos = msg.pose.position
        ori = msg.pose.orientation
        self.robot_pos = np.array([pos.x, pos.y])
        self.robot_yaw = yaw_from_quaternion(ori.x, ori.y, ori.z, ori.w)

        if self.yaw0 is None:
            self.yaw0 = self.robot_yaw
            self.prev_yaw = self.robot_yaw
            self.client.StandUp()
            time.sleep(1)
            self.client.BalanceStand()
            print("✅ Robot standing — starting walk.")
            self.ready = True

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
        aligned = shifted @ np.array([[c, -s], [s, c]]).T
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

    def encode_state(self, control: np.ndarray, lidar_obs: np.ndarray) -> np.ndarray:
        lidar_xy = lidar_obs.reshape(18, 2)
        dists = np.linalg.norm(lidar_xy, axis=1)
        closest_vec = lidar_xy[np.argmin(dists)]
        escape_angle = np.arctan2(-closest_vec[1], -closest_vec[0])
        angle_diff = np.arctan2(math.sin(self.robot_yaw - escape_angle), math.cos(self.robot_yaw - escape_angle))

        angle_obs = np.array([np.cos(angle_diff), np.sin(angle_diff)])
        self.angle_sequence.appendleft(angle_obs)
        while len(self.angle_sequence) < 2:
            self.angle_sequence.append(angle_obs)
        angle_seq = np.concatenate(list(self.angle_sequence), axis=0)

        flat_lidar = lidar_obs.flatten()
        self.obs_sequence.appendleft(flat_lidar)
        while len(self.obs_sequence) < 2:
            self.obs_sequence.append(flat_lidar)
        lidar_seq = np.concatenate(list(self.obs_sequence), axis=0)

        return np.concatenate([control[:2], angle_seq, lidar_seq], axis=0)

    def timer_callback(self):
        if not self.ready:
            return

        vx = 0.5
        yaw_error = math.atan2(math.sin(self.yaw0 - self.robot_yaw), math.cos(self.yaw0 - self.robot_yaw))
        vyaw = 2.0 * yaw_error - 0.5 * ((self.robot_yaw - self.prev_yaw) / self.dt)
        vyaw = max(min(vyaw, 1.0), -1.0)
        self.prev_yaw = self.robot_yaw

        ctrl_input = np.array([vx, vyaw], dtype=np.float32)
        state = self.encode_state(ctrl_input, self.latest_lidar_xy)
        action = self.safety_enforcer.get_action(state, ctrl_input)
        self.client.Move(*action.tolist())
        print(f"[vx: {action[0]:.2f}, vyaw: {action[1]:.2f}] | Q = {self.safety_enforcer.prev_q:.3f}")


def main(args=None):
    ChannelFactoryInitialize(0)
    rclpy.init(args=args)
    walker = SportModeSafeWalker()
    rclpy.spin(walker)
    walker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
