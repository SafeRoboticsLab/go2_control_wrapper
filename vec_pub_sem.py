#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os

DECAY_TIME = 3.0
ROBOT_RADIUS = 0.85
INTENSITY_THRESHOLD = 220
VEC_PATH = "/tmp/state_vec_72.npy"

TRANS = np.array([[0, -1], [-1, 0]])


class LidarVec72Writer(Node):
    def __init__(self):
        super().__init__('lidar_vec72_writer')

        self.create_subscription(PointCloud2, '/lidar_chatter', self.lidar_callback, 10)
        self.create_subscription(PointCloud2, '/semantic/pointcloud', self.semantic_callback, 10)

        self.buffer = []  # time-decayed point cloud (LiDAR + semantic)
        self.prev_lidar = np.zeros((18, 2), dtype=np.float32)
        self.curr_lidar = np.zeros((18, 2), dtype=np.float32)


    def semantic_callback(self, msg: PointCloud2):
        self._ingest_pointcloud(msg, use_intensity=False)

    def lidar_callback(self, msg: PointCloud2):
        self._ingest_pointcloud(msg, use_intensity=True)
        self._process()

    def _ingest_pointcloud(self, msg: PointCloud2, use_intensity=True):
        now = self.get_clock().now().nanoseconds * 1e-9

        fields = ("x", "y", "z", "intensity") if use_intensity else ("x", "y", "z")
        try:
            raw = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
            if not raw:
                return
            raw = np.array(raw)
            xy = raw[:, :2]
            aligned = xy @ TRANS.T

            if use_intensity:
                intensity = raw[:, 3]
            else:
                intensity = np.ones((raw.shape[0],), dtype=np.float32) * 255.0

            aligned_points = np.hstack((aligned, intensity[:, None]))
            self.buffer.append((now, aligned_points))
        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse point cloud: {e}")

    def _process(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.buffer = [(t, pts) for (t, pts) in self.buffer if now - t <= DECAY_TIME]

        if not self.buffer:
            return

        all_points = np.vstack([pts for (_, pts) in self.buffer])
        xy = all_points[:, :2]
        intensity = all_points[:, 2]
        dists = np.linalg.norm(xy, axis=1)

        mask = (dists > ROBOT_RADIUS) | ((dists <= ROBOT_RADIUS) & (intensity > INTENSITY_THRESHOLD))
        filtered = xy[mask]

        if len(filtered) < 18:
            filtered = np.vstack((filtered, np.zeros((18 - len(filtered), 2))))
        else:
            idx = np.argsort(np.linalg.norm(filtered, axis=1))[:18]
            filtered = filtered[idx]

        self.prev_lidar = self.curr_lidar
        self.curr_lidar = filtered.astype(np.float32)


        vec72 = np.concatenate([
            self.curr_lidar.flatten(),
            self.prev_lidar.flatten()
        ], axis=0)

        tmp_path = VEC_PATH + ".tmp"
        with open(tmp_path, 'wb') as f:
            np.save(f, vec72)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, VEC_PATH)


def main():
    rclpy.init()
    node = LidarVec72Writer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


