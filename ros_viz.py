import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
import time

VIZ_PATH = "/tmp/state_vec_viz.npy"
FREQ_HZ = 10  # publish frequency

class VecPublisher(Node):
    def __init__(self):
        super().__init__('vec77_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/safe_walker/viz_state', 10)
        self.timer = self.create_timer(1.0 / FREQ_HZ, self.timer_callback)
        self.prev_vec = None

    def timer_callback(self):
        try:
            vec = np.load(VIZ_PATH)
            if vec.shape != (81,): # task(2), yaw (1), full_state (78)
                self.get_logger().warn(f"Ignoring unexpected shape: {vec.shape}")
                return

            if self.prev_vec is not None and np.array_equal(vec, self.prev_vec):
                return  # skip duplicate

            msg = Float32MultiArray()
            msg.data = vec.astype(np.float32).tolist()
            self.publisher_.publish(msg)
            self.prev_vec = vec

        except Exception as e:
            self.get_logger().warn(f"Failed to read vector: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VecPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
