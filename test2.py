import time
import sys
import math

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

class SportModeForwardPolicy:
    def __init__(self) -> None:
        # Timing
        self.t = 0
        self.dt = 0.01

        # Robot state (initial pose)
        self.px0 = 0
        self.py0 = 0
        self.yaw0 = 0
        self.prev_yaw = 0

        # Connect to SportClient
        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

    def GetInitState(self, robot_state: SportModeState_):
        self.px0 = robot_state.position[0]
        self.py0 = robot_state.position[1]
        self.yaw0 = robot_state.imu_state.rpy[2]
        self.prev_yaw = self.yaw0

    def ForwardFeedbackControl(self, robot_state: SportModeState_):
        """
        PD controller to move forward at constant speed and maintain heading.
        """
        # Desired forward velocity
        vx_desired = 0.5  # m/s
        vy_desired = 0.0

        # Heading error
        yaw = robot_state.imu_state.rpy[2]
        yaw_error = math.atan2(math.sin(self.yaw0 - yaw), math.cos(self.yaw0 - yaw))

        # PD control for yaw
        Kp_yaw = 2.0
        Kd_yaw = 0.5
        yaw_rate = (yaw - self.prev_yaw) / self.dt
        vyaw_command = Kp_yaw * yaw_error - Kd_yaw * yaw_rate
        vyaw_command = max(min(vyaw_command, 1.0), -1.0)

        # Update previous yaw
        self.prev_yaw = yaw

        # Send command
        self.client.Move(vx_desired, vy_desired, vyaw_command)

    def StandAndStart(self):
        """
        Stand up first before starting the walk.
        """
        print("Standing up...")
        self.client.StandUp()
        time.sleep(1)
        self.client.BalanceStand()
        print("Entering feedback control loop")

# === Robot state subscription ===
robot_state = unitree_go_msg_dds__SportModeState_()

def HighStateHandler(msg: SportModeState_):
    global robot_state
    robot_state = msg

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sub.Init(HighStateHandler, 10)
    time.sleep(1)

    test = SportModeForwardPolicy()
    test.GetInitState(robot_state)
    test.StandAndStart()

    print("Starting forward feedback walk...")

    while True:
        test.t += test.dt
        test.ForwardFeedbackControl(robot_state)
        time.sleep(test.dt)
