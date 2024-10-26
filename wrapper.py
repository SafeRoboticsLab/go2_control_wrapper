import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread

from utils.velocity import *

class Wrapper:
  def __init__(self):
    self.PosStopF = 2.146e9
    self.VelStopF = 16000.0
    self.crc = CRC()
    ChannelFactoryInitialize(0)
    self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.sub = ChannelSubscriber("rt/lowstate", LowState_)
    self.pub.Init()
    self.sub.Init(self.LowStateHandler, 1)

    self.cmd = unitree_go_msg_dds__LowCmd_()
    self.cmd.head[0] = 0xFE
    self.cmd.head[1] = 0xEF
    self.cmd.level_flag = 0xFF
    self.cmd.gpio = 0
    for i in range(20):
        self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        self.cmd.motor_cmd[i].q = self.PosStopF
        self.cmd.motor_cmd[i].kp = 0
        self.cmd.motor_cmd[i].dq = self.VelStopF
        self.cmd.motor_cmd[i].kd = 0
        self.cmd.motor_cmd[i].tau = 0
    
    # self.kp = [60.0] * 12
    # self.kd = [5.0] * 12
    self.kp = [50.0] * 12
    self.kd = [3.0] * 12
    self.order = ["FR", "FL", "BR", "BL"] # actual output of Go2, DO NOT EDIT

    self.state = [0] * 36 # x_dot, y_dot, z_dot, roll, pitch, w_x, w_y, w_z, joint_pos x 12, joint_vel x 12
    
    self.msgs = []
    self.last_time = None
  
  def LowStateHandler(self, msg: LowState_):
    if self.last_time is None:
      self.last_time = time.time()

    # calculating v
    self.msgs.append(msg)

    if len(self.msgs) > 10: # 20
      while len(self.msgs) > 10: # 20
        self.msgs.pop(0)
    
    # extrapolate for v
    # v = integrate_velocity(self.msgs)
    
    # use a dummy way to calculate v
    accelerometer = np.array(msg.imu_state.accelerometer)
    v = accelerometer * (time.time() - self.last_time)
    self.last_time = time.time()

    joint_pos = [msg.motor_state[i].q for i in range(12)]
    joint_vel = [msg.motor_state[i].dq for i in range(12)]
    gyroscope = msg.imu_state.gyroscope
    # accelerometer = msg.imu_state.accelerometer
    # quaternion = msg.imu_state.quaternion # w, x, y, z
    rpy = msg.imu_state.rpy

    # map this to sim_order: ["FL", "BL", "FR", "BR"]
    contact = [1 if x > 8 else 0 for x in msg.foot_force]
    foot_force = [contact[2], contact[0], contact[3], contact[1]]
    self.state = list(tuple(v) + tuple(rpy[:2]) + tuple(gyroscope) + tuple(joint_pos) + tuple(joint_vel) + tuple(foot_force))

  def map(self, pos, current_order, new_order):
    mapped_pos = [0.0] * 12
    # the input_order is different from the self.order, remap before writing to the robot
    for i, o in enumerate(current_order):
      index = new_order.index(o)
      for j in range(3):
        mapped_pos[index*3 + j] = pos[i*3 + j]
    return mapped_pos

  def update(self, pos, input_order=None):
    """
      _0: hip; _1: thigh; _2: calf
      order: [FR, FL, RR, RL]

      FR_0, FR_1, FR_2,
      FL_0, FL_1, FL_2,
      RR_0, RR_1, RR_2,
      RL_0, RL_1, RL_2
    """
    if input_order is not None:
      mapped_pos = self.map(pos, input_order, self.order)
    else:
      mapped_pos = pos

    for i, p in enumerate(mapped_pos):
      self.cmd.motor_cmd[i].q = p # position
      self.cmd.motor_cmd[i].kp = self.kp[i]
      self.cmd.motor_cmd[i].dq = 0.0  # Set to stop angular velocity(rad/s)
      self.cmd.motor_cmd[i].kd = self.kd[i]
      self.cmd.motor_cmd[i].tau = 0.0  # target torque (N.m)

    self.cmd.crc = self.crc.Crc(self.cmd)
  
    self.pub.Write(self.cmd)