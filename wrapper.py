import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import Thread

class Wrapper:
  def __init__(self):
    self.PosStopF = 2.146e9
    self.VelStopF = 16000.0
    self.crc = CRC()
    ChannelFactoryInitialize(0)
    self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.pub.Init()

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
    
    self.kp = [65.0] * 12
    self.kd = [5.0] * 12
    self.order = ["FR", "FL", "BR", "BL"]
  
  def update(self, pos):
    """
      _0: hip; _1: thigh; _2: calf
      order: [FR, FL, RR, RL]

      FR_0, FR_1, FR_2,
      FL_0, FL_1, FL_2,
      RR_0, RR_1, RR_2,
      RL_0, RL_1, RL_2
    """
    for i, p in enumerate(pos):
      self.cmd.motor_cmd[i].q = p # position
      self.cmd.motor_cmd[i].kp = self.kp[i]
      self.cmd.motor_cmd[i].dq = 0.0  # Set to stop angular velocity(rad/s)
      self.cmd.motor_cmd[i].kd = self.kd[i]
      self.cmd.motor_cmd[i].tau = 0.0  # target torque (N.m)

    self.cmd.crc = self.crc.Crc(self.cmd)
  
    self.pub.Write(self.cmd)