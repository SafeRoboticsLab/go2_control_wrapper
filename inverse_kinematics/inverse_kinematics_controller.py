import numpy as np
from .kinematic_model import robotKinematics
from .gaitPlanner import trotGait


class InverseKinematicsController():

  def __init__(
      self, dt=1. / 125., Xdist=0.36, Ydist=0.35, height=0.3, Lrot=0, angle=180, L=1.2, T=0.5,
      offset=[0.5, 0.0, 0.0, 0.5], coxa=0.022, femur=0.206, tibia=0.206
  ):
    """
        Initialize an inverse kinematics controller with step size of 0.002
        """
    """
        # Xdist = 0.39
        Xdist=0.36
        # Ydist = 0.284
        Ydist=0.35 # tried 0.35 measured on the real robot
        height = 0.3
        """

    self.Lrot = Lrot
    self.angle = angle
    self.L = L
    self.T = T

    self.dt = dt

    self.offset = np.array(offset)

    self.bodytoFeet0 = np.matrix([[Xdist / 2, -Ydist / 2, -height], [Xdist / 2, Ydist / 2, -height],
                                  [-Xdist / 2, -Ydist / 2, -height], [-Xdist / 2, Ydist / 2, -height]])

    self.robotKinematics = robotKinematics(L=Xdist, W=Ydist, height=height, coxa=coxa, femur=femur, tibia=tibia)
    self.trot = trotGait(dt=dt)

  def get_action(self, **kwargs):
    """
        Return the next joint positions of the inverse kinematics controller
        """
    bodytoFeet = self.trot.loop(self.L, self.angle, self.Lrot, self.T, self.offset, self.bodytoFeet0)
    FR_angles, FL_angles, BR_angles, BL_angles, _ = self.robotKinematics.solve(
        np.zeros([3]), np.zeros([3]), bodytoFeet
    )

    joint_order = ["FL", "BL", "FR", "BR"]
    offset = np.array([0, 0, np.pi, 0, 0, np.pi, 0, 0, np.pi, 0, 0, np.pi])

    if "joint_order" in kwargs.keys():
      joint_order = kwargs["joint_order"]
    if "offset" in kwargs.keys():
      offset = np.array(kwargs["offset"])

    result = np.array([])
    for joint_type in joint_order:
      if joint_type == "FL":
        result = np.concatenate((result, np.array(FL_angles)))
      elif joint_type == "BL":
        result = np.concatenate((result, np.array(BL_angles)))
      elif joint_type == "FR":
        result = np.concatenate((result, np.array(FR_angles)))
      elif joint_type == "BR":
        result = np.concatenate((result, np.array(BR_angles)))

    return result + offset
