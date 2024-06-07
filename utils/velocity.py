import numpy as np
from scipy.integrate import cumtrapz

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def integrate_velocity(messages):
    times = [0.001 * msg.tick for msg in messages]
    acc_world = []

    for msg in messages:
      R = quaternion_to_rotation_matrix(*msg.imu_state.quaternion)
      acc_world_frame = R @ np.array(msg.imu_state.accelerometer)
      acc_world_frame_corrected = acc_world_frame - np.array([0, 0, 9.81])
      acc_world.append(acc_world_frame_corrected)

    acc_world = np.array(acc_world)
    times = np.array(times)

    vel_x = cumtrapz(acc_world[:, 0], times, initial=0)
    vel_y = cumtrapz(acc_world[:, 1], times, initial=0)
    vel_z = cumtrapz(acc_world[:, 2], times, initial=0)

    # velocities = [{"time": t, "velocity": {"x": vx, "y": vy, "z": vz}} for t, vx, vy, vz in zip(times, vel_x, vel_y, vel_z)]
    return [vel_x[-1], vel_y[-1], vel_z[-1]]