import socket
import pybullet as p
import struct
import copy
from omegaconf import OmegaConf
from agent import ISAACS
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from rl_controller.rl_controller import Go2RLController
from safe_adaptation_dev.simulators import SpiritPybulletZeroSumEnv, Go2PybulletZeroSumEnv
from utils.utils import get_model_index
import pybullet as p
import time
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation

def get_state(env, command=[0, 0, 0]):
    # 36D, excluding previous action
    pos, ang = p.getBasePositionAndOrientation(env.agent.dyn.robot.id, physicsClientId=env.agent.dyn.robot.client)
    rotmat = Rotation.from_quat(ang).as_matrix()
    # ang = p.getEulerFromQuaternion(ang, physicsClientId=robot.client)
    linear_vel, angular_vel = p.getBaseVelocity(env.agent.dyn.robot.id, physicsClientId=env.agent.dyn.robot.client)
    robot_body_linear_vel =  (np.linalg.inv(rotmat) @ np.array(linear_vel).T)
    robot_body_angular_vel = (np.linalg.inv(rotmat) @ np.array(angular_vel).T)
    joint_pos, joint_vel, joint_force, joint_torque = env.agent.dyn.robot.get_joint_state()
    projected_gravity = (np.linalg.inv(rotmat) @ np.array([0, 0, -1]).T)
    obs = (
        tuple(robot_body_linear_vel) + 
        tuple(robot_body_angular_vel) +
        tuple(projected_gravity) + 
        tuple(command) +
        tuple(joint_pos) +
        tuple(joint_vel)
    )
    return torch.Tensor(obs)

HOST = '192.168.0.248'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)

        # config_file = "train_result/test_go2/test_isaacs_centerSampling_withContact/config_new.yaml"
        # config_file = "train_result/test_go2/test_isaacs_postCoRL_arbitraryGx/config_new.yaml"
        config_file = "train_result/test_go2/go2_corldemo_tgda_richURDF_1/config_new.yaml"

        # Loads config.
        cfg = OmegaConf.load(config_file)

        if cfg.agent.dyn == "SpiritPybullet":
            env_class = SpiritPybulletZeroSumEnv
        elif cfg.agent.dyn == "Go2Pybullet":
            env_class = Go2PybulletZeroSumEnv
        else:
            raise ValueError("Dynamics type not supported!")

        # Constructs environment.
        print("\n== Environment information ==")

        env = env_class(cfg.environment, cfg.agent, None)

        # Constructs solver.
        print("\n== Solver information ==")
        solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
        env.agent.policy = copy.deepcopy(solver.ctrl)
        print('#params in ctrl: {}'.format(
            sum(p.numel() for p in solver.ctrl.net.parameters()
                if p.requires_grad)))
        print('#params in dstb: {}'.format(
            sum(p.numel() for p in solver.dstb.net.parameters()
                if p.requires_grad)))
        print('#params in critic: {}'.format(
            sum(p.numel() for p in solver.critic.net.parameters()
                if p.requires_grad)))
        print("We want to use: {}, and Agent uses: {}".format(
            cfg.solver.device, solver.device))
        print("Critic is using cuda: ",
              next(solver.critic.net.parameters()).is_cuda)

        ## RESTORE PREVIOUS RUN
        print("\nRestore model information")
        ## load ctrl and critic
        dstb_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type[1],
                                                cfg.eval.step[1],
                                                type="dstb",
                                                autocutoff=0.9)

        ctrl_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type[0],
                                                cfg.eval.step[0],
                                                type="ctrl",
                                                autocutoff=0.9)

        solver.ctrl.restore(ctrl_step, model_path)
        solver.dstb.restore(dstb_step, model_path)
        solver.critic.restore(ctrl_step, model_path)

        prev_info = {"g_x": np.inf, "l_x": np.inf}
        prev_done = True
        L_horizon = 10
        horizon = 100
        # controller = InverseKinematicsController(Xdist=0.387,
        #                                          Ydist=0.284,
        #                                          height=0.25,
        #                                          coxa=0.03,
        #                                          femur=0.2,
        #                                          tibia=0.2,
        #                                          L=2.0,
        #                                          angle=0,
        #                                          T=0.4,
        #                                          dt=0.02)
        controller = Go2RLController()
        command = [0.4, 0.0, -0.15]

        while True:
            data = conn.recv(1024)
            # struct data includes state (36) + proposed action (12) = 48
            struct_data = np.array(struct.unpack("!48f", data[-192:]))
            state = struct_data[:36]
            action = struct_data[36:]

            # evaluate
            s = env.reset(cast_torch=True,
                          initial_state=state,
                          initial_action=action)
            counter = 0
            while True:
                if L_horizon is None or L_horizon == 1:
                    # already apply initial_action in reset
                    u = solver.ctrl.net(s.float().to(solver.device))
                else:
                    if counter // L_horizon == 0:
                        # apply prev result
                        if prev_info["g_x"] < 0 or prev_info["l_x"] < 0:
                            # prev gameplay failed, run shielding for L steps
                            # if min(env.agent.dyn.robot.target_margin().values()) > -0.1:
                            #     u = torch.FloatTensor(np.array([
                            #         0.5, 0.7, -1.5, 0.5, 0.7, -1.2, -0.5, 0.7, -1.5, -0.5, 0.7, -1.2
                            #     ]) - np.array(env.agent.dyn.robot.get_joint_position())).to(solver.device)
                            # else:
                            #     u = solver.ctrl.net(s.float().to(solver.device))
                            u = solver.ctrl.net(s.float().to(solver.device))
                        else:
                            # prev gameplay is successful, run task
                            # task policy
                            # new_joint_pos = controller.get_action(
                            #     joint_order=["FL", "BL", "FR", "BR"],
                            #     offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            
                            # rl controller
                            new_joint_pos = controller.get_action(get_state(env, command=command))

                            u = torch.FloatTensor(new_joint_pos - np.array(env.agent.dyn.robot.get_joint_position())).to(solver.device)
                    elif counter // L_horizon == 1:
                        # candidate - task policy
                        # new_joint_pos = controller.get_action(
                        #     joint_order=["FL", "BL", "FR", "BR"],
                        #     offset=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                        new_joint_pos = controller.get_action(get_state(env, command=command))
                        u = torch.FloatTensor(new_joint_pos - np.array(env.agent.dyn.robot.get_joint_position())).to(solver.device)
                    else:
                        # back to shielding
                        u = solver.ctrl.net(s.float().to(solver.device))

                s_dstb = [s.float().to(solver.device)]
                if cfg.agent.obsrv_list.dstb is not None:
                    for i in cfg.agent.obsrv_list.dstb:
                        if i == "ctrl":
                            s_dstb.append(u)
                d = solver.dstb.net(*s_dstb)
                # critic_q = max(
                #     solver.critic.net(s.float().to(solver.device),
                #                       solver.combine_action(u, d)))
                a = {'ctrl': u.detach().numpy(), 'dstb': d.detach().numpy()}
                s_, r, done, info = env.step(a, cast_torch=True)
                s = s_
                counter += 1

                if counter > horizon or done:
                    break

            resp = {
                "done": done,
                "done_type": info["done_type"],
                "g_x": info["g_x"],
                "l_x": info["l_x"]
            }
            prev_info = info
            prev_done = done
            conn.sendall(bytes(json.dumps(resp), encoding="utf-8"))
