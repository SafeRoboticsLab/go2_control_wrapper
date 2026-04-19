# --------------------------------------------------------
# ISAACS MuJoCo model classes for deployment.
# Ported from safe_adaptation_dev/agent/model.py
#
# Only includes forward/inference paths (no training code).
# --------------------------------------------------------

from typing import Optional, Union, Tuple, List
import copy
import numpy as np
import torch
import torch.nn as nn

from .neural_network import MLP


def get_mlp_input(
    obsrv: Union[np.ndarray, torch.Tensor],
    action: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, bool, int]:
    """Transform inputs to torch Tensor. Concatenate action if provided."""
    np_input = False
    if isinstance(obsrv, np.ndarray):
        obsrv = torch.FloatTensor(obsrv).to(device)
        np_input = True
    else:
        obsrv = obsrv.to(device)
    if action is not None:
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        else:
            action = action.to(device)

    num_extra_dim = 0
    if obsrv.dim() == 1:
        obsrv = obsrv.unsqueeze(0)
        if action is not None:
            action = action.unsqueeze(0)
        num_extra_dim += 1

    if action is not None:
        obsrv = torch.cat((obsrv, action), dim=-1)

    return obsrv, np_input, num_extra_dim


class TwinnedQNetwork(nn.Module):
    """Twin Q-networks for critic."""

    def __init__(self, obsrv_dim: int, mlp_dim: List[int], action_dim: int,
                 append_dim: int = 0, latent_dim: int = 0,
                 activation_type: str = 'Tanh',
                 device: Union[str, torch.device] = 'cpu',
                 verbose: bool = True):
        super().__init__()
        if verbose:
            print("CRITIC architecture:")
        dim_list = [obsrv_dim + action_dim + append_dim + latent_dim] + list(mlp_dim) + [1]
        self.Q1 = MLP(dim_list, activation_type, verbose=verbose).to(device)
        self.Q2 = copy.deepcopy(self.Q1)
        self.device = torch.device(device)

    def forward(self, obsrv, action):
        obsrv, np_input, num_extra_dim = get_mlp_input(obsrv, action=action, device=self.device)
        q1 = self.Q1(obsrv)
        q2 = self.Q2(obsrv)

        for _ in range(num_extra_dim):
            q1 = q1.squeeze(0)
            q2 = q2.squeeze(0)

        if np_input:
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()
        return q1, q2


class GaussianPolicy(nn.Module):
    """Gaussian policy network (deterministic forward for deployment)."""

    def __init__(self, obsrv_dim: int, mlp_dim: List[int], action_dim: int,
                 action_range: np.ndarray, append_dim: int = 0,
                 latent_dim: int = 0, activation_type: str = 'Tanh',
                 device: Union[str, torch.device] = 'cpu',
                 verbose: bool = True):
        super().__init__()
        self.obsrv_dim = obsrv_dim
        dim_list = [obsrv_dim + append_dim + latent_dim] + list(mlp_dim) + [action_dim]
        self.device = torch.device(device)

        if verbose:
            print("ACTOR (mean) architecture:")
        self.mean = MLP(dim_list, activation_type, out_activation_type="Identity",
                        verbose=verbose).to(device)
        # log_std network exists but not used for deterministic inference
        self.log_std = MLP(dim_list, activation_type, out_activation_type="Identity",
                           verbose=False).to(device)

        if isinstance(action_range, np.ndarray):
            action_range = torch.FloatTensor(action_range).to(self.device)
        if action_range.dim() == 1:
            action_range = action_range.unsqueeze(0)

        self.a_max = action_range[:, 1]
        self.a_min = action_range[:, 0]
        self.scale = (self.a_max - self.a_min) / 2.0
        self.bias = (self.a_max + self.a_min) / 2.0

    def forward(self, obsrv, action=None):
        """Deterministic forward: obs -> scaled action."""
        obsrv, np_input, num_extra_dim = get_mlp_input(
            obsrv, action=action, device=self.device)
        output = self.mean(obsrv)
        output = torch.tanh(output)
        output = output * self.scale + self.bias

        for _ in range(num_extra_dim):
            output = output.squeeze(0)

        if np_input:
            output = output.detach().cpu().numpy()
        return output

    def to(self, device):
        super().to(device)
        self.device = device
        self.a_max = self.a_max.to(device)
        self.a_min = self.a_min.to(device)
        self.scale = self.scale.to(device)
        self.bias = self.bias.to(device)
