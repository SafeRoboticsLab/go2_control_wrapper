# --------------------------------------------------------
# Neural network building blocks for ISAACS MuJoCo models.
# Ported from safe_adaptation_dev/agent/neural_network.py
# --------------------------------------------------------

from collections import OrderedDict
import torch
import torch.nn as nn


class Sin(nn.Module):
    """Element-wise sin activation."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


activation_dict = nn.ModuleDict({
    "ReLU": nn.ReLU(),
    "ELU": nn.ELU(),
    "Tanh": nn.Tanh(),
    "Sin": Sin(),
    "Identity": nn.Identity()
})


class MLP(nn.Module):
    """Fully-connected neural network with flexible depth, width, and activation."""

    def __init__(self, dim_list: list, activation_type: str = 'Tanh',
                 out_activation_type: str = 'Identity', verbose: bool = False):
        super().__init__()
        self.moduleList = nn.ModuleList()
        numLayer = len(dim_list) - 1
        for idx in range(numLayer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            linear_layer = nn.Linear(i_dim, o_dim)

            if idx == numLayer - 1:
                # Output layer
                module = nn.Sequential(OrderedDict([
                    ('linear_1', linear_layer),
                    ('act_1', activation_dict[out_activation_type]),
                ]))
            else:
                # Hidden layer
                module = nn.Sequential(OrderedDict([
                    ('linear_1', linear_layer),
                    ('act_1', activation_dict[activation_type]),
                ]))
            self.moduleList.append(module)

        if verbose:
            print(self.moduleList)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.moduleList:
            x = m(x)
        return x
