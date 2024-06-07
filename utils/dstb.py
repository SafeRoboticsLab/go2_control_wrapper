# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Dict, Union
import numpy as np
import torch

from simulators import BasePolicy


class DummyPolicy(BasePolicy):
  policy_type = "dummy"

  def __init__(self, id: str, action_dim: int) -> None:
    super().__init__(id)
    self.action_dim = action_dim

  @property
  def is_stochastic(self) -> bool:
    return False

  def get_action(
      self, obsrv: Union[np.ndarray, torch.Tensor], agents_action: Optional[Dict[str, np.ndarray]] = None,
      num: Optional[int] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    if num is None:
      action = np.zeros(self.action_dim)
    else:
      action = np.zeros(shape=(num, self.action_dim))

    return action, dict(t_process=0, status=1)


# def dummy_dstb(obsrv, ctrl, append=None, dim: int = 0, **kwargs):
#   return np.zeros(dim)

# def random_dstb(obsrv, ctrl, **kwargs):
#   rng: np.random.Generator = kwargs.get("rng")
#   dstb_range: np.ndarray = kwargs.get("dstb_range")
#   return rng.uniform(low=dstb_range[:, 0], high=dstb_range[:, 1])

# def adversary_dstb(
#     obsrv: np.ndarray, ctrl: np.ndarray, dstb_policy: BasePolicy, append: Optional[np.ndarray] = None,
#     use_ctrl: bool = True, **kwargs
# ) -> np.ndarray:
#   if use_ctrl:
#     obsrv_dstb = np.concatenate((obsrv, ctrl), axis=-1)
#   else:
#     obsrv_dstb = obsrv
#   dstb = dstb_policy.get_action(obsrv=obsrv_dstb, append=append)[0]
#   assert isinstance(dstb, np.ndarray)
#   return dstb
