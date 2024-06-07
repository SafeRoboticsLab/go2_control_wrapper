# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional
import os
import glob
import torch


def get_bellman_update(
    mode: str, batch_size: int, q1_nxt: torch.Tensor, q2_nxt: torch.Tensor, non_final_mask: torch.Tensor,
    reward: torch.Tensor, g_x: torch.Tensor, l_x: torch.Tensor, binary_cost: torch.Tensor, gamma: float,
    terminal_type: Optional[str] = None
):
  # Conservative target Q values: if the control policy we want to learn is to maximize, we take the minimum of the two
  # Q values. Otherwise, we take the maximum.
  if mode == 'risk':
    target_q = torch.max(q1_nxt, q2_nxt).view(-1)
  elif (mode == 'reach-avoid' or mode == 'safety' or mode == 'performance'):
    target_q = torch.min(q1_nxt, q2_nxt).view(-1)
  else:
    raise ValueError("Unsupported RL mode.")

  y = torch.zeros(batch_size).float().to(q1_nxt)  # placeholder
  final_mask = torch.logical_not(non_final_mask)
  if mode == 'reach-avoid':
    # V(s) = min{ g(s), max{ l(s), V(s') }}
    # Q(s, u) = V( f(s,u) ) = main g(s'), max{ ell(s'), min_{u'} Q(s', u')}}
    terminal_target = torch.min(l_x[non_final_mask], g_x[non_final_mask])
    original_target = torch.min(g_x[non_final_mask], torch.max(l_x[non_final_mask], target_q))
    y[non_final_mask] = (1.0-gamma) * terminal_target + gamma*original_target

    if terminal_type == 'g':
      y[final_mask] = g_x[final_mask]
    elif terminal_type == 'all':
      y[final_mask] = torch.min(l_x[final_mask], g_x[final_mask])
    else:
      raise ValueError("invalid terminal type")
  elif mode == 'safety':
    # V(s) = min{ g(s), V(s') }
    # Q(s, u) = V( f(s,u) ) = min{ g(s'), max_{u'} Q(s', u') }
    # normal state
    y[non_final_mask] = ((1.0-gamma) * g_x[non_final_mask] + gamma * torch.min(g_x[non_final_mask], target_q))

    # terminal state
    y[final_mask] = g_x[final_mask]
  elif mode == 'performance':
    y = reward
    y[non_final_mask] += gamma * target_q
  elif mode == 'risk':
    y = binary_cost  # y = 1 if it's a terminal state
    y[non_final_mask] += gamma * target_q
  return y


def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def save_model(model: torch.nn.Module, step: int, model_folder: str, types: str, max_model: Optional[int] = None):
  start = len(types) + 1
  os.makedirs(model_folder, exist_ok=True)
  model_list = glob.glob(os.path.join(model_folder, '*.pth'))

  if max_model is not None:
    if len(model_list) > max_model - 1:
      min_step = min([int(li.split('/')[-1][start:-4]) for li in model_list])
      os.remove(os.path.join(model_folder, '{}-{}.pth'.format(types, min_step)))
  model_path = os.path.join(model_folder, '{}-{}.pth'.format(types, step))
  torch.save(model.state_dict(), model_path)
  # print('  => Saves {} after [{}] updates'.format(model_path, step))


# def remove_model(path: str):
#   if os.path.exists(path):
#     print("  => Remove", path)
#     os.remove(path)
#   else:
#     print(path, "does not exist!")

# def restore_model(
#     model: torch.nn.Module, device: torch.device, step: Optional[int] = None,
#     model_folder: Optional[str] = None, types: Optional[str] = None,
#     model_path: Optional[str] = None, verbose: bool = False
# ):
#   if model_path is None:
#     assert step is not None
#     assert types is not None
#     assert model_folder is not None
#     model_path = os.path.join(model_folder, '{}-{}.pth'.format(types, step))
#   model.load_state_dict(torch.load(model_path, map_location=device))
#   model.to(device)
#   if verbose:
#     print(f'  <= Restore models from {model_path}')
