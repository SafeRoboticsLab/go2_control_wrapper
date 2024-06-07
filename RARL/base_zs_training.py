# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu (kaichieh@princeton.edu)
#          Duy P. Nguyen (duyn@princeton.edu)

from collections import namedtuple
import torch

from .base_training import BaseTraining

TransitionZS = namedtuple(
    'TransitionZS', ['s', 'u', 'd', 'r', 's_', 'done', 'info']
)


class BaseZeroSumTraining(BaseTraining):

  def __init__(self, CONFIG, CONFIG_ENV, CONFIG_UPDATE):
    super().__init__(CONFIG, CONFIG_ENV, CONFIG_UPDATE)
    self.transition_cls = TransitionZS

  def unpack_batch(  #* Overrides.
      self, batch, get_append=False, get_latent=False, get_perf_action=False
  ):
    non_final_mask = torch.tensor(
        tuple(map(lambda s: not s, batch.done)), dtype=torch.bool
    ).view(-1).to(self.device)
    non_final_state_nxt = torch.cat([
        s for done, s in zip(batch.done, batch.s_) if not done
    ]).to(self.device)
    state = torch.cat(batch.s).to(self.device)

    reward = torch.FloatTensor(batch.r).to(self.device)

    g_x = torch.FloatTensor([info['g_x'] for info in batch.info]
                           ).to(self.device).view(-1)
    l_x = torch.FloatTensor([info['l_x'] for info in batch.info]
                           ).to(self.device).view(-1)

    if get_perf_action:  # recovery RL separates a_shield and a_perf.
      if batch.info[0]['a_perf'].dim() == 1:
        ctrl = torch.FloatTensor([info['a_perf'] for info in batch.info])
      else:
        ctrl = torch.cat([info['a_perf'] for info in batch.info])
      ctrl = ctrl.to(self.device)
    else:
      ctrl = torch.cat(batch.u).to(self.device)

    dstb = torch.cat(batch.d).to(self.device)

    ctrl = ctrl.float()
    dstb = dstb.float()
    state = state.float()
    non_final_state_nxt = non_final_state_nxt.float()

    latent = None
    if get_latent:
      latent = torch.cat(batch.z).to(self.device)

    append = None
    non_final_append_nxt = None
    if get_append:
      append = torch.cat([info['append'] for info in batch.info]
                        ).to(self.device)
      non_final_append_nxt = torch.cat([
          info['append_nxt'] for info in batch.info
      ]).to(self.device)[non_final_mask]

    binary_cost = torch.FloatTensor([
        info['binary_cost'] for info in batch.info
    ])
    binary_cost = binary_cost.to(self.device).view(-1)

    return (
        non_final_mask, non_final_state_nxt, state, ctrl, dstb, reward, g_x,
        l_x, latent, append, non_final_append_nxt, binary_cost
    )
