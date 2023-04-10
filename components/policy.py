import torch
import numpy as np
from typing import Any, Dict, Optional

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as


class MeDQNPolicy(DQNPolicy):
  """Implementation of MeDQN(R).
  
  :param torch.nn.Module model: a model following the rules in
    :class:`~tianshou.policy.BasePolicy`. (s -> logits)
  :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
  :param float discount_factor: in [0, 1].
  :param int estimation_step: the number of steps to look ahead. Default to 1.
  :param int target_update_freq: the target network update frequency (0 if
    you do not use the target network). Default to 0.
  :param bool reward_normalization: normalize the reward to Normal(0, 1).
    Default to False.
  :param bool is_double: use double dqn. Default to True.
  :param bool clip_loss_grad: clip the gradient of the loss in accordance
    with nature14236; this amounts to using the Huber loss instead of
    the MSE loss. Default to False.
  :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
    optimizer in each policy.update(). Default to None (no lr_scheduler).

  .. seealso::

    Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
    explanation.
  """
  def __init__(
    self,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    discount_factor: float = 0.99,
    estimation_step: int = 1,
    target_update_freq: int = 0,
    reward_normalization: bool = False,
    is_double: bool = False,
    clip_loss_grad: bool = False,
    consod_epoch: int = 1,
    **kwargs: Any,
  ) -> None:
    super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, clip_loss_grad, **kwargs)
    assert self._target, 'There must be a target Q network in MeDQNPolicy.'
    # For knowledge consolidation
    self.consod_epoch = consod_epoch
    self.lamda = 0.0

  def set_lamda(self, lamda: float) -> None:
    """Set the lamda for knowledge consolidation."""
    self.lamda = lamda

  def learn(self, batch: Batch, consod_batch: Batch, **kwargs: Any) -> Dict[str, float]:
    if self._target and self._iter % self._freq == 0:
      self.sync_weight()
    # Compute TD loss
    q = self(batch).logits
    q = q[np.arange(len(q)), batch.act]
    returns = to_torch_as(batch.returns.flatten(), q)
    td_error = returns - q
    if self._clip_loss_grad:
      y = q.reshape(-1, 1)
      t = returns.reshape(-1, 1)
      loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
    else:
      loss = td_error.pow(2).mean()
    # Add consolidation loss
    loss += self.lamda * self.consolidation_loss(consod_batch)
    # Take an optimization step
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
    self._iter += 1
    return {"loss": loss.item()}

  def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
         **kwargs: Any) -> Dict[str, Any]:
    """Update the policy network and replay buffer.
    It includes 3 function steps: process_fn, learn, and post_process_fn. In
    addition, this function will change the value of ``self.updating``: it will be
    False before this function and will be True when executing :meth:`update`.
    Please refer to :ref:`policy_state` for more detailed explanation.
    :param int sample_size: 0 means it will extract all the data from the buffer,
      otherwise it will sample a batch with given sample_size.
    :param ReplayBuffer buffer: the corresponding replay buffer.
    :return: A dict, including the data needed to be logged (e.g., loss) from
      ``policy.learn()``.
    """
    if buffer is None:
      return {}
    self.updating = True
    batch, indices = buffer.sample(sample_size)
    batch = self.process_fn(batch, buffer, indices)
    # Perform multiple updates
    for i in range(self.consod_epoch):
      consod_batch, _ = buffer.sample(sample_size)
      result = self.learn(batch, consod_batch, **kwargs)
    self.post_process_fn(batch, buffer, indices)
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()
    self.updating = False
    return result

  def consolidation_loss(self, batch: Batch) -> float:
    """Compute consolidation loss."""
    q = self(batch).logits
    with torch.no_grad():
      target_q = self(batch, model="model_old", input="obs").logits
    if self._clip_loss_grad:
      y = q.reshape(-1, 1)
      t = target_q.reshape(-1, 1)
      loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
    else:
      td_error = target_q - q
      loss = td_error.pow(2).mean()
    return loss