import os
import torch
import pandas as pd
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.logger import Logger
from envs.env import make_atari_env
from components.network import DQNNet


class BaseAgent(object):
  def __init__(self, cfg):
    self.cfg = cfg
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.discount = cfg['discount']
    self.batch_size = cfg['batch_size']
    self.cfg['epoch'] = int(self.cfg['epoch'])
    self.save_only_last_obs = True
    if self.cfg['step_per_collect'] < 0:
      self.cfg['step_per_collect'] = round(1 / self.cfg['update_per_step'])
    # Make envs
    self.envs = dict()
    self.env, self.envs['Train'], self.envs['Test'] = make_atari_env(
      task = self.env_name,
      agent = self.agent_name,
      seed = self.cfg['seed'],
      training_num = self.cfg['env']['train_num'],
      test_num = self.cfg['env']['test_num'],
      scale = self.cfg['env']['scale_obs'],
      frame_stack = 4
    )
    self.state_shape = self.get_state_shape(self.env)
    self.action_shape = self.get_action_shape(self.env)
    self.log_path = {'Train': self.cfg['train_log_path'], 'Test': self.cfg['test_log_path']}
    # Set python logger and tensorboard logger
    self.logger = Logger(cfg['logs_dir'], save_interval=self.cfg['save_interval'])

  def createNN(self):
    NN = DQNNet(
      *self.state_shape,
      action_shape = self.action_shape, 
      device = self.device
    )
    return NN.to(self.device)

  def get_state_shape(self, env):
    if isinstance(env.observation_space, Discrete):
      return env.observation_space.n
    else: # Box, MultiBinary
      return env.observation_space.shape
  
  def get_action_shape(self, env):
    if isinstance(env.action_space, Discrete):
      return env.action_space.n
    elif isinstance(env.action_space, Box):
      return env.action_space.shape
    else:
      raise ValueError('Unknown action type.')

  def save_model(self, model):
    torch.save(model.state_dict(), self.cfg['model_path'])

  def save_checkpoint(self, epoch, env_step, gradient_step):
    # Save model and optimizer states
    ckpt_dict = dict(model=self.policy.state_dict(), optim=self.policy.optim.state_dict())
    torch.save(ckpt_dict, self.cfg['ckpt_path'])
    # Save results
    try:
      self.save_result('Train')
      self.save_result('Test')
    except:
      self.logger.info('Failed to save results')
    self.logger.info(f'Save checkpoint at epoch={epoch}')
    return self.cfg['ckpt_path']
  
  def load_checkpoint(self):
    if os.path.exists(self.cfg['ckpt_path']):
      ckpt_dict = torch.load(self.cfg['ckpt_path'], map_location=self.device)
      self.policy.load_state_dict(ckpt_dict['model'])
      self.policy.optim.load_state_dict(ckpt_dict['optim'])
      self.logger.info(f"Successfully restore policy and optim from: {self.cfg['ckpt_path']}.")
    else:
      self.logger.info(f"Checkpoint path: {self.cfg['ckpt_path']} does not exist.")

  def save_result(self, mode):
    # Convert tensorboard data to DataFrame, and save it.
    ea = EventAccumulator(self.cfg['logs_dir'])
    ea.Reload()
    # Get return
    tag = f'{mode.lower()}/reward'
    events = ea.Scalars(tag)
    start_time = events[0].wall_time
    result_list = []
    for event in events:
      result_dict = {
        'Env': self.env_name,
        'Agent': self.agent_name,
        'Step': event.step, 
        'Return': event.value,
        'Time': event.wall_time - start_time
      }
      result_list.append(result_dict)

    result_df = pd.DataFrame(result_list)
    result_df['Env'] = result_df['Env'].astype('category')
    result_df['Agent'] = result_df['Agent'].astype('category')
    result_df.to_feather(self.log_path[mode])