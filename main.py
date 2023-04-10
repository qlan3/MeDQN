import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/dqn.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg.setdefault('show_tb', False)
  cfg.setdefault('save_model', False)
  cfg.setdefault('show_progress', False)
  cfg.setdefault('resume_from_log', False)
  cfg.setdefault('save_interval', 1)

  # Set experiment name and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  local_logs_dir = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  make_dir(local_logs_dir)
  if len(args.slurm_dir) > 0:
    cfg['logs_dir'] = f"{args.slurm_dir}/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(cfg['logs_dir'])
    # Copy from local_logs_dir to slurm_dir
    if cfg['resume_from_log']:
      os.system(f"cp -rf {local_logs_dir}. {cfg['logs_dir']}")
  else:
    cfg['logs_dir'] = local_logs_dir
  cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
  cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
  cfg['model_path'] = cfg['logs_dir'] + 'model.pth'
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'
  # Set checkpoint path
  cfg['ckpt_path'] = cfg['logs_dir'] + 'ckpt.pth'

  # Special case for BreakoutNoFrameskip-v4
  if 'Breakout' in cfg['env']['name']:
    cfg['n_step'] = 1
  
  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)