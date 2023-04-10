import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
from utils.helper import set_one_thread


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-10:].mean(skipna=False) if mode=='Train' else result['Return'][-5:].mean(skipna=False),
    'Return (max)': result['Return'].max(skipna=False) if mode=='Train' else result['Return'].max(skipna=False)
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (max)': result['Return (max)'].max(skipna=False)
  }
  return result_dict

cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Return',
  'hue_label': 'Agent',
  'show': False,
  'rolling_score_window': 20,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None, 
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'upper left',
  'sweep_keys': ['buffer_size', 'agent/consod_end', 'agent/consod_epoch', 'update_per_step'],
  'sort_by': ['Return (mean)', 'Return (max)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Test', indexes='all')
  
  expIndexModeList = {
    "Qbert": [['dqn', 1, 'Test'], ['dqn', 6, 'Test'], ['medqn', 36, 'Test']],
    "BattleZone": [['dqn', 2, 'Test'], ['dqn', 7, 'Test'], ['medqn', 52, 'Test']],
    "DoubleDunk": [['dqn', 3, 'Test'], ['dqn', 8, 'Test'], ['medqn', 53, 'Test']],
    "NameThisGame": [['dqn', 4, 'Test'], ['dqn', 9, 'Test'], ['medqn', 54, 'Test']],
    "Phoenix": [['dqn', 5, 'Test'], ['dqn', 10, 'Test'], ['medqn', 40, 'Test']]
  }
  for env in ["Qbert", "BattleZone", "DoubleDunk", "NameThisGame", "Phoenix"]:
    plotter.plot_expIndexModeList(expIndexModeList[env], env)


if __name__ == "__main__":
  # exp, runs = 'dqn', 5
  exp, runs = 'medqn', 5
  unfinished_index(exp, runs=runs)
  memory_info(exp, runs=runs)
  time_info(exp, runs=runs)
  analyze(exp, runs=runs)