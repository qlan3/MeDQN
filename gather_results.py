import os
import math
import json
import pandas as pd
from utils.sweeper import Sweeper
from utils.helper import make_dir


GAME_NAMES = [
  ('alien', 'Alien'),
  ('amidar', 'Amidar'),
  ('assault', 'Assault'),
  ('asterix', 'Asterix'),
  ('asteroids', 'Asteroids'),
  ('atlantis', 'Atlantis'),
  ('bank_heist', 'Bank Heist'),
  ('battle_zone', 'Battlezone'),
  ('beam_rider', 'Beam Rider'),
  ('berzerk', 'Berzerk'),
  ('bowling', 'Bowling'),
  ('boxing', 'Boxing'),
  ('breakout', 'Breakout'),
  ('centipede', 'Centipede'),
  ('chopper_command', 'Chopper Command'),
  ('crazy_climber', 'Crazy Climber'),
  ('defender', 'Defender'),
  ('demon_attack', 'Demon Attack'),
  ('double_dunk', 'Double Dunk'),
  ('enduro', 'Enduro'),
  ('fishing_derby', 'Fishing Derby'),
  ('freeway', 'Freeway'),
  ('frostbite', 'Frostbite'),
  ('gopher', 'Gopher'),
  ('gravitar', 'Gravitar'),
  ('hero', 'H.E.R.O.'),
  ('ice_hockey', 'Ice Hockey'),
  ('jamesbond', 'James Bond 007'),
  ('kangaroo', 'Kangaroo'),
  ('krull', 'Krull'),
  ('kung_fu_master', 'Kung-Fu Master'),
  ('montezuma_revenge', 'Montezuma’s Revenge'),
  ('ms_pacman', 'Ms. Pac-Man'),
  ('name_this_game', 'Name This Game'),
  ('phoenix', 'Phoenix'),
  ('pitfall', 'Pitfall!'),
  ('pong', 'Pong'),
  ('private_eye', 'Private Eye'),
  ('qbert', 'Q*bert'),
  ('riverraid', 'River Raid'),
  ('road_runner', 'Road Runner'),
  ('robotank', 'Robotank'),
  ('seaquest', 'Seaquest'),
  ('skiing', 'Skiing'),
  ('solaris', 'Solaris'),
  ('space_invaders', 'Space Invaders'),
  ('star_gunner', 'Stargunner'),
  ('surround', 'Surround'),
  ('tennis', 'Tennis'),
  ('time_pilot', 'Time Pilot'),
  ('tutankham', 'Tutankham'),
  ('up_n_down', 'Up’n Down'),
  ('venture', 'Venture'),
  ('video_pinball', 'Video Pinball'),
  ('wizard_of_wor', 'Wizard of Wor'),
  ('yars_revenge', 'Yars’ Revenge'),
  ('zaxxon', 'Zaxxon'),
]
GAME_NAME_MAP = dict(GAME_NAMES)
map_to_dqn_zoo_env_name =  {
  'AlienNoFrameskip-v4': 'alien',
  'AmidarNoFrameskip-v4': 'amidar',
  'AssaultNoFrameskip-v4': 'assault',
  'AsterixNoFrameskip-v4': 'asterix',
  'AsteroidsNoFrameskip-v4': 'asteroids',
  'AtlantisNoFrameskip-v4': 'atlantis',
  'BankHeistNoFrameskip-v4': 'bank_heist',
  'BattleZoneNoFrameskip-v4': 'battle_zone',
  'BeamRiderNoFrameskip-v4': 'beam_rider',
  'BowlingNoFrameskip-v4': 'bowling',
  'BoxingNoFrameskip-v4': 'boxing',
  'BreakoutNoFrameskip-v4': 'breakout',
  'CentipedeNoFrameskip-v4': 'centipede',
  'ChopperCommandNoFrameskip-v4': 'chopper_command',
  'CrazyClimberNoFrameskip-v4': 'crazy_climber',
  'DemonAttackNoFrameskip-v4': 'demon_attack',
  'DoubleDunkNoFrameskip-v4': 'double_dunk',
  'EnduroNoFrameskip-v4': 'enduro',
  'FishingDerbyNoFrameskip-v4': 'fishing_derby',
  'FreewayNoFrameskip-v4': 'freeway',
  'FrostbiteNoFrameskip-v4': 'frostbite',
  'GopherNoFrameskip-v4': 'gopher',
  'GravitarNoFrameskip-v4': 'gravitar',
  'HeroNoFrameskip-v4': 'hero',
  'IceHockeyNoFrameskip-v4': 'ice_hockey',
  'JamesbondNoFrameskip-v4': 'jamesbond',
  'KangarooNoFrameskip-v4': 'kangaroo',
  'KrullNoFrameskip-v4': 'krull',
  'KungFuMasterNoFrameskip-v4': 'kung_fu_master',
  'MontezumaRevengeNoFrameskip-v4': 'montezuma_revenge',
  'MsPacmanNoFrameskip-v4': 'ms_pacman',
  'NameThisGameNoFrameskip-v4': 'name_this_game',
  'PhoenixNoFrameskip-v4': 'phoenix',
  'PitfallNoFrameskip-v4': 'pitfall',
  'PongNoFrameskip-v4': 'pong',
  'PrivateEyeNoFrameskip-v4': 'private_eye',
  'QbertNoFrameskip-v4': 'qbert',
  'RiverraidNoFrameskip-v4': 'riverraid',
  'RoadRunnerNoFrameskip-v4': 'road_runner',
  'RobotankNoFrameskip-v4': 'robotank',
  'SeaquestNoFrameskip-v4': 'seaquest',
  'SolarisNoFrameskip-v4': 'solaris',
  'SpaceInvadersNoFrameskip-v4': 'space_invaders',
  'StarGunnerNoFrameskip-v4': 'star_gunner',
  'TennisNoFrameskip-v4': 'tennis',
  'TimePilotNoFrameskip-v4': 'time_pilot',
  'TutankhamNoFrameskip-v4': 'tutankham',
  'UpNDownNoFrameskip-v4': 'up_n_down',
  'VentureNoFrameskip-v4': 'venture',
  'VideoPinballNoFrameskip-v4': 'video_pinball',
  'WizardOfWorNoFrameskip-v4': 'wizard_of_wor',
  'ZaxxonNoFrameskip-v4': 'zaxxon'
}
# Game: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: score human (float).
_ATARI_DATA = {
  'alien': (227.8, 7127.7),
  'amidar': (5.8, 1719.5),
  'assault': (222.4, 742.0),
  'asterix': (210.0, 8503.3),
  'asteroids': (719.1, 47388.7),
  'atlantis': (12850.0, 29028.1),
  'bank_heist': (14.2, 753.1),
  'battle_zone': (2360.0, 37187.5),
  'beam_rider': (363.9, 16926.5),
  'berzerk': (123.7, 2630.4),
  'bowling': (23.1, 160.7),
  'boxing': (0.1, 12.1),
  'breakout': (1.7, 30.5),
  'centipede': (2090.9, 12017.0),
  'chopper_command': (811.0, 7387.8),
  'crazy_climber': (10780.5, 35829.4),
  'defender': (2874.5, 18688.9),
  'demon_attack': (152.1, 1971.0),
  'double_dunk': (-18.6, -16.4),
  'enduro': (0.0, 860.5),
  'fishing_derby': (-91.7, -38.7),
  'freeway': (0.0, 29.6),
  'frostbite': (65.2, 4334.7),
  'gopher': (257.6, 2412.5),
  'gravitar': (173.0, 3351.4),
  'hero': (1027.0, 30826.4),
  'ice_hockey': (-11.2, 0.9),
  'jamesbond': (29.0, 302.8),
  'kangaroo': (52.0, 3035.0),
  'krull': (1598.0, 2665.5),
  'kung_fu_master': (258.5, 22736.3),
  'montezuma_revenge': (0.0, 4753.3),
  'ms_pacman': (307.3, 6951.6),
  'name_this_game': (2292.3, 8049.0),
  'phoenix': (761.4, 7242.6),
  'pitfall': (-229.4, 6463.7),
  'pong': (-20.7, 14.6),
  'private_eye': (24.9, 69571.3),
  'qbert': (163.9, 13455.0),
  'riverraid': (1338.5, 17118.0),
  'road_runner': (11.5, 7845.0),
  'robotank': (2.2, 11.9),
  'seaquest': (68.4, 42054.7),
  'skiing': (-17098.1, -4336.9),
  'solaris': (1236.3, 12326.7),
  'space_invaders': (148.0, 1668.7),
  'star_gunner': (664.0, 10250.0),
  'surround': (-10.0, 6.5),
  'tennis': (-23.8, -8.3),
  'time_pilot': (3568.0, 5229.2),
  'tutankham': (11.4, 167.6),
  'up_n_down': (533.4, 11693.2),
  'venture': (0.0, 1187.5),
  # Note the random agent score on Video Pinball is sometimes greater than the
  # human score under other evaluation methods.
  'video_pinball': (16256.9, 17667.9),
  'wizard_of_wor': (563.5, 4756.5),
  'yars_revenge': (3092.9, 54576.9),
  'zaxxon': (32.5, 9173.3),
}
_RANDOM_COL, _HUMAN_COL = 0, 1
ATARI_GAMES = tuple(sorted(_ATARI_DATA.keys()))


def get_human_normalized_score(game, raw_score):
  """Converts game score to human-normalized score."""
  game_scores = _ATARI_DATA.get(game, (math.nan, math.nan))
  random, human = game_scores[_RANDOM_COL], game_scores[_HUMAN_COL]
  return (raw_score - random) / (human - random)


def csv_all_results(expIndexList, agent_id, run=5, mode='Test', log_dir='./results'):
  make_dir(log_dir)
  dfs = []
  for r in range(run):
    for exp, i in expIndexList:
      num_combinations = Sweeper(f'./configs/{exp}.json').config_dicts['num_combinations']
      print(f'exp={exp}, index={i}, run={r}')
      # Read result file
      result_file = f'./logs/{exp}/{i+r*num_combinations}/result_{mode}.feather'
      assert os.path.isfile(result_file), f'No such file <{result_file}>'
      df = pd.read_feather(result_file)
      assert df is not None, f'No result in file <{result_file}>'
      # Read config file
      config_file = f'./logs/{exp}/{i}/config.json'
      with open(config_file, 'r') as f:
        config = json.load(f)
        seed = r # Change seeds of all games to the same seed
        environment_name = map_to_dqn_zoo_env_name[config['env']['name']]
      # Add new columns: agent_id, seed, environment_name, frame, normalized_return
      df = df.assign(agent_id=agent_id, seed=seed, environment_name=environment_name)
      df['frame'] = df['Step'] * 4
      df['normalized_return'] = get_human_normalized_score(environment_name, df['Return'])
      df.rename(columns={'Return': 'eval_episode_return'}, inplace=True)
      # Delete useless columns
      df.drop(['Agent', 'Env', 'Step', 'Time'], axis=1, inplace=True)
      # Gather together
      dfs.append(df)
  dfs = pd.concat(dfs, sort=True).reset_index(drop=True)
  # Save to csv file
  dfs.to_csv(f'{log_dir}/{agent_id}.csv', index=False)


if __name__ == "__main__":
  expIndexList = {
    'dqn': [('dqn', 1), ('dqn', 2), ('dqn', 3), ('dqn', 4), ('dqn', 5)],
    'dqn_s': [('dqn', 6), ('dqn', 7), ('dqn', 8), ('dqn', 9), ('dqn', 10)],
    'medqn': [('medqn', 36), ('medqn', 52), ('medqn', 53), ('medqn', 54), ('medqn', 40)]
  }
  for agent_id in ['dqn', 'dqn_s', 'medqn']:
    csv_all_results(expIndexList[agent_id], agent_id, log_dir='./results')