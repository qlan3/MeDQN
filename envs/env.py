import gym
import warnings
from tianshou.env import ShmemVectorEnv
from tianshou.env import SubprocVectorEnv
try:
  import envpool
except ImportError:
  envpool = None
  
from envs.wrapper import wrap_deepmind


def make_atari_env(task, agent, seed, training_num, test_num, **kwargs):
  """Wrapper function for Atari env.
  If EnvPool is installed, it will automatically switch to EnvPool's Atari env.
  :return: a tuple of (single env, training envs, test envs).
  """
  if envpool is not None:
    if kwargs.get("scale", 0):
      warnings.warn(
        "EnvPool does not include ScaledFloatFrame wrapper, "
        "please set `x = x / 255.0` inside CNN network's forward function."
      )
    # parameters convertion
    train_envs = env = envpool.make_gym(
      task.replace("NoFrameskip-v4", "-v5"),
      num_envs=training_num,
      seed=seed,
      episodic_life=True,
      reward_clip=True,
      stack_num=kwargs.get("frame_stack", 4),
    )
    test_envs = envpool.make_gym(
      task.replace("NoFrameskip-v4", "-v5"),
      num_envs=test_num,
      seed=seed,
      episodic_life=False,
      reward_clip=False,
      stack_num=kwargs.get("frame_stack", 4),
    )
  else:
    warnings.warn(
      "Recommend using envpool (pip install envpool) "
      "to run Atari games more efficiently."
    )
    env = wrap_deepmind(task, **kwargs)
    train_envs = ShmemVectorEnv([
      lambda: wrap_deepmind(task, episode_life=True, clip_rewards=True, **kwargs)
      for _ in range(training_num)
    ])
    test_envs = ShmemVectorEnv([
      lambda: wrap_deepmind(task, episode_life=False, clip_rewards=False, **kwargs)
      for _ in range(test_num)
    ])
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)
  return env, train_envs, test_envs