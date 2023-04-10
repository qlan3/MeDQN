import os
import sys
import torch
import random
import psutil
import datetime
import numpy as np


def get_time_str():
  return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")

def rss_memory_usage():
  '''
  Return the resident memory usage in MB
  '''
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / float(2 ** 20)
  return mem

def str_to_class(module_name, class_name):
  '''
  Convert string to class
  '''
  return getattr(sys.modules[module_name], class_name)

def set_one_thread():
  '''
  Set number of threads for pytorch to 1
  '''
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  torch.set_num_threads(1)

def to_tensor(x, device):
  '''
  Convert an array to tensor
  '''
  x = torch.as_tensor(x, device=device, dtype=torch.float32)
  return x

def to_numpy(t):
  '''
  Convert a tensor to numpy
  '''
  return t.cpu().detach().numpy()

def set_random_seed(seed):
  '''
  Set all random seeds
  '''
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)

def generate_batch_idxs(length, batch_size):
  idxs = np.asarray(np.random.permutation(length))
  batches = idxs[:length // batch_size * batch_size].reshape(-1, batch_size)
  for batch in batches:
    yield batch