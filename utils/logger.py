import logging
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
  def __init__(self, logs_dir, save_interval=1, file_name='log.txt', filemode='w'):
    self.logs_dir = logs_dir
    # Set python logger
    logging.basicConfig(
      format='%(asctime)s - %(levelname)s: %(message)s',
      filename=f'{logs_dir}{file_name}',
      filemode=filemode
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    self.debug = logger.debug
    self.info = logger.info
    self.warning = logger.warning
    self.error = logger.error
    self.critical = logger.critical
    # Set tensorboard writer
    self.tb_writer = TensorboardLogger(SummaryWriter(logs_dir), save_interval=save_interval)
    self.write = self.tb_writer.write
    self.log_train_data = self.tb_writer.log_train_data
    self.log_test_data = self.tb_writer.log_test_data
    self.log_update_data = self.tb_writer.log_update_data
    self.save_data = self.tb_writer.save_data
    self.restore_data = self.tb_writer.restore_data
    self.add_scalar = self.tb_writer.writer.add_scalar