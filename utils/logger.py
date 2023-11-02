import logging
# import colorlog
import os.path
from datetime import datetime

from utils.tools import get_project_root


def init_logger(file_path, log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s %(name)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def get_logger(log_name):
    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = init_logger(os.path.join(get_project_root(), f'log/{cur_time}_{log_name}.log'), log_name)
    return logger
