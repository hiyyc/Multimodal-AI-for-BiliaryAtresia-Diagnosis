import os
import random

import numpy as np
import torch

from utils.logger import get_logger
from utils.tools import get_project_root

def set_seed(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(666)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger('BA')
logger.info('describe: swin v2 aggr ddp')
data_root = f'/path/data/BA_Dataset'
fold_id = 2
dataloader_args = {
    'num_workers': 4,
    'batch_size': 16,
    # 'batch_size': 512,
    'test_batch_size': 1,
    # 'img_size': 224,
    'img_size': 256, # swin v2
    'train_img_csv': os.path.join(data_root, f'data_train.csv'),
    'val_img_csv': os.path.join(data_root, f'data_val.csv'),
    'test_img_csv': os.path.join(data_root, f'data_test.csv'),
    'record_csv': os.path.join(data_root, f'record.csv'),
    'img_dir': os.path.join(data_root, f'Dataset'),
    'logger': logger,
    'img_types': [0, 1, 2, 3],
    'missing_modal': 1, # 0 for zero 1 for copy
}
num_epochs = 10
config = {
    'hidden_size': 2048,
    'img_size': dataloader_args['img_size'],
    'label_size': 2,
    'device': None,
    'logger': logger,
    'num_class': 2,
    'load_path': '',
    'img_types': dataloader_args['img_types'],
    # 'image_encoder_type': 'swin_tiny_patch4_window7_224',
    'image_encoder_type': 'swinv2_small_window8_256',
    'dim_projection': 512,
    'tabular_len': 640,
    # 'tabular_len': 560,
    'num_epochs': num_epochs,
    'model_save_path':
        os.path.join(
            get_project_root(),
            f'store/bs32_5loss_sma-epoch{num_epochs}.pth.tar'),
    'heatmap': True,
    # 'finetune': True,
    'finetune': False,
    'attn': 'sma', # sma, sa, none
}
# logger.info(f'dataloader args: {dataloader_args}')
# logger.info(f'config: {config}')