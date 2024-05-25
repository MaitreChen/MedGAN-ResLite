from datetime import datetime
import random
import logging

import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_filename):
    current_time = datetime.now().strftime('%Y%m%d%H%M')
    file_name = f'log_{log_filename}_{current_time}.txt'

    logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO, format='%(message)s')
    print(f"Logging results to {log_filename}.")


def log_hyperparameters(args):
    logging.info("Hyper parameters:")
    logging.info("===================================")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    # logging.info("===================================")


def log_fid_value(epoch, fid):
    logging.info(f"Epoch {epoch}  Fid value: {fid}")
