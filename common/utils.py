import argparse
import logging
import os

import numpy as np
import torch.nn
import torch.nn.init as init
from torch.autograd import Variable


def cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func

    return decorate


def str2bool(v):
    """
    Thank to stackoverflow user: Maxim
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    :param v: A command line argument with values [yes, true, t, y, 1, True, no, false, f, n, 0, False]
    :return: Boolean version of the command line argument
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LinearScheduler:
    def __init__(self, start_value, target_value=None, epochs=None):
        self.start_value = start_value
        self.target_value = target_value
        assert start_value != target_value, 'start_value and target_value should be different'
        self.mode = min if target_value > start_value else max
        self.per_step = (target_value - start_value) / epochs

    def step(self, step_num):
        return self.mode(self.start_value + step_num * self.per_step, self.target_value)


def _init_layer(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight.data)


def init_layers(modules):
    for block in modules:
        from collections.abc import Iterable
        if isinstance(modules[block], Iterable):
            for m in modules[block]:
                _init_layer(m)
        else:
            _init_layer(modules[block])


def is_time_for(iteration, milestone):
    return iteration % milestone == milestone - 1


def initialize_seeds(seed):
    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def set_environment_variables(dset_dir, dset_name):
    """
    If the argument dset_dir is set, overwrite DISENTANGLEMENT_LIB_DATA.
    else if only $DATASETS is set, use the same for $DISENTANGLEMENT_LIB_DATA
    else if only $DISENTANGLEMENT_LIB_DATA is set, use the same for $DATASETS
    else print warning that the environment variables are not set or inconsistent.

    If the argument dset_name is set, overwrite $DATASET_NAME.
    else if only $DATASET_NAME is set, use the same for $AICROWD_DATASET_NAME
    else if only $AICROWD_DATASET_NAME is set, use the same for $DATASET_NAME
    else print warning that the environment variables are not set or inconsistent.

    :param dset_dir: directory where all the datasets are saved
    :param dset_name: name of the dataset to be loaded by the dataloader
    """
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    if not os.environ.get('DISENTANGLEMENT_LIB_DATA'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}\n")

    if dset_name:
        os.environ['DATASET_NAME'] = dset_name
    if os.environ.get('DATASET_NAME') and not os.environ.get('AICROWD_DATASET_NAME'):
        os.environ['AICROWD_DATASET_NAME'] = os.getenv('DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') and not os.environ.get('DATASET_NAME'):
        os.environ['DATASET_NAME'] = os.getenv('AICROWD_DATASET_NAME')
    elif os.environ.get('AICROWD_DATASET_NAME') != os.environ.get('DATASET_NAME'):
        logging.warning(f"Environment variables are not correctly set:\n"
                        f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}\n"
                        f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")

    logging.info(f"$AICROWD_DATASET_NAME={os.environ.get('AICROWD_DATASET_NAME')}")
    logging.info(f"$DATASET_NAME={os.environ.get('DATASET_NAME')}")
    logging.info(f"$DISENTANGLEMENT_LIB_DATA={os.environ.get('DISENTANGLEMENT_LIB_DATA')}")


def make_dirs(args):
    # makedirs
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.train_output_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)
