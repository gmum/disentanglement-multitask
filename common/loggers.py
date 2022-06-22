import logging
import os
from importlib import reload
import torch

import neptune
import numpy as np

from aicrowd.aicrowd_utils import evaluate_disentanglement_metric
from common import constants as c
from common.utils import is_time_for

DEFAULT_ITER = 1


class NeptuneLogger:
    def __init__(self, experiment_name, params):
        self.experiment_name = experiment_name
        self.params = params
        self.isinit = False
        self.init()

    def init(self):
        neptune.init("<user>/disentanglement-multitask")
        neptune.create_experiment(name=self.experiment_name, params=self.params)
        self.isinit = True

    def log_metric_dict(self, dict):
        assert self.isinit, "Neptune logging must be initialized"
        for key, value in dict.items():
            neptune.log_metric(key, value)

    def log_metric(self, key, value):
        assert self.isinit, "Neptune logging must be initialized"
        neptune.log_metric(key, value)

    def close(self):
        neptune.stop()
        self.isinit = False

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor  

class Accumulator():
    def __init__(self):
        self.values = {}
        self.weights = {}

    def cumulate(self, key, value, n=1):
        '''
        Cumulate the values of some loss/metric
        Args:
            key: the name of the loss
            value: the value for this batch
            n: the weight for this batch. For instance, if MSE is used, one may wish to store the sum instead of the mean,
            and then average the outputs for whole epoch using get_average.

        Returns:

        '''
        
        val = to_numpy(value)
        count = to_numpy(n)
            
        assert type(key) == str, "key must be a string" 
        
        if key not in self.values.keys():
            self.values[key] = []
        if key not in self.weights.keys():
            self.weights[key] = []

        self.values[key].append(val)
        self.weights[key].append(count)

    def get_average(self):
        averages = {}
        for key in self.values:
            averages[key] = np.sum(np.array(self.values[key]) * np.array(self.weights[key])) / np.sum(
                np.array(self.weights[key]))
        return averages

    def get_values(self):
        return self.values

    def reinit(self):
        self.values = {}
        self.weights = {}


class MetricLogger:
    def __init__(self, args, num_batches, neptune_logging):
        # logging
        self.logging_dict = {}
        self.num_batches = num_batches
        self.evaluation_metric = args.evaluation_metric
        
        if neptune_logging:
            self.neptune_logger = NeptuneLogger(args.name, params=args.__dict__)
        else:
            self.neptune_logger = None


        # logging
        self.epoch = 0
        self.evaluate_results = dict()

        # logging iterations
        self.float_iter = args.float_iter if args.float_iter else DEFAULT_ITER
        self.evaluate_iter = args.evaluate_iter if args.evaluate_iter else DEFAULT_ITER

    def compute_metrics(self, model, iter, split, **kwargs):
        # pass None iter to compute regardless the iteration.

        if iter is None or is_time_for(iter, self.float_iter):
            self.display_message(iter, kwargs)
            for key, value in kwargs.items():
                if key not in self.logging_dict.keys():
                    self.logging_dict[key] = []
                self.logging_dict[key].append(value)
                self.neptune_logger.log_metric(split + "/" + key, value)
            if self.neptune_logger is not None and iter is not None:
                self.neptune_logger.log_metric(split + "/float_iter", iter)

        if iter is None or is_time_for(iter, self.evaluate_iter):
            evaluation_results = evaluate_disentanglement_metric(model, metric_names=self.evaluation_metric)
            for key, value in evaluation_results.items():
                if key not in self.logging_dict.keys():
                    self.logging_dict[key] = []
                self.logging_dict[key].append(value)
                self.neptune_logger.log_metric(split + "/" + key, value)
            if self.neptune_logger is not None and iter is not None:
                self.neptune_logger.log_metric(split + "/evaluate_iter", iter)

    def display_message(self, iter, kwargs):
        msg = '[{}:{}]  '.format(self.epoch, iter)
        for key, value in kwargs.get(c.LOSS, dict()).items():
            msg += '{}_{}={:.3f}  '.format(c.LOSS, key, value)
        for key, value in kwargs.get(c.ACCURACY, dict()).items():
            msg += '{}_{}={:.3f}  '.format(c.ACCURACY, key, value)
        print(msg)

    def save_results(self, logdir, filename="results.npy"):
        np.save(os.path.join(logdir, filename), self.logging_dict)


class StyleFormatter(logging.Formatter):
    CSI = "\x1B["
    YELLOW = '33;40m'
    RED = '31;40m'

    # Add %(asctime)s after [ to include the time-date of the log
    high_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, RED, CSI)
    medium_style = '{}{}(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s{}0m'.format(CSI, YELLOW, CSI)
    low_style = '(%(levelname)s)[%(filename)s:%(lineno)d]  %(message)s'

    def __init__(self, fmt=None, datefmt='%b-%d %H:%M', style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style = logging.PercentStyle(StyleFormatter.low_style)
        elif record.levelno <= logging.WARNING:
            self._style = logging.PercentStyle(StyleFormatter.medium_style)
        else:
            self._style = logging.PercentStyle(StyleFormatter.high_style)

        return logging.Formatter.format(self, record)


def setup_logging(verbose):
    # verbosity
    reload(logging)  # to turn off any changes to logging done by other imported libraries
    h = logging.StreamHandler()
    h.setFormatter(StyleFormatter())
    h.setLevel(0)
    logging.root.addHandler(h)
    logging.root.setLevel(verbose)
