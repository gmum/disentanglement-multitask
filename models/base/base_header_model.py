import logging
import os

import torch

from common.data_loader import get_dataloader
from common.loggers import NeptuneLogger, MetricLogger, Accumulator
from common.savers import ModelSaver
import numpy as np

DEBUG = False


class BaseHeaderModel(object):
    def __init__(self, args):

        # Cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info('Device: {}'.format(self.device))
        self.seed = args.seed

        # Misc
        self.name = args.name
        self.alg = args.alg
        self.controlled_capacity_increase = args.controlled_capacity_increase
        self.loss_terms = args.loss_terms
        self.lr_scheduler = None
        self.optim_G = None
        self.optims = None  # Used with separate multi tasks
        self.lr_schedulers = []  # Used with separate multi tasks

        # Output directory
        self.train_output_dir = args.train_output_dir
        self.test_output_dir = args.test_output_dir
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Latent space
        self.z_dim = args.z_dim
        self.l_dim = args.l_dim
        self.num_labels = args.num_labels

        # Solvers
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr_G = args.lr_G
        self.max_epoch = int(args.max_epoch)

        # Data
        self.dset_dir = args.dset_dir
        self.dset_name = args.dset_name
        self.batch_size = args.batch_size
        self.image_size = args.image_size

        
        self.test_loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.shuffle_seed, args.image_size,
                                          split="test", test_prec=args.test_prec, val_prec=args.val_prec,
                                          num_workers=args.num_workers, pin_memory=False, n_task_headers = args.n_task_headers)
        if not args.test:
            self.valid_loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.shuffle_seed, args.image_size,
                                           split="valid", test_prec=args.test_prec, val_prec=args.val_prec,
                                           num_workers=args.num_workers, pin_memory=False, n_task_headers = args.n_task_headers)
        else:
            self.valid_loader=None
            
        if not args.test:
            self.train_loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.shuffle_seed, args.image_size,
                                           split="train", test_prec=args.test_prec, val_prec=args.val_prec,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory, n_task_headers = args.n_task_headers)
        else:
            self.train_loader=None
            
        #assert self.test_loader.dataset.has_labels(), "For header models labels are required."
        #assert self.valid_loader.dataset.has_labels(), "For header models labels are required."
        #if not args.test:
        #    assert self.train_loader.dataset.has_labels(), "For header models labels are required."


        if not self.test:
            self.num_classes = self.train_loader.dataset.num_classes()
            if self.num_classes is not None:
                self.total_num_classes = sum(self.train_loader.dataset.num_classes(False))
                self.class_values = self.train_loader.dataset.class_values()
            self.num_channels = self.train_loader.dataset.num_channels()
        else:
            self.num_classes = self.test_loader.dataset.num_classes()
            if self.num_classes is not None:
                self.total_num_classes = sum(self.test_loader.dataset.num_classes(False))
                self.class_values = self.test_loader.dataset.class_values()            

            self.num_channels = self.test_loader.dataset.num_channels()

        if args.test:
            self.num_batches = len(self.test_loader)
            logging.info('Number of samples: {}'.format(len(self.test_loader.dataset)))
        else:
            self.num_batches = len(self.train_loader)
            logging.info('Number of batches per epoch: {}'.format(self.num_batches))

        logging.info('Number of channels: {}'.format(self.num_channels))

        # TODO: How important is the experiment name?
        self.metric_logger = MetricLogger(args, self.num_batches, args.neptune_logging)
        self.model_saver = ModelSaver(args, self.num_batches, args.ckpt_dir)
        self.ckpt_path = args.ckpt_dir
        self.accumulator = Accumulator()

        self.net_dict = dict()
        self.optim_dict = dict()

        # model is the only attribute that all sub-classes should have
        self.model = None

    def save_checkpoint(self, iter, ckptname='last'):
        self.model_saver.save_checkpoint(iter, self.net_dict, self.optim_dict, ckptname)

    def save_metric_results(self, logdir, filename="results.npy"):
        self.metric_logger.save_results(logdir, filename)

    def load_checkpoint(self, filepath, ignore_failure=True, load_optim=True):

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            for key, value in self.net_dict.items():
                try:
                    if isinstance(value, dict):
                        state_dicts = checkpoint['model_states'][key]
                        for sub_key, net in value.items():
                            value[sub_key].load_state_dict(state_dicts[sub_key], strict=False)
                    else:
                        value.load_state_dict(checkpoint['model_states'][key], strict=False)
                except Exception as e:
                    logging.warning("Could not load {}".format(key))
                    logging.warning(str(e))
                    if not ignore_failure:
                        raise e
            if load_optim:
                for key, value in self.optim_dict.items():
                    try:
                        if isinstance(value, dict):
                            state_dicts = checkpoint['optim_states'][key]
                            for sub_key, net in value.items():
                                value[sub_key].load_state_dict(state_dicts[sub_key])
                        else:
                            value.load_state_dict(checkpoint['optim_states'][key])
                    except Exception as e:
                        logging.warning("Could not load {}".format(key))
                        logging.warning(str(e))
                        if not ignore_failure:
                            raise e

            logging.info("Model Loaded: {}".format(filepath))
        else:
            logging.error("File does not exist: {}".format(filepath))

    def net_mode(self, train):
        for net in self.net_dict.values():
            if train:
                net.train()
            else:
                net.eval()

    def encode_deterministic(self, **kwargs):
        images = kwargs['images']
        if len(images.size()) == 3:
            images = images.unsqueeze(0)
        return self.model.encode(images)

    def encode_stochastic(self, **kwargs):
        raise NotImplementedError()

    def loss_fn(self, **kwargs):
        raise NotImplementedError()
        
    def test(self, data_loader, iter=None, split="test", save=True, save_preds=False):
        raise NotImplementedError()
     
    def evaluate(self, save_preds):
        preds, trues = self.test(self.test_loader, iter=None, split="test", save=True, save_preds=save_preds)
        res = {"preds":preds, "trues":trues}
        if save_preds:
            np.save(os.path.join(self.test_output_dir, "test_preds.npy"), res)
