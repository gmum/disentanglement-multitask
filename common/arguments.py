import argparse
import os

from common import constants as c
from common.utils import str2bool


def update_args(args):
    args.ckpt_load_iternum = False
    args.use_wandb = False
    args.file_save = True
    args.gif_save = True
    return args


def get_args(sys_args):
    parser = argparse.ArgumentParser(description='disentanglement-pytorch')

    # NeurIPS2019 AICrowd Challenge
    parser.add_argument('--evaluation_metric', default=[], type=str, choices=c.EVALUATION_METRICS, nargs='*',
                        help='Metric to evaluate the model during training')

    # name
    parser.add_argument('--alg', type=str, help='the disentanglement algorithm', choices=c.ALGS)
    parser.add_argument('--controlled_capacity_increase', help='to use controlled capacity increase', default=False)
    parser.add_argument('--loss_terms', help='loss terms to be incldued in the objective', nargs='*',
                        default=list(), choices=c.LOSS_TERMS)
    parser.add_argument('--name', default='unknown_experiment', type=str, help='name of the experiment')

    # Neural architectures
    parser.add_argument('--encoder', type=str, nargs='+', required=True, choices=c.ENCODERS,
                        help='name of the encoder network')
    parser.add_argument('--header_type', type=str, required=True, choices=c.HEADERS,
                        help='name of the header network')
    parser.add_argument('--decoder', type=str, nargs='+', required=True, choices=c.DECODERS,
                        help='name of the decoder network')
    parser.add_argument('--label_tiler', type=str, nargs='*', choices=c.TILERS,
                        help='the tile network used to convert one hot labels to 2D channels')

    # Test or train
    parser.add_argument('--test', default=False, type=str2bool, help='to test')

    # training hyper-params
    parser.add_argument('--max_epoch', default=200, type=float, help='maximum training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    # latent encoding
    parser.add_argument('--z_dim', default=16, type=int, help='size of the encoded z space')
    parser.add_argument('--include_labels', default=None, type=str, nargs='*',
                        help='Labels (indices or names) to include in latent encoding.')
    parser.add_argument('--l_dim', default=0, type=str, help='size of the encoded w space (for each label)')

    # optimizer
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 parameter of the Adam optimizer')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 parameter of the Adam optimizer')
    parser.add_argument('--lr_G', default=1e-4, type=float, help='learning rate of the main autoencoder')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of all the discriminators')

    # Neural architectures hyper-parameters
    parser.add_argument('--num_layer_disc', default=6, type=int, help='number of fc layers in discriminators')
    parser.add_argument('--size_layer_disc', default=1000, type=int, help='size of fc layers in discriminators')


    # Hyperparameters for [WICA-AE]
    parser.add_argument('--wica_beta', default=10.0, type=float,
                        help='Hyperparameter for wica loss function')
    parser.add_argument('--gausses_number', default=10, type=int,
                        help='Hyperparameter for number of gausses used in wica')

    # Hyperparameters for Multitask models
    parser.add_argument('--losses_weights', default=None, type=float, nargs='+',
                        help='Weights of subsequent loss functions that we consider')
    parser.add_argument('--single_task_batch', action='store_true',
                        help='Use only loss from a single task for the current batch')

    # Dataset
    parser.add_argument('--dset_dir', default=os.getenv('DISENTANGLEMENT_LIB_DATA', './data'),
                        type=str, help='main dataset directory')
    parser.add_argument('--dset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='width and height of image')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers for the data loader')
    parser.add_argument('--pin_memory', default=True, type=str2bool,
                        help='pin_memory flag of data loader. Check this blogpost for details:'
                             'https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/')

    # Logging and visualization
    parser.add_argument('--train_output_dir', default='train_outputs', type=str, help='output directory')
    parser.add_argument('--test_output_dir', default='test_outputs', type=str, help='test output directory')
    parser.add_argument('--neptune-logging', action="store_true", help="whether to use neptune.ai for logging")
    parser.add_argument('--verbose', default=20, type=int, help='verbosity level')

    # Save/Load checkpoint
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_load_optim', default=True, type=str2bool, help='load the optimizer state')
    parser.add_argument('--save-preds', default=False, type=str2bool, help = 'whether to store the predictions and true values for test dataset after the evaluation')

    # Iterations [default for all is equal to 1 epoch]
    parser.add_argument('--ckpt_save_iter', default=None, type=int, help='iters to save checkpoint [default: 1 epoch]')
    parser.add_argument('--evaluate_iter', default=1, type=int, help='iters to evaluate [default: 1 epoch]')
    parser.add_argument('--float_iter', default=1, type=int, help='use same iteration for all [default: 1 epoch]')
    parser.add_argument('--all_iter', default=None, type=int, help='use same iteration for all [default: 1 epoch]')

    # Other
    parser.add_argument('--seed', default=123, type=int, help='Seed value for torch, cuda, and numpy.')
    parser.add_argument('--header_dim', default=64, type=int, help="hidden dimension of header")
    parser.add_argument('--output_dim', default=3, type=int, help="output dimension for regessors and classifiers")

    parser.add_argument('--test_prec', default=0.125, type=float, help="if a dataset does not have a specified train/valid/test split, this percentage of the whole dataset will be used for test set")
    parser.add_argument('--val_prec', default=0.125, type=float)
    parser.add_argument('--shuffle_seed', default=123, type=int, help="if a dataset does not have a specified train/valid/test split, this seed will be used to shuffle the dataset before spliting into train/valid/test.")
    parser.add_argument('--n_task_headers', type=int, default=10, help="number of task headers")
    
    args = parser.parse_args(sys_args)

    assert args.image_size == 64, 'for now, models are hard coded to support only image size of 64x64'

    args.num_labels = 0
    if args.include_labels is not None:
        args.num_labels = len(args.include_labels)

    # test
    args = update_args(args) if args.test else args

    # make sure arguments for supplementary neural architectures are included
    if c.FACTORVAE in args.loss_terms:
        assert args.discriminator is not None, 'The FactorVAE algorithm needs a discriminator to test the ' \
                                               'permuted latent factors ' \
                                               '(try the flag: --discriminator=SimpleDiscriminator)'

    return args
