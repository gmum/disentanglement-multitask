import logging
import os
import sys

import torch

import models
from common.arguments import get_args
from common.utils import initialize_seeds, set_environment_variables
from common.loggers import setup_logging

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(_args):

    model_class = getattr(models, _args.alg)
    model = model_class(_args)

    # load checkpoint
    if _args.ckpt_load:
        print("Restoring Model")
        model.load_checkpoint(_args.ckpt_load, load_optim=_args.ckpt_load_optim)

    # run test or train
    if not _args.test:
        model.train()
    else:
        model.evaluate(_args.save_preds)

    from aicrowd import utils_pytorch as pyu
    path = os.path.join(_args.ckpt_dir, 'pytorch_model.pt')
    path_to_saved = pyu.export_model(pyu.RepresentationExtractor(model.model.encoder, 'mean'), path=path,
                                         input_shape=(1, model.num_channels, model.image_size, model.image_size))
    logging.info(f'A copy of the model saved in {path_to_saved}')


if __name__ == "__main__":
    _args = get_args(sys.argv[1:])
    setup_logging(_args.verbose)
    initialize_seeds(_args.seed)

    # set the environment variables for dataset directory and name, and check if the root dataset directory exists.
    set_environment_variables(_args.dset_dir, _args.dset_name)
    assert os.path.exists(os.environ.get('DISENTANGLEMENT_LIB_DATA', '')), \
        'Root dataset directory does not exist at: \"{}\"'.format(_args.dset_dir)

    main(_args)
