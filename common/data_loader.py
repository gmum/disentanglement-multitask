import logging
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from common.datasets import *


def get_transform(image_size=None):
    if image_size is None:
        transform = transforms.Compose([
            transforms.ToTensor(), ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])
    return transform


def _get_dataloader_with_labels(name, dset_dir, batch_size, shuffle_seed, transform,
                                split, test_prec=0.125, val_prec=0.125, num_workers=4, pin_memory=True, n_task_headers=10):
    if name.lower() == 'celeba':
        dataset = CelebAWrapper(dset_dir, split=split, transform=transform, name=name)

    elif name.lower() == 'dsprites_multitask':
        dataset = DspritesMultiTaskDataset(
                os.path.join(dset_dir, "dsprites_multitask.npz"), transform=transform,
                shuffle_seed=shuffle_seed, split=split, test_prec=test_prec, val_prec=val_prec,
                task_headers=n_task_headers)
    elif name.lower() == 'shapes3d_multitask':
        dataset = Shapes3dMultiTaskDataset(
                os.path.join(dset_dir, "3dshapes_multitask.npz"), transform=transform,
                shuffle_seed=shuffle_seed, split=split, test_prec=test_prec, val_prec=val_prec,
                task_headers=n_task_headers)
    elif name.lower() == 'mpi3d_multitask':
        dataset = Mpi3dMultiTaskDataset(
                os.path.join(dset_dir, "mpi3d_multitask.npz"), transform=transform,
                shuffle_seed=shuffle_seed, split=split, test_prec=test_prec, val_prec=val_prec,
                task_headers=n_task_headers)

    else:
        raise NotImplementedError()

    if split == "train":
        shuffle = True
        droplast = True
    else:
        shuffle = False
        droplast = False

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             drop_last=droplast)

    return data_loader


def get_dataset_name(name):
    """Returns the name of the dataset from its input argument (name) or the
    environment variable `AICROWD_DATASET_NAME`, in that order."""
    return name or os.getenv('AICROWD_DATASET_NAME', c.DEFAULT_DATASET)


def get_datasets_dir(dset_dir):
    if dset_dir:
        os.environ['DISENTANGLEMENT_LIB_DATA'] = dset_dir
    return dset_dir or os.getenv('DISENTANGLEMENT_LIB_DATA')


def get_dataloader(dset_name, dset_dir, batch_size, shuffle_seed, image_size,
                   split, test_prec, val_prec, num_workers, pin_memory, n_task_headers):
    dset_name = get_dataset_name(dset_name)
    dsets_dir = get_datasets_dir(dset_dir)

    logging.info('{} {} {} {}'.format(num_workers, pin_memory, split, batch_size))
    logging.info(f'Datasets root: {dset_dir}')
    logging.info(f'Dataset: {dset_name}')

    transforms = get_transform(image_size)

    return _get_dataloader_with_labels(dset_name, dsets_dir, batch_size, shuffle_seed, transforms, split, test_prec, val_prec,
                                       num_workers, pin_memory, n_task_headers)
