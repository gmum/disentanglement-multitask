import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

from common import constants as c


class CelebAWrapper(CelebA):
    def __init__(self, root, split, transform, name):
        super(CelebAWrapper, self).__init__(root, split=split, transform=transform)
        self._name = name

        self._num_classes_torch = torch.tensor([2] * c.CELEB_NUM_CLASSES)
        self._num_classes_list = [2] * c.CELEB_NUM_CLASSES

    @property
    def name(self):
        return self._name

    def num_classes(self, as_tensor=True):
        if as_tensor:
            return self._num_classes_torch
        else:
            return self._num_classes_list

    def class_values(self):
        return [[0, 1]] * c.CELEB_NUM_CLASSES
        # return self.label_handler.class_values()

    def has_labels(self):
        return True

    def num_channels(self):
        return 3


class DspritesMultiTaskDataset(Dataset):
    def __init__(self, datapath, transform, shuffle_seed, split, num_channels=1, name="dsprites_multitask",
                 val_prec=0.125,
                 test_prec=0.125, task_headers=10):

        self.split = split
        self.data = np.load(datapath)
        data_images = self.data['imgs']
        labels = self.data['multitask_targets'][:, :task_headers]
        ground_true = self.data['latents_values']

  
        
        data_images, labels, ground_true = shuffle(data_images, labels, ground_true, random_state=shuffle_seed)
        self.val_ind = int((1 - (test_prec + val_prec)) * len(data_images))
        self.test_ind = int((1 - test_prec) * len(data_images))

        if self.split == "train":
            self.data_npz = data_images[:self.val_ind]
            self.labels = labels[:self.val_ind]
            self.ground_true = ground_true[:self.val_ind]
        elif self.split == "valid":
            self.data_npz = data_images[self.val_ind:self.test_ind]
            self.labels = labels[self.val_ind:self.test_ind]
            self.ground_true = ground_true[self.val_ind:self.test_ind]
        elif self.split == "test":
            self.data_npz = data_images[self.test_ind:]
            self.labels = labels[self.test_ind:]
            self.ground_true = ground_true[self.test_ind:]
            
        self.shuffle_seed = shuffle_seed
        self._name = name
        self._num_channels = num_channels

        self.transform = transform
        self.indices = range(len(self))

    @property
    def name(self):
        return self._name

    def has_labels(self):
        return True

    def num_classes(self, as_tensor=True):
        return None

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        img1 = Image.fromarray(self.data_npz[index1] * 255)
        if self.transform is not None:
            img1 = self.transform(img1)
        label1 = self.labels[index1]
        return img1, label1

    def __len__(self):
        return self.data_npz.shape[0]

class Shapes3dMultiTaskDataset(Dataset):
    def __init__(self, datapath, transform, shuffle_seed, split, num_channels=3, name="shapes3d_multitask",
                 val_prec=0.125,
                 test_prec=0.125, task_headers=10):

        self.split = split
        self.data = np.load(datapath)
        data_images = self.data['images']
        labels = self.data['multitask_targets'][:, :task_headers]
        ground_true = self.data['labels']
        
        data_images, labels, ground_true = shuffle(data_images, labels, ground_true, random_state=shuffle_seed)
        self.val_ind = int((1 - (test_prec + val_prec)) * len(data_images))
        self.test_ind = int((1 - test_prec) * len(data_images))

        if self.split == "train":
            self.data_npz = data_images[:self.val_ind]
            self.labels = labels[:self.val_ind]
            self.ground_true = ground_true[:self.val_ind]
        elif self.split == "valid":
            self.data_npz = data_images[self.val_ind:self.test_ind]
            self.labels = labels[self.val_ind:self.test_ind]
            self.ground_true = ground_true[self.val_ind:self.test_ind]
        elif self.split == "test":
            self.data_npz = data_images[self.test_ind:]
            self.labels = labels[self.test_ind:]
            self.ground_true = ground_true[self.test_ind:]
            
        self.shuffle_seed = shuffle_seed
        self._name = name
        self._num_channels = num_channels

        self.transform = transform
        self.indices = range(len(self))

    @property
    def name(self):
        return self._name

    def has_labels(self):
        return True

    def num_classes(self, as_tensor=True):
        return None

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        img1 = Image.fromarray(self.data_npz[index1])
        if self.transform is not None:
            img1 = self.transform(img1)
        label1 = self.labels[index1]
        return img1, label1

    def __len__(self):
        return self.data_npz.shape[0]

class Mpi3dMultiTaskDataset(Dataset):
    def __init__(self, datapath, transform, shuffle_seed, split, num_channels=3, name="mpi3d_multitask",
                 val_prec=0.125,
                 test_prec=0.125, task_headers=10):

        self.split = split
        self.data = np.load(datapath)
        data_images = self.data['images']
        labels = self.data['multitask_targets'][:, :task_headers]
        ground_true = self.data['labels']

        data_images, labels, ground_true = shuffle(data_images, labels, ground_true, random_state=shuffle_seed)
        self.val_ind = int((1 - (test_prec + val_prec)) * len(data_images))
        self.test_ind = int((1 - test_prec) * len(data_images))

        if self.split == "train":
            self.data_npz = data_images[:self.val_ind]
            self.labels = labels[:self.val_ind]
            self.ground_true = ground_true[:self.val_ind]
        elif self.split == "valid":
            self.data_npz = data_images[self.val_ind:self.test_ind]
            self.labels = labels[self.val_ind:self.test_ind]
            self.ground_true = ground_true[self.val_ind:self.test_ind]            
        elif self.split == "test":
            self.data_npz = data_images[self.test_ind:]
            self.labels = labels[self.test_ind:]
            self.ground_true = ground_true[self.test_ind:]
            
        self.shuffle_seed = shuffle_seed
        self._name = name
        self._num_channels = num_channels

        self.transform = transform
        self.indices = range(len(self))

    @property
    def name(self):
        return self._name

    def has_labels(self):
        return True

    def num_classes(self, as_tensor=True):
        return None

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        img1 = Image.fromarray(self.data_npz[index1])
        if self.transform is not None:
            img1 = self.transform(img1)
        label1 = self.labels[index1]
        return img1, label1

    def __len__(self):
        return self.data_npz.shape[0]
