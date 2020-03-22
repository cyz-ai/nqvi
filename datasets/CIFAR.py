from PIL import Image
from typing import Dict, Tuple
import collections
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from random import shuffle


CIFAR10_ROOT = '/root/DATASET/CIFAR10'

class saturation:
    def __init__(self, p=265):
        self.p=p

    def __call__(self, x):
        xcen = x*2-1
        out = torch.sign(xcen)* torch.abs(xcen)**(2/self.p) /2 + 1/2
        return out

def get_suffle_index(data_len, seed=0):
    subset_index = [i for i in range(data_len)]
    random.seed(seed)
    shuffle(subset_index)
    return subset_index

class CIFAR_SELECT(torchvision.datasets.CIFAR10):
    def __init__(self, label_list=None, train=True, transform=None, download=False):
        super().__init__(CIFAR10_ROOT, train, transform, None, download)
        self.label_list = label_list
        self.class_num = 10
        if self.label_list is not None:
            self.remap_dict = {}
            for i, label in enumerate(label_list):
                self.remap_dict[label] = i
            self.preprocess()
            self.class_num = len(label_list)

    def preprocess(self):
        selected_data = []
        selected_targets = []
        
        for i in range(len(self.data)):
            if self.targets[i] in self.label_list:
                data, target = self.data[i], self.targets[i]
                selected_data.append(data)
                selected_targets.append(self.remap_dict[target])
        
        self.data = selected_data
        self.targets = selected_targets
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def load_cifar10(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset (download if necessary) and split data into training,
        validation, and test sets.
    Args:
        downsample_pct: the proportion of the dataset to use for training,
            validation, and test
        train_pct: the proportion of the downsampled data to use for training
    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """

    train_transform = transforms.Compose(
        [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # saturation(p=256)
        ]               
    ) 
    test_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # saturation(p=256)
        ]               
    ) 
    
    # Load training set
    train_valid_set = CIFAR_SELECT(label_list, train=True, transform=train_transform, download=True)
    n_train_val = int(downsample_pct * len(train_valid_set))
    n_train= int(train_pct * n_train_val)
    n_valid = n_train_val - n_train
    train_set, valid_set, _ = torch.utils.data.random_split(
        train_valid_set,
        lengths=[
            n_train,
            n_valid,
            len(train_valid_set) - n_train_val,
        ],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    if train_pct < 1.0:
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        valid_loader = None

    # Load test set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    test_set_all = CIFAR_SELECT(label_list=label_list, train=False, download=True, transform=test_transform)
    subset_index = get_suffle_index(len(test_set_all))
    n_test = int(downsample_pct * len(test_set_all))
    test_set = torch.utils.data.Subset(test_set_all, indices=subset_index[0:n_test])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader, _