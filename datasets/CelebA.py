import torch.utils.data as data
from PIL import Image
from typing import Dict, Tuple
import collections
import os
import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from random import shuffle


CELEBA_ROOT = '/root/DATASET/CelebA'

class CelebA_SELECT(torchvision.datasets.ImageFolder):
    
    def __init__(self, transform=None):
        CelebA_Image_root = os.path.join(CELEBA_ROOT, 'Img')
        super().__init__(root=CelebA_Image_root, transform=transform)
        self.labels = self.get_labels()
        self.truncate(n_image=30)

    def get_labels(self):
        CelebA_Attr_file = os.path.join(CELEBA_ROOT, 'Anno/identity_CelebA.txt')
        Attr_type = 21   
        labels = []
        with open(CelebA_Attr_file, "r") as Attr_file:
            Attr_info = Attr_file.readlines()
            Attr_info = Attr_info[0:]
            for line in Attr_info:
                info = line.split()
                id = int(info[0].split('.')[0])
                label = int(info[1])
                labels.append(label)
        return labels
            
    def truncate(self, n_image):
        
        # O(n): get the count of each label
        count = torch.zeros(10180, 1).view(-1)
        n = len(self.labels)
        for i in range(n):
            label = self.labels[i]
            count[label] += 1
        
        # O(n): construct the logical index -> physical index mapping
        self.index_mapping = []
        for i in range(n):
            label = self.labels[i]
            if count[label] >= n_image: self.index_mapping.append(i)
        print('n samples=', len(self.index_mapping))
                
        # O(n): construct the old label -> new label mapping
        idx = count.ge(n_image).nonzero().view(-1)
        self.label_mapping = torch.zeros(10180, 1).view(-1)
        print('n identities =', len(idx))
        for j in range(len(idx)):
            old_label = idx[j]
            new_label = j
            self.label_mapping[old_label] = new_label
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        physical_index = self.index_mapping[index]
        image = super().__getitem__(physical_index)[0]
        label = self.label_mapping[self.labels[physical_index]].long()
        return image, label

    def __len__(self):
        return len(self.index_mapping)
    
    
def load_celeba(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 28, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load CelebA dataset (download if necessary) and split data into training,
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

    transform = transforms.Compose([
                               transforms.Resize(int(img_size*1.3)),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebA_SELECT(transform=transform) 
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, train_loader, train_loader, train_loader
