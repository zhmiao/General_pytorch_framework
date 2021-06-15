import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

dataset_obj = {}
def register_dataset_obj(name):

    """
    Dataset register
    """

    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


def get_dataset(name, rootdir, dset, **add_args):

    """
    Dataset getter
    """

    print('Getting dataset:\n class - {}\n rootdir - {}\n type - {} \n'.format(name, rootdir, dset))

    trans = data_transforms[dset]

    print('\n\n')

    return dataset_obj[name](rootdir, dset=dset, transform=trans, **add_args)


def load_dataset(name, rootdir, dset, batch_size=64, num_workers=1, **add_args):

    """
    Dataset loader
    """

    dataset = get_dataset(name, rootdir, dset, **add_args)

    assert len(dataset) > 0

    shuffle = True if dset == 'train' else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader


class BaseDataset(Dataset):

    def __init__(self, dset='train', transform=None):
        pass

    def load_data(self, ann_dir):
        pass

    def class_counts_cal(self):
        unique_labels, unique_counts = np.unique(self.labels, return_counts=True)
        return unique_labels, unique_counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_id = self.data[index]
        label = self.labels[index]
        file_dir = os.path.join(self.img_root, file_id)

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, file_dir
