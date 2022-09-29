import os
import json
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


__all__ = [
    'BeeAnts'
]


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
}

class BeeAnts_DS(Dataset):

    def __init__(self, rootdir, dset='train', transform=None):
        self.img_root = None
        self.ann_root = None
        self.transform = transform
        self.dset = dset

        # BeeAnts doesn't have test yet.
        if dset == 'test':
            dset = 'val'

        self.data = []
        self.labels = []
        self.img_root = os.path.join(rootdir, 'hymenoptera_data')
        self.ann_root = os.path.join(rootdir, 'lists')
        ann_dir = os.path.join(self.ann_root, '{}.txt'.format(dset))

        self.load_data(ann_dir)

    def load_data(self, ann_dir):
        with open(ann_dir, 'r') as f:
            for line in f:
                line_sp = line.replace('\n', '').rsplit(' ', 1)
                file_id = line_sp[0]
                lab = int(line_sp[1])
                self.data.append(file_id)
                self.labels.append(lab)

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

class BeeAnts(pl.LightningDataModule):
    def __init__(self, conf):
        self.conf = conf
        self.prepare_data_per_node = True 
        self._log_hyperparams = False

        print("Loading data...")
        self.dset_tr = BeeAnts_DS(rootdir=self.conf.dataset_root,
                                  dset='train',
                                  transform=data_transforms['train'])

        self.dset_te = BeeAnts_DS(rootdir=self.conf.dataset_root,
                                  dset='val',
                                  transform=data_transforms['val'] )

        self.dset_te = BeeAnts_DS(rootdir=self.conf.dataset_root,
                                  dset='val',
                                  transform=data_transforms['val'])

        _, self.train_class_counts = self.dset_tr.class_counts_cal()

        print("Done.")

    def train_dataloader(self):
        return DataLoader(
            self.dset_tr, batch_size=self.conf.batch_size, shuffle=True, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dset_te, batch_size=self.conf.batch_size, shuffle=False, 
            pin_memory=True, num_workers=self.conf.num_workers, drop_last=False
        )