import os
import json
import numpy as np

from .utils import register_dataset_obj, BaseDataset


@register_dataset_obj('BeeAnts')
class BeeAnts(BaseDataset):

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

