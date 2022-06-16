# %%
import os 
import numpy as np
from PIL import Image
import torch

# %%
#########
# DATASET
#########
# Three components
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# %%
# TRANSFORMATION DEFINITION
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

# %%
# DATASET DEFINITION
class BeeAnts(Dataset):

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

# %%
dataset_tr = BeeAnts(rootdir='./supp/bee_ants', dset='train', transform=data_transforms['train'])
dataset_val = BeeAnts(rootdir='./supp/bee_ants', dset='val', transform=data_transforms['val'])

# %%
# DATALOADER 
# NOTE: DATALOADER doesn't need customization
# NOTE: PIN_MEMORY for GPU trainig
# NOTE: If sampler is not None, shuffle should be False
dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=32,
                           shuffle=True, num_workers=2, pin_memory=False,
                           drop_last=True, sampler=None)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=32,
                            shuffle=False, num_workers=2, pin_memory=False,
                            drop_last=False, sampler=None)

# %%
#########
# MODEL (NETWORK)
#########
# NOTE: We can load predefined pytorch models or we can directly write our own
from torchvision.models.resnet import resnet18

# %%
# First, load a predefined and pretrained model
network = resnet18(pretrained=True)

# %%
import torch.nn as nn
# Redefine classifier
num_input_features = network.fc.in_features
network.fc = nn.Linear(num_input_features, 2)

# %%
#########
# ALGORITHM
#########
# FIVE MAJOR COMPONENTS
# 0) DEVICE
# 1) LOSS
# 2) OPTIMIZERS
# 3) TRAINING STEPS
# 4) EVALUATION STEPS

# %%
### DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
### LOSS 
criterion = nn.CrossEntropyLoss()

# %%
### OPTIMIZER
import torch.optim as optim
opt_net = optim.SGD(params=network.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
# NOTE: gamma controls how the learning rate is reduced
scheduler = optim.lr_scheduler.StepLR(opt_net, step_size=4, gamma=0.1)

# %%
### TRAINING and validation
num_epochs = 12

for epoch in range(num_epochs):

    ### TRAIN
    # Set the network to train
    network.train()

    for i, batch in enumerate(dataloader_tr):
        # Load data
        data, labels, file_ids = batch

        # To device
        # NOTE: we can also do data.cuda()
        data, labels = data.to(device), labels.to(device)
        
        # No gradients for data and labels
        data.requires_grad = False
        labels.requires_grad = False

        ####################
        # Forward and loss #
        ####################
        # forward
        logits = network(data)
        # calculate loss 
        # NOTE: No need for softmax for pytorch cross-entropy loss
        loss = criterion(logits, labels)

        #############################
        # Backward and optimization #
        #############################
        # zero gradients for optimizer
        opt_net.zero_grad()
        # loss backpropagation
        loss.backward()
        # optimize step
        opt_net.step()
        print("Epoch: {}, Step: {}, Loss: {}".format(epoch, i, loss))

    # Step the lr scheduler
    scheduler.step()

    ### VALIDATION
    # Set the network to val
    network.eval()
    total_preds = []
    total_labels = []

    for batch in dataloader_val:

        # Load data
        data, labels, file_ids = batch

        # To device
        # NOTE: we can also do data.cuda()
        data, labels = data.to(device), labels.to(device)
        
        # No gradients for data and labels
        data.requires_grad = False
        labels.requires_grad = False

        ####################
        # Forward #
        ####################
        # forward
        logits = network(data)
        preds = logits.argmax(dim=1)

        total_preds.append(preds.detach().cpu().numpy())
        total_labels.append(labels.detach().cpu().numpy())
    
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    print("Validation Acc.: {}".format((total_preds == total_labels).sum() / len(total_preds)))

# %%
