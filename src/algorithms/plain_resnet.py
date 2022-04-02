import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from .utils import register_algorithm, acc
from src.models.utils import get_model



@register_algorithm('PlainResNet')
class PlainResNet(pl.LightningModule):

    """
    Overall training function.
    """

    name = 'PlainResNet'

    def __init__(self, conf, train_class_counts, **kwargs):
        super().__init__()
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.net = get_model(name=self.hparams.model_name, num_cls=self.hparams.num_classes,
                             num_layers=self.hparams.num_layers)
        self.train_class_counts = train_class_counts

    def configure_optimizers(self):
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': self.hparams.lr_feature,
             'momentum': self.hparams.momentum_feature,
             'weight_decay': self.hparams.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': self.hparams.lr_classifier,
             'momentum': self.hparams.momentum_classifier,
             'weight_decay': self.hparams.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        optimizer = torch.optim.SGD(net_optim_params_list)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)   
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.best_acc = 0
        self.net.feat_init()
        self.net.setup_criteria()

    def training_step(self, batch, batch_idx):

        data, labels, _ = batch

        # forward
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        # calculate loss
        loss = self.net.criterion_cls(logits, labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels, file_ids = batch
        # forward
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)

        return (preds.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                logits.detach().cpu().numpy(), 
                file_ids)

    def validation_epoch_end(self, outputs):
        total_preds = np.concatenate([x[0] for x in outputs], axis=0)
        total_labels = np.concatenate([x[1] for x in outputs], axis=0)

        class_acc, mac_acc, mic_acc, unique_eval_classes = acc(total_preds, total_labels, self.hparams.num_classes)

        self.log('valid_mac_acc', mac_acc)
        self.log('valid_mic_acc', mic_acc)
        for i in range(len(class_acc)):
            label = unique_eval_classes[i]
            self.log('Class {} (tr counts {:>5}) Acc'.format(label, self.train_class_counts[label]),
                     class_acc[i] * 100)

