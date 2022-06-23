import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random

import torch
import torch.optim as optim

from .utils import register_algorithm, Algorithm, acc
from src.data.utils import load_dataset
from src.models.utils import get_model

import numpy as np


def load_data(args):

    """
    Dataloading function. This function can change alg by alg as well.
    """

    trainloader = load_dataset(name=args.dataset_name,
                               rootdir=args.dataset_root,
                               dset='train',
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

    testloader = load_dataset(name=args.dataset_name,
                              rootdir=args.dataset_root,
                              dset='test',
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)

    valloader = load_dataset(name=args.dataset_name,
                             rootdir=args.dataset_root,
                             dset='val',
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    return trainloader, testloader, valloader


@register_algorithm('PlainResNet')
class PlainResNet(Algorithm):

    """
    Overall training function.
    """

    name = 'PlainResNet'
    net = None
    opt_net = None
    scheduler = None

    def __init__(self, args):
        super(PlainResNet, self).__init__(args=args)

        # Training epochs and logging intervals
        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval

        #######################################
        # Setup data for training and testing #
        #######################################
        self.trainloader, self.testloader, self.valloader = load_data(args)
        _, self.train_class_counts = self.trainloader.dataset.class_counts_cal()

    def set_train(self):
        ###########################
        # Setup cuda and networks #
        ###########################
        # setup network
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.net = get_model(name=self.args.model_name, num_cls=self.args.num_classes,
                             weights_init=self.args.weights_init, num_layers=self.args.num_layers,
                             init_feat_only=True, parallel=self.args.parallel)

        self.set_optimizers()

    def set_eval(self):
        ###############################
        # Load weights for evaluation #
        ###############################
        self.logger.info('\nGetting {} model.'.format(self.args.model_name))
        self.logger.info('\nLoading from {}'.format(self.weights_path))
        self.net = get_model(name=self.args.model_name, num_cls=self.args.num_classes,
                             weights_init=self.weights_path, num_layers=self.args.num_layers,
                             init_feat_only=False)

    def set_optimizers(self):
        self.logger.info('** SETTING OPTIMIZERS!!! **')
        ######################
        # Optimization setup #
        ######################
        # Setup optimizer parameters for each network component
        net_optim_params_list = [
            {'params': self.net.feature.parameters(),
             'lr': self.args.lr_feature,
             'momentum': self.args.momentum_feature,
             'weight_decay': self.args.weight_decay_feature},
            {'params': self.net.classifier.parameters(),
             'lr': self.args.lr_classifier,
             'momentum': self.args.momentum_classifier,
             'weight_decay': self.args.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        self.opt_net = optim.SGD(net_optim_params_list)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt_net, step_size=self.args.step_size, gamma=self.args.gamma)

    def train(self):

        self.net.setup_critera()

        best_acc = 0.
        best_epoch = 0

        for epoch in range(self.num_epochs):

            # Training
            self.train_epoch(epoch)

            # Validation
            self.logger.info('\nValidation.')
            val_acc, _ = self.evaluate(self.valloader)#, test=False)
            if val_acc > best_acc:
                self.net.update_best()
                best_acc = val_acc
                best_epoch = epoch

        self.logger.info('\nBest Model Appears at Epoch {} with Mac Acc {:.3f}...'.format(best_epoch, best_acc * 100))
        self.save_model()

    def evaluate(self, loader, eval_output=False):

        outputs = self.evaluate_epoch(loader)

        eval_info, mac_acc, mic_acc = self.evaluate_metric(outputs[0], outputs[1])

        self.logger.info(eval_info)

        if eval_output:
            output_path = self.weights_path.replace('.pth', '_{}.npz').format('val' if loader == self.valloader else 'test')
            np.savez(output_path, preds=outputs[0], labels=outputs[1], logits=outputs[2], file_ids=outputs[3])

        return mac_acc, mic_acc

    def train_epoch(self, epoch):

        self.net.train()

        N = len(self.trainloader)

        tr_iter = iter(self.trainloader)

        for batch_idx in range(N):

            data, labels, _ = next(tr_iter)

            # log basic adda train info
            info_str = '[Train {}] Epoch: {} [batch {}/{} ({:.2f}%)] '.format(self.name, epoch, batch_idx,
                                                                              N, 100 * batch_idx / N)

            ########################
            # Setup data variables #
            ########################
            data, labels = data.cuda(), labels.cuda()
            data.requires_grad = False
            labels.requires_grad = False

            ####################
            # Forward and loss #
            ####################
            # forward
            feats = self.net.feature(data)
            logits = self.net.classifier(feats)
            # calculate loss
            loss = self.net.criterion_cls(logits, labels)

            #############################
            # Backward and optimization #
            #############################
            # zero gradients for optimizer
            self.opt_net.zero_grad()
            # loss backpropagation
            loss.backward()
            # optimize step
            self.opt_net.step()

            ###########
            # Logging #
            ###########
            if batch_idx % self.log_interval == 0:
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                # log update info
                info_str += 'Acc: {:0.1f} Xent: {:.3f}'.format(acc.item() * 100, loss.item())
                self.logger.info(info_str)

        self.scheduler.step()

    def evaluate_epoch(self, loader):
        self.net.eval()

        total_preds = []
        total_labels = []
        total_logits = []
        total_file_ids = []

        # Forward and record # correct predictions of each class
        with torch.set_grad_enabled(False):

            for data, labels, file_ids in tqdm(loader, total=len(loader)):

                # setup data
                data, labels = data.cuda(), labels.cuda()
                data.requires_grad = False
                labels.requires_grad = False

                # forward
                feats = self.net.feature(data)
                logits = self.net.classifier(feats)
                preds = logits.argmax(dim=1)

                # max_probs, preds = F.softmax(logits, dim=1).max(dim=1)

                total_preds.append(preds.detach().cpu().numpy())
                total_labels.append(labels.detach().cpu().numpy())
                total_logits.append(logits.detach().cpu().numpy())
                total_file_ids.append(np.array(file_ids))

        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)
        total_logits = np.concatenate(total_logits, axis=0)
        total_file_ids = np.concatenate(total_file_ids, axis=0)

        return (total_preds, total_labels, total_logits, total_file_ids)

    def evaluate_metric(self, total_preds, total_labels):
        class_acc, mac_acc, mic_acc, unique_eval_classes = acc(total_preds, total_labels, self.args.num_classes)

        eval_info = '{} Per-class evaluation results: \n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        for i in range(len(class_acc)):
            label = unique_eval_classes[i]
            eval_info += 'Class {} (tr counts {:>5}):'.format(label, self.train_class_counts[label])
            eval_info += 'Acc {:.3f} \n'.format(class_acc[i] * 100)

        eval_info += 'Macro Acc: {:.3f}; Micro Acc: {:.3f}\n'.format(mac_acc * 100, mic_acc * 100)
        return eval_info, mac_acc, mic_acc

    def save_model(self):
        os.makedirs(self.weights_path.rsplit('/', 1)[0], exist_ok=True)
        self.logger.info('Saving to {}'.format(self.weights_path))
        self.net.save(self.weights_path)
