import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import *

__all__ = [
    'PlainResNetClassifier'
]

class ResNetBackbone(ResNet):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNetBackbone, self).__init__(
            block=block,
            layers=layers,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class PlainResNetClassifier(nn.Module):

    name = 'PlainResNetClassifier'

    def __init__(self, num_cls=10, num_layers=18):
        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None

        # Model setup and weights initialization
        self.setup_net()

    def setup_net(self):

        kwargs = {}

        if self.num_layers == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
            self.pretrained_weights = ResNet18_Weights.IMAGENET1K_V1
        elif self.num_layers == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            self.pretrained_weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            raise Exception('ResNet Type not supported.')

        self.feature = ResNetBackbone(block, layers, **kwargs)
        self.classifier = nn.Linear(512 * block.expansion, self.num_cls)

    def setup_criteria(self):
        self.criterion_cls = nn.CrossEntropyLoss()

    def feat_init(self):
        init_weights = self.pretrained_weights.get_state_dict(progress=True)
        init_weights = OrderedDict({k.replace('module.', '').replace('feature.', ''): init_weights[k]
                                    for k in init_weights})

        self.feature.load_state_dict(init_weights, strict=False)

        load_keys = set(init_weights.keys())
        self_keys = set(self.feature.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print('missing keys: {}'.format(sorted(list(missing_keys))))
        print('unused_keys: {}'.format(sorted(list(unused_keys))))


