import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from Bio import PDB

class RepVGGBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deploy=False, dropout_rate=0.3):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=True)
        else:
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding, has_bias=False)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, pad_mode='pad', padding=0, has_bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def construct(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))
        else:
            out = self.conv3x3(x) + self.conv1x1(x) + self.bn(x)
            out = self.relu(out)
            return self.dropout(out)  # train in Dropout

class RepVGG(nn.Cell):
    def __init__(self, in_channels, num_classes=10, deploy=False, dropout_rate=0.5):
        super(RepVGG, self).__init__()
        self.stage1 = RepVGGBlock(in_channels, 64, deploy=deploy)
        self.stage2 = RepVGGBlock(64, 128, stride=2, deploy=deploy)
        self.stage3 = RepVGGBlock(128, 256, stride=2, deploy=deploy)
        self.stage4 = RepVGGBlock(256, 512, stride=2, deploy=deploy)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.fc = nn.Dense(512, num_classes)

    def construct(self, x):
        x = x.expand_dims(1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)
