
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor
from ..base import SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ImageLevelContext'''
class ImageLevelContext(nn.Cell):
    def __init__(self, feats_channels, transform_channels, concat_input=False, align_corners=False, norm_cfg=None, act_cfg=None):
        super(ImageLevelContext, self).__init__()
        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.concat_input = concat_input
        if self.concat_input:
            # self.bottleneck = nn.Sequential(
            #     nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
            #     BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
            #     BuildActivation(act_cfg),
            # )
            self.bottleneck = nn.SequentialCell(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
                BuildNormalization(constructnormcfg(placeholder=feats_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
    '''forward'''
    def construct(self, x):
        x_global = self.global_avgpool(x)
        # x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        x_global = ops.interpolate(x_global, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        feats_il = self.correlate_net(x, ops.cat([x_global, x], axis=1))
        # if hasattr(self, 'bottleneck'):
        if self.concat_input:
            # feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
            feats_il = self.bottleneck(ops.cat([x, feats_il], axis=1))
        return feats_il