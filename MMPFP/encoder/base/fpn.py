
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


class FPN(nn.Cell):
    def __init__(self, in_channels_list, out_channels, upsample_cfg=dict(mode='nearest'), norm_cfg=None, act_cfg=None):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.upsample_cfg = upsample_cfg
        self.lateral_convs = nn.CellList()
        self.fpn_convs = nn.CellList()
        if 'inplace' in act_cfg: act_cfg['inplace'] = False
        for i in range(0, len(in_channels_list)):
            l_conv = nn.SequentialCell(
                nn.Conv2d(in_channels_list[i], out_channels, kernel_size=1, stride=1, padding=0, has_bias=False, pad_mode='pad'),
                BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            fpn_conv = nn.SequentialCell(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode='pad'),
                BuildNormalization(constructnormcfg(placeholder=out_channels, norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    def construct(self, inputs):
        assert len(inputs) == len(self.in_channels_list)
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] += ops.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
        # build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # return
        return tuple(outs)