
# import torch
# import torch.nn as nn

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor

from ...backbones import BuildNormalization, constructnormcfg

class Feature2Pyramid(nn.Cell):
    def __init__(self, embed_dim, rescales=[4, 2, 1, 0.5], norm_cfg=None):
        super(Feature2Pyramid, self).__init__()
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.SequentialCell(
                    # nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2, pad_mode='pad'),
                    nn.Conv2dTranspose(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=2, pad_mode='pad'),
                    BuildNormalization(constructnormcfg(placeholder=embed_dim, norm_cfg=norm_cfg)),
                    nn.GELU(),
                    nn.Conv2dTranspose(embed_dim, embed_dim, kernel_size=2, stride=2, pad_mode='pad'),
                )
                # self.upsample_4x = nn.Conv2dTranspose(embed_dim, embed_dim, 2, stride=2, pad_mode='pad', has_bias=False, weight_init='normal')
            elif k == 2:
                # self.upsample_2x = nn.SequentialCell([
                #     nn.Conv2dTranspose(embed_dim, embed_dim, kernel_size=2, stride=2, pad_mode='pad')
                # ]
                # )
                self.upsample_2x = nn.Conv2dTranspose(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=2, pad_mode='pad')
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')
        
        if len(self.rescales) == 4:
            if self.upsample_4x is not None:
                self.ops = [self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x]
            else:
                self.ops = [self.upsample_2x, self.identity, self.downsample_2x, self.downsample_4x]
        
        if len(self.rescales) == 3:
             self.ops = [self.upsample_4x, self.upsample_2x, self.identity]
    
    '''forward'''
    def construct(self, inputs):
        assert len(inputs) == len(self.rescales)
        outputs = []
        # if self.upsample_4x is not None:
        #     ops = [self.upsample_4x, self.upsample_2x, self.identity, self.downsample_2x]
        # else:
        #     ops = [self.upsample_2x, self.identity, self.downsample_2x, self.downsample_4x]
        for i in range(len(inputs)):
            outputs.append(self.ops[i](inputs[i]))
        
        # len_inputs = len(inputs)
        # if len_inputs == 1:
        #     outputs.append(ops[0](inputs[0]))

        # if len_inputs == 2:
        #     outputs.append(ops[0](inputs[0]))
        #     outputs.append(ops[1](inputs[1]))
        
        # if len_inputs == 3:
        #     outputs.append(ops[0](inputs[0]))
        #     outputs.append(ops[1](inputs[1]))
        #     outputs.append(ops[2](inputs[2]))
        
        # if len_inputs == 4:
        # TEMP_INPUT = inputs[0]
        # # CUR_OP = ops[0]
        # # TEMP1 = CUR_OP(TEMP_INPUT)
        # TEMP1 = self.upsample_4x(TEMP_INPUT)
        # outputs.append(TEMP1)
        # outputs.append(ops[1](inputs[1]))
        # outputs.append(ops[2](inputs[2]))
        # outputs.append(ops[3](inputs[3]))
        # outputs.append(ops[0](inputs[0]))
        return tuple(outputs)