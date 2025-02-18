'''
Function:
    Implementation of UPerNet
Author:
    Zhenchao Jin
'''
import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor

from collections import OrderedDict

from ..base import BaseSegmentor
from ..pspnet import PyramidPoolingModule
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''UPerNet'''
class UPerNet(BaseSegmentor):
    def __init__(self, cfg, losses_cfg, mode):
        super(UPerNet, self).__init__(cfg, losses_cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        self.cfg = cfg
        # build feature2pyramid
        if 'feature2pyramid' in head_cfg:
            from ..base import Feature2Pyramid
            head_cfg['feature2pyramid']['norm_cfg'] = norm_cfg.copy()
            self.feats_to_pyramid_net = Feature2Pyramid(**head_cfg['feature2pyramid'])
        # build pyramid pooling module
        ppm_cfg = {
            'in_channels': head_cfg['in_channels_list'][-1],
            'out_channels': head_cfg['feats_channels'],
            'pool_scales': head_cfg['pool_scales'],
            'align_corners': align_corners,
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.ppm_net = PyramidPoolingModule(**ppm_cfg)
        
        # build lateral convs
        act_cfg_copy = copy.deepcopy(act_cfg)
        if 'inplace' in act_cfg_copy: act_cfg_copy['inplace'] = False
        # self.lateral_convs = nn.CellList()
        # layers = []

        # layers = OrderedDict()
        # layers['conv1']
        
        # name_k = 0
        layers = nn.CellList()
        for in_channels in head_cfg['in_channels_list'][:-1]:
            # layer_name =  'lateral_convs_{}'.format(name_k)
            # layers[layer_name] = nn.SequentialCell(
            #             nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, has_bias=False),
            #             BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            #             BuildActivation(act_cfg_copy)
            #         )
            # name_k = name_k + 1
            layers.append(nn.SequentialCell(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, has_bias=False),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy))
            )
        # self.lateral_convs = nn.SequentialCell(layers)
        self.lateral_convs = layers

        # for in_channels in head_cfg['in_channels_list'][:-1]:
        #     self.lateral_convs.append(nn.SequentialCell(
        #         nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, has_bias=False),
        #         BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
        #         BuildActivation(act_cfg_copy),
        #     ))


        # build fpn convs
        # self.fpn_convs = nn.CellList()
        # for in_channels in [head_cfg['feats_channels'],] * len(self.lateral_convs):
        #     self.fpn_convs.append(nn.SequentialCell(
        #         nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
        #         BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
        #         BuildActivation(act_cfg_copy),
        #     ))

        # layers = OrderedDict()
        layers = nn.CellList()
        # name_k = 0
        for in_channels in [head_cfg['feats_channels'],] * len(self.lateral_convs):
            # layer_name =  'fpn_convs_{}'.format(name_k)
            # layers[layer_name] = nn.SequentialCell(
            #     nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            #     BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            #     BuildActivation(act_cfg_copy)
            # )
            # name_k = name_k + 1
            layers.append(nn.SequentialCell(
                nn.Conv2d(in_channels, head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg_copy)
            ))
        
        # self.fpn_convs = nn.SequentialCell(layers)
        self.fpn_convs = layers
        # self.fpn_convs = layers
        
        # build decoder
        self.decoder = nn.SequentialCell(
            nn.Conv2d(head_cfg['feats_channels'] * len(head_cfg['in_channels_list']), head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            # nn.Dropout2d(1-head_cfg['dropout']),
            nn.Dropout2d(p=(1-head_cfg['dropout'])),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # # build bottleneck decoder
        # self.bottleneck = nn.SequentialCell(
        #     nn.Conv2d(head_cfg['in_channels_list'][-1] + len(head_cfg['pool_scales']) * head_cfg['feats_channels'], head_cfg['feats_channels'], 3, padding=1, pad_mode='pad'),
        #     BuildNormalization(constructnormcfg(placeholder= head_cfg['feats_channels'], norm_cfg=norm_cfg)),
        #     BuildActivation(act_cfg)
        # )

        self.fpn_bottleneck = nn.SequentialCell(
            nn.Conv2d(len(head_cfg['in_channels_list']) * head_cfg['feats_channels'], head_cfg['feats_channels'], 3, padding=1, pad_mode='pad'),
            BuildNormalization(constructnormcfg(placeholder= head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg)
        )

        self.conv_seg = nn.Conv2d(head_cfg['feats_channels'], self.cfg['num_classes'], kernel_size=1, pad_mode='pad')
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'ppm_net', 'lateral_convs', 'feats_to_pyramid_net', 'decoder', 'auxiliary_decoder']
        
        # self.interpolate = nn.ResizeBilinear()
        self.con_op = ops.Concat(axis=1)
        # self.interpolate = nn.ResizeBilinear()
        self.losses_cfg = losses_cfg

        print("self.losses_cfg: ", len(self.losses_cfg))

        self.head_cfg = head_cfg

        self.print = ops.Print()
    '''forward'''
    # def construct(self, x, targets=None, losses_cfg=None):
    # def construct(self, x, segmentation=None, edge=None, losses_cfg=None):
        
    #     targets = {
    #         'segmentation': segmentation,
    #         'edge': edge
    #     }

    #     img_size = x.shape[2], x.shape[3]
    #     # feed to backbone network
    #     backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
    #     # feed to feats_to_pyramid_net
    #     # if hasattr(self, 'feats_to_pyramid_net'): backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
    #     if 'feature2pyramid' in self.head_cfg:
    #         backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
    #     # feed to pyramid pooling module
    #     ppm_out = self.ppm_net(backbone_outputs[-1])
    #     # apply fpn
    #     inputs = backbone_outputs[:-1]
    #     lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
    #     lateral_outputs.append(ppm_out)
    #     for i in range(len(lateral_outputs) - 1, 0, -1):
    #         prev_shape = lateral_outputs[i - 1].shape[2:]
    #         # lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
    #         # lateral_outputs[i - 1] = lateral_outputs[i - 1] + self.interpolate(lateral_outputs[i], size=prev_shape, align_corners=self.align_corners)
    #         lateral_outputs[i - 1] = lateral_outputs[i - 1] + ops.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
    #     fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
    #     fpn_outputs.append(lateral_outputs[-1])
    #     # fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
    #     # fpn_outputs = [self.interpolate(out, size=fpn_outputs[0].shape[2:], align_corners=self.align_corners) for out in fpn_outputs]
    #     fpn_outputs = [ops.interpolate(out, size=fpn_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
    #     # fpn_out = torch.cat(fpn_outputs, dim=1)
    #     fpn_out = self.con_op(fpn_outputs)
    #     # feed to decoder
    #     predictions = self.decoder(fpn_out)


    #     # predictions = self.interpolate(predictions, size=img_size, align_corners=self.align_corners)
    #     predictions = ops.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)

    #     # forward according to the mode
    #     if self.mode == 'TRAIN':
    #         loss, losses_log_dict = self.forwardtrain(
    #             predictions=predictions,
    #             targets=targets,
    #             backbone_outputs=backbone_outputs,
    #             losses_cfg=self.losses_cfg,
    #             img_size=img_size,
    #         )
    #         # print("loss: ", loss)
    #         # print("losses_log_dict: ", losses_log_dict)
    #         return loss
    #         # return loss, losses_log_dict
    #     return predictions
    # def psp_forward(self, inputs):
    #     """Forward function of PSP module."""
    #     x = inputs[-1]
    #     psp_outs = [x]
    #     psp_outs.extend(self.ppm_net(x))
    #     psp_outs = ops.cat(psp_outs, axis=1)
    #     output = self.bottleneck(psp_outs)

    #     return output

    def construct(self, x, segmentation=None, edge=None, losses_cfg=None):
        if 'cmx_fusion' not in self.cfg:
            cmx_fusion = False
        else:
            cmx_fusion = self.cfg['cmx_fusion']
        if cmx_fusion==False:
            targets = {
                'segmentation': segmentation,
                'edge': edge
            }
            img_size = x.shape[2], x.shape[3]
            # feed to backbone network
            backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
            # feed to feats_to_pyramid_net
            # if hasattr(self, 'feats_to_pyramid_net'): backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
            if 'feature2pyramid' in self.head_cfg:
                backbone_outputs = self.feats_to_pyramid_net(backbone_outputs)
            # feed to pyramid pooling module
            ppm_out = self.ppm_net(backbone_outputs[-1])
            # apply fpn
            inputs = backbone_outputs[:-1]
            lateral_outputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
            lateral_outputs.append(ppm_out)
            for i in range(len(lateral_outputs) - 1, 0, -1):
                prev_shape = lateral_outputs[i - 1].shape[2:]
                # lateral_outputs[i - 1] = lateral_outputs[i - 1] + F.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
                # lateral_outputs[i - 1] = lateral_outputs[i - 1] + self.interpolate(lateral_outputs[i], size=prev_shape, align_corners=self.align_corners)
                lateral_outputs[i - 1] = lateral_outputs[i - 1] + ops.interpolate(lateral_outputs[i], size=prev_shape, mode='bilinear', align_corners=self.align_corners)
            fpn_outputs = [self.fpn_convs[i](lateral_outputs[i]) for i in range(len(lateral_outputs) - 1)]
            fpn_outputs.append(lateral_outputs[-1])
            # fpn_outputs = [F.interpolate(out, size=fpn_outputs[0].size()[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            # fpn_outputs = [self.interpolate(out, size=fpn_outputs[0].shape[2:], align_corners=self.align_corners) for out in fpn_outputs]
            fpn_outputs = [ops.interpolate(out, size=fpn_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners) for out in fpn_outputs]
            # fpn_out = torch.cat(fpn_outputs, dim=1)
            fpn_out = self.con_op(fpn_outputs)
            # feed to decoder
            predictions = self.decoder(fpn_out)
            # predictions = self.interpolate(predictions, size=img_size, align_corners=self.align_corners)
            predictions = ops.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)

            # forward according to the mode
            if self.mode == 'TRAIN':
                loss, losses_log_dict = self.forwardtrain(
                    predictions=predictions,
                    targets=targets,
                    backbone_outputs=backbone_outputs,
                    losses_cfg=self.losses_cfg,
                    img_size=img_size,
                )
                # print("loss: ", loss)
                # print("losses_log_dict: ", losses_log_dict)
                return loss
                # return loss, losses_log_dict
            return predictions
        elif cmx_fusion:
            img_size = x.shape[2], x.shape[3]
            # backbone outputs
            inputs = self.backbone_net(x, segmentation)
            # build laterals
            laterals = [
                lateral_conv(inputs[i])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            # laterals.append(self.psp_forward(inputs))
            laterals.append(self.ppm_net(inputs[-1]))

            # build top-down path
            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += ops.interpolate(
                    laterals[i],
                    size=prev_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)

            # build outputs
            fpn_outs = [
                self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels - 1)
            ]
            # append psp feature
            fpn_outs.append(laterals[-1])

            for i in range(used_backbone_levels - 1, 0, -1):
                fpn_outs[i] = ops.interpolate(
                    fpn_outs[i],
                    size=fpn_outs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
            fpn_outs = ops.cat(fpn_outs, axis=1)
            output = self.fpn_bottleneck(fpn_outs)
            output = self.conv_seg(output)

            out = ops.interpolate(output, size=img_size, mode='bilinear', align_corners=False)
            # aux_head
            # if self.aux_head:
            #     aux_fm = self.aux_head(x[self.aux_index])
            #     aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)

            # loss
            if self.mode == 'TRAIN':
                # loss = self.CrossEntropyLoss_op(out, edge.long())
                # out_format = out.permute((0, 2, 3, 1))
                # out_format = out_format.view(-1, out_format.shape[-1])
                # edge = edge.view(-1)
                # loss = ops.cross_entropy(input=out_format, target=edge.long(), ignore_index=0, reduction='mean')
                # out = out.permute((0, 2, 3, 1))
                # loss = ops.cross_entropy(input=out.view(-1, out.shape[-1]), target=edge.view(-1).long(), ignore_index=0, reduction='mean')
                loss, _ = self.calculatelosses({'loss_cls': out}, edge, self.losses_cfg)
                # if self.aux_head:
                #     loss += self.aux_rate * self.criterion(aux_fm, label.long())
                return loss
            return out