
import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor
from ..base import BaseSegmentor
from .imagelevel import ImageLevelContext
from .semanticlevel import SemanticLevelContext
from ...backbones import BuildActivation, BuildNormalization, constructnormcfg


'''ISNet'''
class ISNet(BaseSegmentor):
    def __init__(self, cfg, losses_cfg, mode):
        super(ISNet, self).__init__(cfg, losses_cfg, mode)
        align_corners, norm_cfg, act_cfg, head_cfg = self.align_corners, self.norm_cfg, self.act_cfg, cfg['head']
        # build bottleneck
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
        #     BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
        #     BuildActivation(act_cfg),
        # )
        self.losses_cfg = losses_cfg
        self.bottleneck = nn.SequentialCell(
            nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
        )
        # build image-level context module
        ilc_cfg = {
            'feats_channels': head_cfg['feats_channels'],
            'transform_channels': head_cfg['transform_channels'],
            'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
            'align_corners': align_corners,
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)
        # build semantic-level context module
        slc_cfg = {
            'feats_channels': head_cfg['feats_channels'],
            'transform_channels': head_cfg['transform_channels'],
            'concat_input': head_cfg['concat_input'],
            'norm_cfg': copy.deepcopy(norm_cfg),
            'act_cfg': copy.deepcopy(act_cfg),
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)
        # build decoder
        # self.decoder_stage1 = nn.Sequential(
        #     nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
        #     BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
        #     BuildActivation(act_cfg),
        #     nn.Dropout2d(head_cfg['dropout']),
        #     nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        # )
        self.decoder_stage1 = nn.SequentialCell(
            nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=False),
            BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            BuildActivation(act_cfg),
            nn.Dropout2d(head_cfg['dropout']),
            nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0, pad_mode='pad')
        )
        if head_cfg['shortcut']['is_on']:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(head_cfg['shortcut']['in_channels'], head_cfg['shortcut']['feats_channels'], kernel_size=1, stride=1, padding=0),
            #     BuildNormalization(constructnormcfg(placeholder=head_cfg['shortcut']['feats_channels'], norm_cfg=norm_cfg)),
            #     BuildActivation(act_cfg),
            # )
            # self.decoder_stage2 = nn.Sequential(
            #     nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut']['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            #     BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
            #     BuildActivation(act_cfg),
            #     nn.Dropout2d(head_cfg['dropout']),
            #     nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            # )
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(head_cfg['shortcut']['in_channels'], head_cfg['shortcut']['feats_channels'], kernel_size=1, stride=1, padding=0, pad_mode='pad'),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['shortcut']['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
            )
            self.decoder_stage2 = nn.SequentialCell(
                nn.Conv2d(head_cfg['feats_channels'] + head_cfg['shortcut']['feats_channels'], head_cfg['feats_channels'], kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=False),
                BuildNormalization(constructnormcfg(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg)),
                BuildActivation(act_cfg),
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0, pad_mode='pad')
            )
        else:
            # self.decoder_stage2 = nn.Sequential(
            #     nn.Dropout2d(head_cfg['dropout']),
            #     nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            # )
            self.decoder_stage2 = nn.SequentialCell(
                nn.Dropout2d(head_cfg['dropout']),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0, pad_mode='pad')
            )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
        # layer names for training tricks
        self.layer_names = ['backbone_net', 'bottleneck', 'ilc_net', 'slc_net', 'shortcut', 'decoder_stage1', 'decoder_stage2', 'auxiliary_decoder']
        self.shortcut_on = hasattr(self, 'shortcut')
    '''forward'''
    def construct(self, x, targets=None, losses_cfg=None):
        # img_size = x.size(2), x.size(3)
        img_size = x.shape[2], x.shape[3]
        test = Tensor(1.0)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to bottleneck
        feats = self.bottleneck(backbone_outputs[-1])
        # feed to image-level context module
        feats_il = self.ilc_net(feats)
        # feed to decoder stage1
        preds_stage1 = self.decoder_stage1(feats)
        # feed to semantic-level context module
        preds = preds_stage1
        # if preds_stage1.size()[2:] != feats.size()[2:]:
        if preds_stage1.shape[2:] != feats.shape[2:]:
            # preds = F.interpolate(preds_stage1, size=feats.size()[2:], mode='bilinear', align_corners=self.align_corners)
            preds = ops.interpolate(preds_stage1, size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        feats_sl = self.slc_net(feats, preds, feats_il)
        # feed to decoder stage2
        # if hasattr(self, 'shortcut'):
        if self.shortcut_on:
            shortcut_out = self.shortcut(backbone_outputs[0])
            # feats_sl = F.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            # feats_sl = torch.cat([feats_sl, shortcut_out], dim=1)
            feats_sl = ops.interpolate(feats_sl, size=shortcut_out.shape[2:], mode='bilinear', align_corners=self.align_corners)
            feats_sl = ops.cat([feats_sl, shortcut_out], axis=1)
        preds_stage2 = self.decoder_stage2(feats_sl)
        # return according to the mode
        if self.mode == 'TRAIN':
            # top2_loss_dict = {k: self.losses_cfg[k] for k in list(self.losses_cfg)[:2]}
            # outputs_dict = self.forwardtrain(
            #     predictions=preds_stage2,
            #     targets=targets,
            #     backbone_outputs=backbone_outputs,
            #     losses_cfg= top2_loss_dict,
            #     img_size=img_size,
            #     compute_loss=False,
            # )

            outputs_dict = self.forwardtrain(
                predictions=preds_stage2,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=self.losses_cfg,
                img_size=img_size,
                compute_loss=False,
            )

            # outputs_dict = self.forwardtrain_without_loss(
            #     predictions=preds_stage2,
            #     targets=targets,
            #     backbone_outputs=backbone_outputs,
            #     losses_cfg=self.losses_cfg,
            #     img_size=img_size,
            #     compute_loss=False,
            # )
            
            # output = self.test_base(predictions=targets)
            
            # outputs = targets

            preds_stage2 = outputs_dict['loss_cls']
            outputs_dict_new = {}

            for item in outputs_dict.items():
                key, value = item[0], item[1]
                if key != 'loss_cls':
                    outputs_dict_new[key] = value

            outputs_dict_new['loss_aux'] = outputs_dict['loss_aux']
            # del outputs_dict['loss_cls']
            # # preds_stage2 = outputs_dict.pop('loss_cls_stage2')
            # # preds_stage1 = F.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            preds_stage1 = ops.interpolate(preds_stage1, size=img_size, mode='bilinear', align_corners=self.align_corners)
            # # outputs_dict.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            outputs_dict_new.update({'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2})
            # outputs_dict_new['loss_cls_stage1'] = preds_stage1
            # outputs_dict_new['loss_cls_stage2'] = preds_stage2
            # # outputs_dict_new = {'loss_cls_stage1': preds_stage1, 'loss_cls_stage2': preds_stage2}
           
            loss, losses_log_dict =self.calculatelosses(
                predictions=outputs_dict_new, 
                targets=targets, 
                losses_cfg=self.losses_cfg
            )

            # loss = Tensor(1.0)
            return loss
            # return self.calculatelosses(
            #     predictions=outputs_dict, 
            #     targets=targets, 
            #     losses_cfg=losses_cfg
            # )
        return preds_stage2