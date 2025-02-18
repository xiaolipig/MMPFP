
import copy
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import nn, ops, Parameter, Tensor

from ...losses import BuildLoss
from ...backbones import BuildBackbone, BuildActivation, BuildNormalization, constructnormcfg


class BaseSegmentor(nn.Cell):
    def __init__(self, cfg, losses_cfg, mode):
        super(BaseSegmentor, self).__init__()
        self.cfg = cfg
        self.losses_cfg = losses_cfg
        self.mode = mode
        assert self.mode in ['TRAIN', 'TEST']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.norm_cfg, self.act_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        # backbone_cfg = cfg['backbone']
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
            # backbone_cfg.update({'norm_cfg': self.norm_cfg})
        self.backbone_net = BuildBackbone(backbone_cfg)

        self.print = ops.Print()
        
        self.layer_names = []


        # self.cls_weight_edge = Parameter(Tensor(np.zeros((1, 2)), mindspore.float32), name="cls_weight_edge", requires_grad=False)

    '''forward'''
    def construct(self, x, targets=None, losses_cfg=None):
        raise NotImplementedError('not to be implemented')
    '''forward when mode = `TRAIN`'''
    def constructtrain_OLD(self, predictions, targets, backbone_outputs, losses_cfg, img_size, compute_loss=True):
        # predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
        # predictions = ops.interpolate(predictions, None, None, sizes=img_size, mode='bilinear', coordinate_transformation_mode='align_corners')
        interpolate = nn.ResizeBilinear()
        predictions = interpolate(predictions, size=img_size, align_corners=self.align_corners)
        outputs_dict = {'loss_cls': predictions}
        self.print("********************************************************")
        self.print("has auxiliary_decoder: ", hasattr(self, 'auxiliary_decoder'))
        self.print("********************************************************")

        # if 'auxiliary_decoder' in self.layer_names:

        # # if hasattr(self, 'auxiliary_decoder'):
            
        #     # ops.Print()("has auxiliary_decoder")
            

        #     backbone_outputs = backbone_outputs[:-1]
        #     if isinstance(self.auxiliary_decoder, nn.CellList):
        #         assert len(backbone_outputs) >= len(self.auxiliary_decoder)
        #         backbone_outputs = backbone_outputs[-len(self.auxiliary_decoder):]
        #         for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoder)):
        #             predictions_aux = dec(out)
        #             # predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
        #             predictions_aux = interpolate(predictions_aux, size=img_size, align_corners=self.align_corners)
        #             outputs_dict[f'loss_aux{idx+1}'] = predictions_aux
        #     else:
        #         predictions_aux = self.auxiliary_decoder(backbone_outputs[-1])
        #         # predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
        #         predictions_aux = interpolate(predictions_aux, size=img_size, align_corners=self.align_corners)
        #         outputs_dict = {'loss_cls': predictions, 'loss_aux': predictions_aux}


        # predictions_aux = self.auxiliary_decoder(backbone_outputs[-1])
        # # predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
        # predictions_aux = interpolate(predictions_aux, size=img_size, align_corners=self.align_corners)
        # outputs_dict = {'loss_cls': predictions, 'loss_aux': predictions_aux}


        if not compute_loss: 
            return outputs_dict
        
        print('*************************')
        print(outputs_dict)
        print('*************************')

        return self.calculatelosses(
            predictions=outputs_dict, 
            targets=targets, 
            losses_cfg=self.losses_cfg
        )
    
    # def _CrossEntropyLoss(self, prediction, target, scale_factor=1.0, weight=None, ignore_index=255, reduction='mean', lowest_loss_value=None, label_smoothing=None):
    #     # calculate the loss
    #     ce_args = {
    #         'weight': weight,
    #         'ignore_index': ignore_index,
    #         'reduction': reduction,
    #     }
    #     if label_smoothing is not None:
    #         ce_args.update({'label_smoothing': label_smoothing})
    #     # loss = F.cross_entropy(prediction, target.long(), **ce_args)
    #     # loss = ops.cross_entropy(prediction, target.long(), **ce_args)
    #     loss = ops.cross_entropy(prediction, target, **ce_args)
    #     # scale the loss
    #     loss = loss * scale_factor
    #     # return the final loss
    #     if lowest_loss_value:
    #         # return torch.abs(loss - lowest_loss_value) + lowest_loss_value
    #         return ops.abs(loss - lowest_loss_value) + lowest_loss_value
    #     # print('loss: ', loss)
    #     return loss
        
    def constructtrain(self, predictions, targets, backbone_outputs, losses_cfg, img_size, compute_loss=True):
        
        # interpolate = nn.ResizeBilinear()
        # predictions = interpolate(predictions, size=img_size, align_corners=self.align_corners)
        predictions = ops.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
        outputs_dict = {'loss_cls': predictions}
        backbone_outputs = backbone_outputs[:-1]


        if ('auxiliary_decoder' in self.layer_names) and len(backbone_outputs) and len(self.auxiliary_decoder):
            # and hasattr(self, "auxiliary_decoder"):
            if type(self.auxiliary_decoder) == nn.CellList:
                assert len(backbone_outputs) >= len(self.auxiliary_decoder)
                backbone_outputs = backbone_outputs[-len(self.auxiliary_decoder):]
                for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoder)):
                    predictions_aux = dec(out)
                    # predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                    predictions_aux = ops.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                    outputs_dict[f'loss_aux{idx+1}'] = predictions_aux
            else:
                predictions_aux = self.auxiliary_decoder(backbone_outputs[-1])
                # predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                predictions_aux = ops.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                outputs_dict = {'loss_cls': predictions, 'loss_aux': predictions_aux}

                
        
        
        if len(outputs_dict) != len(self.losses_cfg):
            return outputs_dict

        # calculate loss according to losses_cfg
        # print('len(predictions): ', len(outputs_dict))
        # print('len(losses_cfg): ', len(self.losses_cfg))
        
        assert len(outputs_dict) == len(self.losses_cfg), 'length of losses_cfg should be equal to predictions'

        # # loss = self._CrossEntropyLoss(outputs_dict['loss_cls'], targets['segmentation'])

        # # loss = BuildLoss('celoss')(outputs_dict['loss_cls'], targets['segmentation'])
        # loss_cfg = { 'celoss': {'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'} }
        # loss = self.calculateloss(outputs_dict['loss_cls'], targets['segmentation'], loss_cfg)

        # return loss, {}
        
        # update output dict name, so that losses_cfg name is the same as that in output dict
        final_output_dict = {}
        finale_output_name = list(self.losses_cfg.keys())
        i = 0
        for value in outputs_dict.values():
            final_output_dict[finale_output_name[i]] =  value
            i = i + 1

        # return self.calculatelosses(
        #     predictions=final_output_dict, 
        #     targets=targets, 
        #     losses_cfg=self.losses_cfg
        # )
        return self.calculatelosses(
            predictions=final_output_dict, 
            targets=targets, 
            losses_cfg=losses_cfg
        )
    def test_base(self, predictions):
        return predictions
    def constructtrain_without_loss(self, predictions, targets, backbone_outputs, losses_cfg, img_size, compute_loss=True):
        
        return predictions


    '''forward when mode = `TEST`'''
    def constructtest(self):
        raise NotImplementedError('not to be implemented')
    '''transform inputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['series'] in ['hrnet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    '''return all layers with learnable parameters'''
    def alllayers(self):
        assert len(self.layer_names) == len(set(self.layer_names))
        require_training_layers = {}
        for layer_name in self.layer_names:
            if hasattr(self, layer_name) and layer_name not in ['backbone_net']:
                require_training_layers[layer_name] = getattr(self, layer_name)
            elif hasattr(self, layer_name) and layer_name in ['backbone_net']:
                if hasattr(getattr(self, layer_name), 'nonzerowdlayers'):
                    assert hasattr(getattr(self, layer_name), 'zerowdlayers')
                    tmp_layers = []
                    for key, value in getattr(self, layer_name).zerowdlayers().items():
                        tmp_layers.append(value)
                    require_training_layers.update({f'{layer_name}_zerowd': nn.SequentialCell(*tmp_layers)})
                    tmp_layers = []
                    for key, value in getattr(self, layer_name).nonzerowdlayers().items():
                        tmp_layers.append(value)
                    require_training_layers.update({f'{layer_name}_nonzerowd': nn.SequentialCell(*tmp_layers)})
                else:
                    require_training_layers[layer_name] = getattr(self, layer_name)
            elif hasattr(self, layer_name):
                raise NotImplementedError(f'layer name {layer_name} error')
        return require_training_layers
    '''set auxiliary decoder as attribute'''
    def setauxiliarydecoder(self, auxiliary_cfg):
        norm_cfg, act_cfg, num_classes = self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes']
        if auxiliary_cfg is None: return
        if isinstance(auxiliary_cfg, dict):
            auxiliary_cfg = [auxiliary_cfg]
        self.auxiliary_decoder = nn.CellList()
        for aux_cfg in auxiliary_cfg:
            num_convs = aux_cfg.get('num_convs', 1)
            dec = []
            for idx in range(num_convs):
                if idx == 0:
                    dec += [nn.Conv2d(aux_cfg['in_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),]
                else:
                    dec += [nn.Conv2d(aux_cfg['out_channels'], aux_cfg['out_channels'], kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),]
                dec += [
                    BuildNormalization(constructnormcfg(placeholder=aux_cfg['out_channels'], norm_cfg=norm_cfg)),
                    BuildActivation(act_cfg)
                ]
                if 'upsample' in aux_cfg:
                    # dec += [nn.ResizeBilinear(**aux_cfg['upsample'])]
                    dec += [ops.interpolate(**aux_cfg['upsample'])]
            dec.append(nn.Dropout2d(1.0-aux_cfg['dropout']))
            if num_convs > 0:
                dec.append(nn.Conv2d(aux_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0, pad_mode='pad', ))
            else:
                dec.append(nn.Conv2d(aux_cfg['in_channels'], num_classes, kernel_size=1, stride=1, padding=0, pad_mode='pad', ))
            dec = nn.SequentialCell(*dec)
            self.auxiliary_decoder.append(dec)
        if len(self.auxiliary_decoder) == 1:
            self.auxiliary_decoder = self.auxiliary_decoder[0]
    '''freeze normalization'''
    def freezenormalization(self):
        for module in self.modules():
            if type(module) in BuildNormalization(only_get_all_supported=True):
                # module.eval()
                module.set_train(False)
    '''calculate the losses'''
    def calculatelosses_old(self, predictions, targets, losses_cfg, map_preds_to_tgts_dict=None):
        # parse targets
        target_seg = targets['segmentation']
        print('*************************')
        print('target_seg: ', target_seg.shape)
        print('*************************')
    
        # target_edge = ops.Zeros()(target_seg.shape, target_seg.dtype)
        cls_weight_edge = ops.Ones()((1,2), mindspore.float32)
        if 'edge' in targets:
            # target_edge = targets['edge']
            # num_neg_edge, num_pos_edge = torch.sum(target_edge == 0, dtype=torch.float), torch.sum(target_edge == 1, dtype=torch.float)
            # num_neg_edge, num_pos_edge = ops.ReduceSum(target_edge == 0, dtype=mindspore.float32), ops.ReduceSum(target_edge == 1, dtype=mindspore.float32)
            num_neg_edge, num_pos_edge = ops.ReduceSum()((targets['edge'] == 0).astype(mindspore.float32)), ops.ReduceSum()((targets['edge'] == 1).astype(mindspore.float32))

            # num_neg_edge, num_pos_edge = np.sum((targets['edge'] == 0).asnumpy()).astype(np.float32), np.sum((targets['edge'] == 1).asnumpy()).astype(np.float32)

            weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
            # cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(target_edge)
            # cls_weight_edge = Tensor([weight_neg_edge.asnumpy(), weight_pos_edge.asnumpy()]).astype(mindspore.float32)
            cls_weight_edge = [weight_neg_edge, weight_pos_edge]
            print("-------------------------------")
        # calculate loss according to losses_cfg
        print('len(predictions): ', len(predictions))
        print('len(losses_cfg): ', len(self.losses_cfg))
        assert len(predictions) == len(self.losses_cfg), 'length of losses_cfg should be equal to predictions'
        losses_log_dict = {}
        for loss_name, loss_cfg in self.losses_cfg.items():
            if 'edge' in loss_name:
                # loss_cfg = copy.deepcopy(loss_cfg)
                loss_cfg = loss_cfg
                loss_cfg_keys = loss_cfg.keys()
                for key in loss_cfg_keys:
                    loss_cfg[key].update({'weight': cls_weight_edge})
            if map_preds_to_tgts_dict is None:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets['edge'] if 'edge' in loss_name else target_seg,
                    # target=target_seg,
                    loss_cfg=loss_cfg,
                )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets[map_preds_to_tgts_dict[loss_name]],
                    loss_cfg=loss_cfg,
                )
        loss = 0
        for key, value in losses_log_dict.items():
            value = value.mean()
            loss += value
            losses_log_dict[key] = value
        losses_log_dict.update({'total': loss})
        # convert losses_log_dict
        for key, value in losses_log_dict.items():
            # if dist.is_available() and dist.is_initialized():
            #     value = value.data.clone()
            #     dist.all_reduce(value.div_(dist.get_world_size()))
            #     losses_log_dict[key] = value.item()
            # else:
            #     losses_log_dict[key] = torch.Tensor([value.item()]).type_as(loss)
            # losses_log_dict[key] = mindspore.Tensor([value.item()]).astype(loss.type)
            losses_log_dict[key] = mindspore.Tensor([value.asnumpy()]).astype(loss.dtype)
        # return the loss and losses_log_dict
        # return loss, losses_log_dict
        return loss, losses_log_dict
    
    def calculatelosses(self, predictions, targets, losses_cfg, map_preds_to_tgts_dict=None):
        # parse targets
        if isinstance(targets, dict):
            target_seg = targets['segmentation']
        else:
            target_seg = targets
        # print('*************************')
        # print('target_seg: ', target_seg.shape)
        # print('*************************')

        target_edge = ops.Zeros()(target_seg.shape, target_seg.dtype)
        # cls_weight_edge = ops.Ones()((2,1), mindspore.float32) #it may changed according to the classes number
        cls_weight_edge = ops.Ones()((4*512*512,1), mindspore.float32)
        if isinstance(targets, dict):
            if 'edge' in targets:
                # target_edge = targets['edge']
                # num_neg_edge, num_pos_edge = torch.sum(target_edge == 0, dtype=torch.float), torch.sum(target_edge == 1, dtype=torch.float)
                # num_neg_edge, num_pos_edge = ops.ReduceSum(target_edge == 0, dtype=mindspore.float32), ops.ReduceSum(target_edge == 1, dtype=mindspore.float32)
                num_neg_edge, num_pos_edge = ops.ReduceSum()((targets['edge'] == 0).astype(mindspore.float32)), ops.ReduceSum()((targets['edge'] == 1).astype(mindspore.float32))

                # num_neg_edge, num_pos_edge = np.sum((targets['edge'] == 0).asnumpy()).astype(np.float32), np.sum((targets['edge'] == 1).asnumpy()).astype(np.float32)

                weight_pos_edge, weight_neg_edge = num_neg_edge / (num_pos_edge + num_neg_edge), num_pos_edge / (num_pos_edge + num_neg_edge)
                # cls_weight_edge = torch.Tensor([weight_neg_edge, weight_pos_edge]).type_as(target_edge)
                # cls_weight_edge = Tensor([weight_neg_edge.asnumpy(), weight_pos_edge.asnumpy()]).astype(mindspore.float32)
                cls_weight_edge[0] = weight_pos_edge
                cls_weight_edge[1] = weight_neg_edge
                # cls_weight_edge = [weight_neg_edge, weight_pos_edge]
        #     print("-------------------------------")
        # print('cls_weight_edge: ', cls_weight_edge)
        # calculate loss according to losses_cfg
        # print('len(predictions): ', len(predictions))
        # print('len(losses_cfg): ', len(self.losses_cfg))
       
        # assert len(predictions) == len(self.losses_cfg), 'length of losses_cfg should be equal to predictions'
        
        # losses_log_dict = {}
        # for loss_name, loss_cfg in self.losses_cfg.items():
        #    if loss_name == 'loss_cls':
        #     losses_log_dict[loss_name]  = self.calculateloss(
        #                     prediction=predictions[loss_name],
        #                     target= target_seg,
        #                     # target=target_seg,
        #                     loss_cfg=loss_cfg,
        #                 )
        
        

        losses_log_dict = {}
        for loss_name, loss_cfg in self.losses_cfg.items():

        # for loss_name, loss_cfg in losses_cfg.items():

        # losses_stages_cfg = list(self.losses_cfg.items())[:2]
        # for loss_name, loss_cfg in losses_stages_cfg:

            if 'edge' in loss_name:
                # loss_cfg = copy.deepcopy(loss_cfg)
                loss_cfg = loss_cfg
                loss_cfg_keys = loss_cfg.keys()
                for key in loss_cfg_keys:
                    loss_cfg[key].update({'weight': cls_weight_edge})
            if map_preds_to_tgts_dict is None:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    # target=targets['edge'] if 'edge' in loss_name else target_seg,
                    target=target_edge if 'edge' in loss_name else target_seg,
                    # target=target_seg,
                    loss_cfg=loss_cfg,
                )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name],
                    target=targets[map_preds_to_tgts_dict[loss_name]],
                    loss_cfg=loss_cfg,
                )
        loss = 0
        for key, value in losses_log_dict.items():
            value = value.mean()
            loss += value
            losses_log_dict[key] = value
        losses_log_dict.update({'total': loss})
        # losses_log_dict.update({'total': loss})
        # convert losses_log_dict
        # for key, value in losses_log_dict.items():
            # if dist.is_available() and dist.is_initialized():
            #     value = value.data.clone()
            #     dist.all_reduce(value.div_(dist.get_world_size()))
            #     losses_log_dict[key] = value.item()
            # else:
            #     losses_log_dict[key] = torch.Tensor([value.item()]).type_as(loss)
            # losses_log_dict[key] = mindspore.Tensor([value.item()]).astype(loss.type)
            # losses_log_dict[key] = mindspore.Tensor([value.asnumpy()]).astype(loss.dtype)
            # losses_log_dict[key] = value
        # return the loss and losses_log_dict
        # return loss, losses_log_dict
        return loss, losses_log_dict

    '''calculate the loss'''
    def calculateloss_old(self, prediction, target, loss_cfg):
        # format prediction
        if prediction.dim() == 4:
            # prediction_format = prediction.permute((0, 2, 3, 1)).contiguous()
            prediction_format = prediction.permute((0, 2, 3, 1))
        elif prediction.dim() == 3:
            # prediction_format = prediction.permute((0, 2, 1)).contiguous()
            prediction_format = prediction.permute((0, 2, 1))
        else:
            prediction_format = prediction
        prediction_format = prediction_format.view(-1, prediction_format.size(-1))
        prediction_format = prediction_format.view(-1, prediction_format.shape[-1])

        # calculate the loss
        loss = 0
        for key, value in loss_cfg.items():
            if (key in ['binaryceloss']) and hasattr(self, 'onehot'):
                prediction_iter = prediction_format
                target_iter = self.onehot(target, self.cfg['num_classes'])
            elif key in ['diceloss', 'lovaszloss', 'kldivloss', 'l1loss', 'cosinesimilarityloss']:
                prediction_iter = prediction
                target_iter = target
            else:
                prediction_iter = prediction_format
                target_iter = target.view(-1)
            loss += BuildLoss(key)(
                prediction=prediction_iter, 
                target=target_iter, 
                **value
            )
        # return the loss
        return loss

    
    def calculateloss(self, prediction, target, loss_cfg):
        # format prediction
        if len(prediction.shape) == 4:
            # prediction_format = prediction.permute((0, 2, 3, 1)).contiguous()
            prediction_format = prediction.permute((0, 2, 3, 1))
        elif len(prediction.shape) == 3:
            # prediction_format = prediction.permute((0, 2, 1)).contiguous()
            prediction_format = prediction.permute((0, 2, 1))
        else:
            prediction_format = prediction
        # prediction_format = prediction_format.view(-1, prediction_format.size(-1))
        prediction_format = prediction_format.view(-1, prediction_format.shape[-1])

        # calculate the loss
        loss = 0
        for key, value in loss_cfg.items():
            if (key in ['binaryceloss']) and hasattr(self, 'onehot'):
                prediction_iter = prediction_format
                target_iter = self.onehot(target, self.cfg['num_classes'])
                # continue
            elif key in ['diceloss', 'lovaszloss', 'kldivloss', 'l1loss', 'cosinesimilarityloss']:
                prediction_iter = prediction
                target_iter = target
            else:
                prediction_iter = prediction_format
                target_iter = target.view(-1)
            loss += BuildLoss(key)(
                prediction=prediction_iter, 
                target=target_iter, 
                **value
            )
        # return the loss
        return loss