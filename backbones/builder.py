
import copy
from .resnet import BuildResNet
from .resnest import BuildResNeSt

from .convnextv2 import BuildConvNeXtv2

from .mobilenet import BuildMobileNet

from .vit import BuildVisionTransformer

#from .timmwrapper import BuildTIMMBackbone
from .mae import BuildMAE


def BuildBackbone(backbone_cfg):
    supported_backbones = {

        'resnet': BuildResNet,
        'resnest': BuildResNeSt,

        'convnextv2': BuildConvNeXtv2,
        

        'mobilenet': BuildMobileNet,

        'vit': BuildVisionTransformer,

        'mae': BuildMAE, 

    }
    selected_backbone = supported_backbones[backbone_cfg['series']]
    backbone_cfg = copy.deepcopy(backbone_cfg)
    backbone_cfg.pop('series')
    return selected_backbone(backbone_cfg)