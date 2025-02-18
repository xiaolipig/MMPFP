
import copy
from .isnet import ISNet
from .upernet import UPerNet
from .fcn import FCN, DepthwiseSeparableFCN




def BuildSegmentor(segmentor_cfg, losses_cfg, mode):
    supported_segmentors = {
        'fcn': FCN,
        'isnet': ISNet,
        'upernet': UPerNet,
        'depthwiseseparablefcn': DepthwiseSeparableFCN,
    }
    print(segmentor_cfg['type'])
    selected_segmentor = supported_segmentors[segmentor_cfg['type']]
    segmentor_cfg = copy.deepcopy(segmentor_cfg)
    segmentor_cfg.pop('type')
   
    return selected_segmentor(segmentor_cfg, losses_cfg, mode)
    # return selected_segmentor(segmentor_cfg, losses_cfg, mode='TRAIN')