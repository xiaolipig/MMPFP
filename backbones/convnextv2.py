
import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from .bricks.normalization.grn import GRN
from .bricks.dropout.droppath import DropPath
from .bricks import BuildNormalization, BuildActivation, constructnormcfg


'''DEFAULT_MODEL_URLS'''
DEFAULT_MODEL_URLS = {
    'convnextv2_atto_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_atto_1k_224_fcmae.pt',
    'convnextv2_femto_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_femto_1k_224_fcmae.pt',
    'convnextv2_pico_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_pico_1k_224_fcmae.pt',
    'convnextv2_nano_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt',
    'convnextv2_tiny_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt',
    'convnextv2_base_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt',
    'convnextv2_large_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt',
    'convnextv2_huge_1k_224_fcmae': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt',
    'convnextv2_atto_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
    'convnextv2_femto_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
    'convnextv2_pico_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
    'convnextv2_nano_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_nano_1k_224_ema.pt',
    'convnextv2_tiny_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt',
    'convnextv2_base_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt',
    'convnextv2_large_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt',
    'convnextv2_huge_1k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt',
    'convnextv2_nano_22k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt',
    'convnextv2_nano_22k_384_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt',
    'convnextv2_tiny_22k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt',
    'convnextv2_tiny_22k_384_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt',
    'convnextv2_base_22k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt',
    'convnextv2_base_22k_384_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt',
    'convnextv2_large_22k_224_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt',
    'convnextv2_large_22k_384_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt',
    'convnextv2_huge_22k_384_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt',
    'convnextv2_huge_22k_512_ema': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt',
}
'''AUTO_ASSERT_STRUCTURE_TYPES'''
AUTO_ASSERT_STRUCTURE_TYPES = {}


'''LayerNorm'''
class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super(LayerNorm, self).__init__()
        # assert
        assert data_format in ['channels_last', 'channels_first']
        # set attributes
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
        # build parameters
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.weight = Parameter(ops.ones(normalized_shape))
        self.bias = Parameter(ops.zeros(normalized_shape))
        self.layer_norm = nn.LayerNorm(normalized_shape=self.normalized_shape, gamma_init=self.weight, beta_init=self.bias, epsilon=self.eps)
    '''forward'''
    def construct(self, x):
        if self.data_format == 'channels_last':
            # return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return self.layer_norm(x)
        else:
            u = x.mean(1, keep_dims=True)
            s = (x - u).pow(2).mean(1, keep_dimss=True)
            # x = (x - u) / torch.sqrt(s + self.eps)
            x = (x - u) / ops.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# '''BuildNormalization'''
# BuildNormalization = NormalizationBuilder(
#     # requires_renew_modules={'LayerNorm': LayerNorm}
#     requires_renew_modules={'layernorm': LayerNorm}
# ).build



'''ConvNeXtV2Block'''
class ConvNeXtV2Block(nn.Cell):
    def __init__(self, dim, drop_path=0., norm_cfg=None, act_cfg=None):
        super(ConvNeXtV2Block, self).__init__()
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # self.norm = BuildNormalization(placeholder=dim, norm_cfg=norm_cfg)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        # self.act = BuildActivation(act_cfg)
        # self.grn = GRN(4 * dim)
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, group=dim, pad_mode='pad')
        self.norm = BuildNormalization(constructnormcfg(placeholder=dim, norm_cfg=norm_cfg))
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = BuildActivation(act_cfg)
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    '''forward'''
    def construct(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = identity + self.drop_path(x)
        return x


'''ConvNeXtV2'''
class ConvNeXtV2(nn.Cell):
    arch_settings = {
        'atto': {'depths': [2, 2, 6, 2], 'dims': [40, 80, 160, 320]},
        'femto': {'depths': [2, 2, 6, 2], 'dims': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'dims': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        'huge': {'depths': [3, 3, 27, 3], 'dims': [352, 704, 1408, 2816]},
    }
    def __init__(self, structure_type, in_channels=3, arch='tiny', drop_path_rate=0., out_indices=(0, 1, 2, 3), norm_cfg={'type': 'layernorm', 'epsilon': 1e-6}, 
                 act_cfg={'type': 'gelu'}, pretrained=True, pretrained_model_path=''):
        super(ConvNeXtV2, self).__init__()
        assert arch in self.arch_settings
        arch = self.arch_settings[arch]
        # set attributes
        self.structure_type = structure_type
        self.in_channels = in_channels
        self.depths = arch['depths']
        self.dims = arch['dims']
        self.drop_path_rate = drop_path_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrained = pretrained
        self.pretrained_model_path = pretrained_model_path
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'invalid out_indices {index}'
        self.out_indices = out_indices
        # assert
        if structure_type in AUTO_ASSERT_STRUCTURE_TYPES:
            for key, value in AUTO_ASSERT_STRUCTURE_TYPES[structure_type].items():
                assert hasattr(self, key) and (getattr(self, key) == value)
        # build downsample_layers
        # self.downsample_layers = nn.CellList()
        self.downsample_layers = nn.CellList()
        norm_cfg['data_format'] = 'channels_first'
        # stem = nn.Sequential(
        #     nn.Conv2d(in_channels, self.dims[0], kernel_size=4, stride=4),
        #     BuildNormalization(placeholder=self.dims[0], norm_cfg=norm_cfg),
        # )
        stem = nn.SequentialCell(
            nn.Conv2d(in_channels, self.dims[0], kernel_size=4, stride=4, pad_mode='pad'),
            BuildNormalization(constructnormcfg(placeholder=self.dims[0], norm_cfg=norm_cfg)),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            # downsample_layer = nn.Sequential(
            #     BuildNormalization(placeholder=self.dims[i], norm_cfg=norm_cfg),
            #     nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            # )
            downsample_layer = nn.SequentialCell(
                BuildNormalization(constructnormcfg(placeholder=self.dims[i], norm_cfg=norm_cfg)),
                nn.Conv2d(self.dims[i], self.dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        # build stages
        # dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # self.stages = nn.CellList()
        dp_rates = [x.value() for x in ops.linspace(Tensor(0.), Tensor(drop_path_rate), sum(self.depths))]
        self.stages = nn.CellList()
        cur = 0
        for i in range(4):
            norm_cfg['data_format'] = 'channels_last'
            # stage = nn.Sequential(
            #     *[ConvNeXtV2Block(dim=self.dims[i], drop_path=dp_rates[cur + j], norm_cfg=norm_cfg, act_cfg=act_cfg) for j in range(self.depths[i])]
            # )
            stage = nn.SequentialCell(
                *[ConvNeXtV2Block(dim=self.dims[i], drop_path=dp_rates[cur + j], norm_cfg=norm_cfg, act_cfg=act_cfg) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]
            if i in self.out_indices:
                norm_cfg['data_format'] = 'channels_first'
                norm_layer = BuildNormalization(constructnormcfg(placeholder=self.dims[i], norm_cfg=norm_cfg))
                # self.add_module(f'norm{i}', norm_layer)
                self.insert_child_to_cell(f'norm{i}', norm_layer)
        # # load pretrained weights
        # if self.pretrained:
        #     # self.initweights(structure_type, pretrained_model_path)
        #     param_dict = mindspore.load_checkpoint(pretrained_model_path)
        #     mindspore.load_param_into_net(model, param_dict)
    '''forward'''
    def construct(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # outs.append(norm_layer(x).contiguous())
                outs.append(norm_layer(x))
        return tuple(outs)
    # '''initweights'''
    # def initweights(self, structure_type, pretrained_model_path=''):
    #     if pretrained_model_path:
    #         checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    #     else:
    #         checkpoint = model_zoo.load_url(DEFAULT_MODEL_URLS[structure_type], map_location='cpu')
    #     if 'state_dict' in checkpoint:
    #         state_dict = checkpoint['state_dict']
    #     elif 'model' in checkpoint:
    #         state_dict = checkpoint['model']
    #     else:
    #         state_dict = checkpoint
    #     state_dict_convert = {}
    #     for key, value in state_dict.items():
    #         state_dict_convert[key.replace('backbone.', '')] = value
    #         if 'grn.gamma' in key:
    #             state_dict_convert[key] = value.reshape(1, 1, 1, -1)
    #         if 'grn.beta' in key:
    #             state_dict_convert[key] = value.reshape(1, 1, 1, -1)
    #     self.load_state_dict(state_dict_convert, strict=False)

def BuildConvNeXtv2(convnextv2_cfg):
    # assert whether support
    convnextv2_type = convnextv2_cfg.pop('type')
    # parse cfg
    default_cfg = {
        'arch': 'tiny', 
        'in_channels': 3, 
        'norm_cfg': {'type': 'layernorm', 'epsilon': 1e-6},
        'act_cfg': {'type': 'gelu'},
        'drop_path_rate': 0., 
        'out_indices': (0, 1, 2, 3), 
        'pretrained': False,
        'pretrained_model_path': '',
        'structure_type': convnextv2_type, 
    }
    for key, value in convnextv2_cfg.items():
        if key in default_cfg: 
            default_cfg.update({key: value})
    # obtain convnext_cfg
    convnextv2_cfg = default_cfg.copy()
    pretrained = convnextv2_cfg.pop('pretrained')
    pretrained_model_path = convnextv2_cfg.pop('pretrained_model_path')
    # obtain the instanced convnext
    model = ConvNeXtV2(**convnextv2_cfg)
    # load weights of pretrained model
    # if pretrained:
    #     model.initweights(convnext_type, pretrained_model_path)
    if pretrained and os.path.exists(pretrained_model_path):
        param_dict = mindspore.load_checkpoint(pretrained_model_path)
        mindspore.load_param_into_net(model, param_dict)
    # return the model
    return model