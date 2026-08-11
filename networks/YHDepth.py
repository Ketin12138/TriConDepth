# yhdepth.py

import torch
import torch.nn as nn
from networks.layers import PyramidFeatureFusionV2, DctDownsample
from .newcrf_layers import NewCRF
from .depth_update import DepthUpdateModule
from .swin_transformer import SwinTransformer
from .encoder import get_backbone  

class YHDepth(nn.Module):
    def __init__(self, backbone_name='resnet101', pretrained=False, scale: float = 1.0, ape: bool = False,
                 img_size: tuple = None, drop_path_rate: float = 0.2, drop_path_rate_crf: float = 0.,
                 seq_dropout_rate: float = 0., downsample_strategy: str = 'dct', **kwargs):
        super().__init__()

        self.patch_size = 8
        self.scale = scale
        self.img_size = img_size      

        if backbone_name.startswith('resnet'):
            in_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'densenet161':
            in_channels = [96, 384, 768, 2208]
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}')

        embed_dim = 512
        self.hidden_dim = 192

        self.backbone = get_backbone(backbone_name, pretrained=pretrained)
        
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3],
                           num_heads=32, drop_path=drop_path_rate_crf)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2],
                           num_heads=16, drop_path=drop_path_rate_crf)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=self.hidden_dim, window_size=win, v_dim=v_dims[1],
                           num_heads=8, drop_path=drop_path_rate_crf)

        self.update = DepthUpdateModule(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            scale=self.scale,
            seq_drop_rate=seq_dropout_rate
        )

        self.decoder = PyramidFeatureFusionV2([8, 4, 2, 1], in_channels, embed_dim, downsample_strategy)
        self.project_hidden = nn.Conv2d(self.hidden_dim + in_channels[0], self.hidden_dim, 3, padding=1)
        self.project_context = DctDownsample(2, 5, 2, in_channels[0], in_channels[0])
        
    def parameters_1x(self):
        yield from self.backbone.parameters()

    def parameters_5x(self):
        param_1x = set(self.parameters_1x())
        for param in self.parameters():
            if param not in param_1x:
                yield param
                
    def extract_feature(self, imgs: torch.Tensor):
        feats = self.backbone(imgs)
        pff_out = self.decoder(*feats)
        e3 = self.crf3(feats[-1], pff_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[-2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[-3], e2)
        context = self.project_context(feats[0])
    
        return [pff_out, e3, e2, e1], context, feats[0]
    
    def fuse_features(self, decoded_feats, context, feats0):
        e1 = decoded_feats[-1]  
        gru_hidden = torch.tanh(
        self.project_hidden(torch.cat([e1, context], 1))
        )
        return gru_hidden

    def decode_from_feature(self, feat):
        depths = self.update(feat, max_iters=1)
        return depths

    def forward(self, imgs: torch.Tensor, max_iters: int = None, mode: str = 'inference',return_feat: bool = False):
        assert imgs.shape[-2:] == self.img_size, f'Input image size {imgs.shape[-2:]} is not equal to {self.img_size}.'

        feats = self.backbone(imgs)
        pff_out = self.decoder(*feats)

        e3 = self.crf3(feats[-1], pff_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[-2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[-3], e2)

        context = self.project_context(feats[0])
        gru_hidden = torch.tanh(
            self.project_hidden(torch.cat([e1, context], 1))
        )

        depths = self.update(gru_hidden, max_iters=max_iters)

        if self.training:
            if mode == 'train_stage2':
                main_depth = depths[0] if isinstance(depths, (list, tuple)) else depths
                freq_regs = depths[1:] if isinstance(depths, (list, tuple)) else []
                return main_depth, freq_regs, e1
            elif mode == 'train_stage3':
                return depths  
        else:
            return depths[0]
