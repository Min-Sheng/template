import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.aspp import build_aspp
from src.model.nets.decoder import build_decoder
from src.model.nets.backbone import build_backbone

from src.model.nets.base_net import BaseNet


class DeepLabV3Plus(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
    """
    def __init__(self, in_channels, out_channels, backbone_name='resnet50', output_stride=16, label_type='3cls_label'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone_name = backbone_name
        self.output_stride = output_stride
        self.backbone = build_backbone(self.backbone_name, self.in_channels, self.output_stride)
        self.aspp = build_aspp(self.backbone_name, self.output_stride)
        self.decoder = build_decoder(self.out_channels, self.backbone_name)
        self.label_type = label_type
    def forward(self, input):
        x, low_level_feat = self.backbone(input, with_output_feature_map=True)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.label_type=='instance':
            x = torch.softmax(x, dim=1)
        elif self.label_type=='3cls_label':
            x = torch.softmax(x, dim=1)
        elif self.label_type=='watershed_label':
            x_1 = torch.tanh(x[:,0:2,:,:])
            x_2 = torch.softmax(x[:,2:5,:,:], dim=1)
            x = torch.cat([x_1, x_2], 1)
        return x