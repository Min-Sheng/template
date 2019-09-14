import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.model.nets.base_net import BaseNet
from src.model.nets.resnet50 import ResNet50

class _ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class _Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            _ConvBlock(in_channels, out_channels),
            _ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class _UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = _ConvBlock(in_channels, out_channels)
        self.conv_block_2 = _ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        h_diff = down_x.size(2) - x.size(2)
        w_diff = down_x.size(3) - x.size(3)
        x = F.pad(x, (w_diff // 2, w_diff - w_diff // 2,
                      h_diff // 2, h_diff - h_diff // 2))
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class ResNet50UNet(BaseNet):
    depth = 6

    def __init__(self, in_channels, out_channels, label_type='3cls_label'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.resnet = ResNet50(in_channels, pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(self.resnet.children()))[:3]
        self.input_pool = list(self.resnet.children())[3]
        for bottleneck in list(self.resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = _Bridge(2048, 2048)
        up_blocks.append(_UpBlock(2048, 1024))
        up_blocks.append(_UpBlock(1024, 512))
        up_blocks.append(_UpBlock(512, 256))
        up_blocks.append(_UpBlock(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(_UpBlock(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)
        
        self.backbone = self.down_blocks
        self.newly_added = nn.ModuleList([self.bridge, self.up_blocks, self.out])
        self.label_type = label_type
        
    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResNet50UNet.depth - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{ResNet50UNet.depth - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        
        if self.label_type=='instance':
            x = torch.softmax(x, dim=1)
        elif self.label_type=='3cls_label':
            x = torch.softmax(x, dim=1)
        elif self.label_type=='watershed_label':
            x_1 = torch.tanh(x[:,0:2,:,:])
            print(x_1.max(), x_1.min())
            x_2 = torch.softmax(x[:,2:5,:,:], dim=1)
            x = torch.cat([x_1, x_2], 1)
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
            
    #def trainable_parameters(self):
    #    return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
