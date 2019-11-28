import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.nets.backbone import dilated_resnet
from src.model.nets.jpu import JPU
from torch.nn.functional import upsample
from src.model.nets.base_net import BaseNet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class FastFCN(BaseNet):
    """
        Fast FCN.
        Originally, Fast FCN has ResNet101 as a backbone.
        Please refer to:
            H. Wu et al., FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation
            https://arxiv.org/abs/1903.11816
        Args:
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            backbone_name (str): The backbone name.
            output_stride (int): The reduction ratio of the output and original image.
            label_type (str): The type of the label.
    """

    def __init__(self, in_channels, out_channels, backbone_name='resnet101', output_stride=16, dilated=True, norm_layer=nn.BatchNorm2d, label_type='3cls_label'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = FCNHead(2048, self.out_channels, norm_layer)
        if backbone_name == 'resnet50':
            self.pretrained = dilated_resnet.resnet50(pretrained=True, dilated=dilated, norm_layer=norm_layer, root='./pretrain_encoding')
        elif backbone_name == 'resnet101':
            self.pretrained = dilated_resnet.resnet101(pretrained=True, dilated=dilated, norm_layer=norm_layer, root='./pretrain_encoding')
        elif backbone_name == 'resnet152':
            self.pretrained = dilated_resnet.resnet152(pretrained=True, dilated=dilated, norm_layer=norm_layer, root='./pretrain_encoding')
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone_name))
        
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.backbone = backbone_name
        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, up_kwargs=up_kwargs)
        
    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4
        
    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = upsample(x, imsize, **self._up_kwargs)
        
        return x

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
    
# for debug
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = torch.randn(10, 3, 224, 224).to(device)

    model = FastFCN(3, out_channels=10).to(device)
    y = model(img)

    print(y.shape)