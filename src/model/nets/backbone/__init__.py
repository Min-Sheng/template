from src.model.nets.backbone import resnet
from src.model.nets.backbone import dilated_resnet

def build_backbone(backbone, in_channels, output_stride, pretrained):
    if backbone == 'resnet50':
        return resnet.ResNet50(in_channels, output_stride, pretrained)
    elif backbone == 'resnet101':
        return resnet.ResNet101(in_channels, output_stride, pretrained)
    else:
        raise NotImplementedError
