from src.model.nets.backbone import resnet

def build_backbone(backbone, in_channels, output_stride):
    if backbone == 'resnet50':
        return resnet.ResNet50(in_channels, output_stride)
    elif backbone == 'resnet101':
        return resnet.ResNet101(in_channels, output_stride)
    else:
        raise NotImplementedError
