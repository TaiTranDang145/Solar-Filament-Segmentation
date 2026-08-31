from __future__ import annotations


def build_model(pretrained_backbone: bool = True):
    try:
        from torch import nn
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50
    except ImportError as exc:
        raise RuntimeError("training requires torch and torchvision") from exc

    backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    model = deeplabv3_resnet50(weights=None, weights_backbone=backbone_weights)
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    model.aux_classifier = None
    return model
