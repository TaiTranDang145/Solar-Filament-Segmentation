from __future__ import annotations


def _native_unet(pretrained_backbone: bool):
    import timm
    import torch
    from torch import nn
    from torch.nn import functional

    class ConvBlock(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    class NativeUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(
                "efficientvit_b2.r288_in1k",
                pretrained=pretrained_backbone,
                features_only=True,
                in_chans=1,
            )
            reductions = self.encoder.feature_info.reduction()
            channels = self.encoder.feature_info.channels()
            by_reduction = dict(zip(reductions, channels))
            current = channels[-1]
            blocks = []
            skip_reductions = []
            for reduction, output in zip((16, 8, 4, 2, 1), (192, 96, 48, 24, 12)):
                skip = by_reduction.get(reduction, 0)
                blocks.append(ConvBlock(current + skip, output))
                skip_reductions.append(reduction if skip else None)
                current = output
            self.blocks = nn.ModuleList(blocks)
            self.skip_reductions = skip_reductions
            self.head = nn.Conv2d(current, 1, 1)

        def forward(self, image):
            height, width = image.shape[-2:]
            features = self.encoder(image)
            by_reduction = dict(zip(self.encoder.feature_info.reduction(), features))
            output = features[-1]
            for block, reduction in zip(self.blocks, self.skip_reductions):
                output = functional.interpolate(output, scale_factor=2, mode="nearest")
                if reduction is not None:
                    output = torch.cat([output, by_reduction[reduction]], dim=1)
                output = block(output)
            return {"out": self.head(output)[..., :height, :width]}

    return NativeUNet()


def build_model(
    pretrained_backbone: bool = True,
    model_name: str = "deeplabv3_resnet50",
):
    if model_name == "native_unet":
        return _native_unet(pretrained_backbone)
    if model_name != "deeplabv3_resnet50":
        raise ValueError(f"unknown model_name: {model_name}")
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
