"""
Network definition.

Default: MONAI's 3D UNet. This is a reasonable starting point for muscle
segmentation on MRI/CT volumes - swap for SegResNet or SwinUNETR later if
UNet plateaus, without touching anything else in the pipeline.
"""

from __future__ import annotations

from monai.networks.nets import UNet

from .config import Config


def build_model(config: Config) -> UNet:
    num_classes = config.label_scheme.num_classes
    model = UNet(
        spatial_dims=3,
        in_channels=1,  # single-channel MRI/CT; bump this if you have multiple sequences/modalities stacked
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="BATCH",
    )
    return model.to(config.device)
