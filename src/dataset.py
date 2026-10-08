"""
MONAI dataset + transform pipeline.

Images and labels are both NRRD volumes (confirmed via scripts/inspect_data.py).
MONAI's LoadImaged reads .nrrd through the ITK backend (the `itk` package in
requirements.txt), so no custom loader is needed - data_utils.PngStackLoaderd
and convert_rtstruct_to_mask exist for other formats but aren't used here.
"""

from __future__ import annotations

import random

from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
)

from .config import Config
from .data_utils import discover_case_pairs

KEYS = ["image", "label"]


def get_transforms(config: Config, train: bool) -> Compose:
    base = [
        LoadImaged(keys=KEYS, reader="ITKReader", ensure_channel_first=False, image_only=True),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes="RAS"),
        Spacingd(
            keys=KEYS,
            pixdim=config.target_spacing,
            mode=("bilinear", "nearest"),  # interpolate image smoothly, labels stay discrete
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),
    ]

    if train:
        base += [
            RandCropByPosNegLabeld(
                keys=KEYS,
                label_key="label",
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=KEYS, prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]

    base.append(EnsureTyped(keys=KEYS))
    return Compose(base)


def build_datasets(config: Config) -> tuple[Dataset, Dataset]:
    pairs = discover_case_pairs(config.images_dir, config.labels_dir)
    if not pairs:
        raise RuntimeError(
            f"No matched image/label pairs found under {config.images_dir} and "
            f"{config.labels_dir}. Run scripts/inspect_data.py first to check pairing."
        )

    data_dicts = [{"image": str(p["image"]), "label": str(p["label"]), "case_id": p["case_id"]} for p in pairs]

    rng = random.Random(config.seed)
    shuffled = data_dicts[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * config.val_fraction)) if len(shuffled) > 1 else 0
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    if not train_files:
        raise RuntimeError(
            f"Only {len(data_dicts)} case(s) found - need at least 2 to form a "
            "train/val split. Add more cases to data/."
        )

    train_ds = CacheDataset(
        data=train_files,
        transform=get_transforms(config, train=True),
        cache_rate=1.0,
        num_workers=config.num_workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_transforms(config, train=False),
        cache_rate=1.0,
        num_workers=config.num_workers,
    )
    return train_ds, val_ds
