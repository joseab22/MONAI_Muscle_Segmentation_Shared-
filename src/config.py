"""
Central configuration: paths, label scheme, and training hyperparameters.

Everything marked TODO depends on what scripts/inspect_data.py reports about
your real data. Don't guess these — fill them in once you've actually looked
at a few cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LabelScheme:
    """Maps integer mask values to muscle names.

    CONFIRM before training: this assumes label order 1=adductor,
    2=quadriceps, 3=hamstring. inspect_data.py's NRRD inspector prints any
    Segment*_Name / Segment*_LabelValue fields from the NRRD header, which
    is the authoritative source if these masks came out of 3D Slicer -
    check that output against the mapping below rather than trusting label
    order alone, since segment order in Slicer doesn't always mean the
    label values are 1, 2, 3 in that same order.
    """

    id_to_name: dict[int, str] = field(
        default_factory=lambda: {0: "background", 1: "adductor", 2: "quadriceps", 3: "hamstring"}
    )

    @property
    def num_classes(self) -> int:
        return len(self.id_to_name)


@dataclass
class Config:
    # --- data ---
    images_dir: Path = IMAGES_DIR
    labels_dir: Path = LABELS_DIR
    label_scheme: LabelScheme = field(default_factory=LabelScheme)

    # Verified against P03-P06 via scripts/inspect_data.py: native spacing
    # ranges 0.139-0.164mm per axis (these are mouse scans, ~20mm field of
    # view - not human). This resamples everything to a common 0.164mm
    # isotropic grid, which is
    # close to native for all 4 cases (minimal interpolation) and keeps
    # volumes around ~125-140 voxels/axis, comfortably bigger than
    # patch_size below.
    #
    # DO NOT set this to something like (1.0, 1.0, 1.0) assuming "typical"
    # clinical MRI spacing - at 1mm this data resamples down to roughly
    # 20x20x19 voxels, smaller than patch_size in every dimension, which
    # crashes RandCropByPosNegLabeld on the first training step. Re-check
    # this value with inspect_data.py if cases outside P03-P06 turn out to
    # have substantially different native resolution.
    target_spacing: tuple[float, float, float] = (0.164, 0.164, 0.164)

    # Patch size used for RandCropByPosNegLabeld during training and as the
    # sliding-window size during validation/inference. Must be divisible by
    # 2^(number of downsampling levels in the UNet, see model.py).
    patch_size: tuple[int, int, int] = (96, 96, 96)

    # --- splits ---
    val_fraction: float = 0.2
    seed: int = 42

    # --- training ---
    batch_size: int = 2
    num_workers: int = 4
    max_epochs: int = 200
    learning_rate: float = 1e-4
    val_interval: int = 5  # run validation every N epochs
    early_stop_patience: int = 20  # stop if val dice hasn't improved in this many val checks

    # --- output ---
    output_dir: Path = OUTPUT_DIR

    device: torch.device = field(default_factory=pick_device)


CONFIG = Config()
