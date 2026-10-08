"""
Format detection, case pairing, and conversion helpers.

The parts of this module that are actually exercised depend entirely on
what scripts/inspect_data.py reports for your real data:

  - NIfTI image + NIfTI label mask   -> MONAI's LoadImaged handles both
                                         natively, nothing custom needed.
  - DICOM series image               -> MONAI's LoadImaged can read a
                                         directory of .dcm slices directly
                                         as a volume, nothing custom needed.
  - PNG/JPG slice-stack image        -> use load_png_stack_as_volume /
                                         PngStackLoaderd below; MONAI has no
                                         built-in "stack these 2D slices
                                         into a 3D volume along Z" reader.
  - DICOM RTSTRUCT label             -> use convert_rtstruct_to_mask below,
                                         which needs the `rt-utils` package
                                         and the referenced DICOM image
                                         series (RTSTRUCT contours are only
                                         meaningful relative to the image
                                         grid they were drawn on).
  - NumPy label mask                 -> MONAI's LoadImaged handles .npy/.npz
                                         natively, nothing custom needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NIFTI_SUFFIXES = (".nii", ".nii.gz")
IMAGE_SLICE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMPY_EXTS = {".npy", ".npz"}
# Double-barrel extensions that need stripping in one go, or Path.stem
# would leave a trailing ".nii" / ".seg" stuck to the case id.
# ".seg.nrrd" is 3D Slicer's default extension when saving a Segmentation
# node directly (as opposed to exporting a plain labelmap volume).
MULTI_PART_SUFFIXES = (".nii.gz", ".seg.nrrd")


def _case_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suf in MULTI_PART_SUFFIXES:
        if lower.endswith(suf):
            return name[: -len(suf)]
    for suf in NIFTI_SUFFIXES:
        if lower.endswith(suf):
            return name[: -len(suf)]
    return path.stem if path.is_file() else path.name


def _list_cases(root: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    if not root.exists():
        return cases
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        cases[_case_stem(entry)] = entry
    return cases


def discover_case_pairs(images_dir: Path, labels_dir: Path) -> list[dict]:
    """Pair up image/label entries by case id.

    Returns a list of {"case_id": str, "image": Path, "label": Path} dicts,
    one per case that has BOTH an image and a label. Cases missing either
    side are skipped with a printed warning rather than silently dropped,
    since a silent mismatch here means a model trained on the wrong pairing.
    """
    image_cases = _list_cases(images_dir)
    label_cases = _list_cases(labels_dir)

    pairs = []
    for case_id in sorted(set(image_cases) | set(label_cases)):
        image_path = image_cases.get(case_id)
        label_path = label_cases.get(case_id)
        if image_path is None:
            print(f"WARNING: case '{case_id}' has a label but no image, skipping")
            continue
        if label_path is None:
            print(f"WARNING: case '{case_id}' has an image but no label, skipping")
            continue
        pairs.append({"case_id": case_id, "image": image_path, "label": label_path})
    return pairs


def load_png_stack_as_volume(slice_dir: Path) -> np.ndarray:
    """Stack a directory of 2D image slices into a 3D volume.

    Slices are sorted by filename, so filenames must sort in the correct
    anatomical order (e.g. slice_001.png, slice_002.png, ...). If your
    slices are NOT numbered this way, fix the naming before training -
    getting slice order wrong silently produces a scrambled volume that
    will still "train" without erroring.

    Returns an array of shape (Z, H, W).
    """
    from PIL import Image

    slices = sorted(p for p in slice_dir.iterdir() if p.suffix.lower() in IMAGE_SLICE_EXTS)
    if not slices:
        raise FileNotFoundError(f"no image slices found in {slice_dir}")

    arrays = []
    for p in slices:
        with Image.open(p) as im:
            arrays.append(np.array(im))
    volume = np.stack(arrays, axis=0)
    return volume


class PngStackLoaderd:
    """Minimal MONAI-style dictionary transform: loads a PNG/JPG slice
    stack directory into a (1, Z, H, W) array under the given key(s).

    Use this in place of monai.transforms.LoadImaged for any key whose
    data is a directory of 2D slices rather than a volumetric file.
    """

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            volume = load_png_stack_as_volume(Path(d[key]))
            d[key] = volume[np.newaxis, ...].astype(np.float32)  # add channel dim
        return d


def convert_rtstruct_to_mask(
    dicom_series_dir: Path,
    rtstruct_path: Path,
    roi_names: list[str],
) -> np.ndarray:
    """Rasterize DICOM RTSTRUCT contours into an integer label mask aligned
    with the referenced DICOM image series.

    Requires: pip install rt-utils

    `roi_names` controls both which structures are included and the label
    value assigned to each: roi_names[0] -> label 1, roi_names[1] -> label 2,
    etc. Background is always 0. Order matters and should match
    config.LabelScheme.

    NOTE: this wraps the rt-utils API as of its common usage pattern. If the
    installed version's API differs, adjust the two calls below (building
    the RTStruct object and reading a mask by ROI name) - the rasterization
    math itself (contour -> per-slice boolean mask aligned to the image
    grid) is what rt-utils is doing internally, re-derive it from pydicom
    only if rt-utils doesn't work out.
    """
    try:
        from rt_utils import RTStructBuilder
    except ImportError as exc:
        raise ImportError(
            "rt_utils is required to convert RTSTRUCT contours to masks. "
            "Install it with: pip install rt-utils"
        ) from exc

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(dicom_series_dir),
        rt_struct_path=str(rtstruct_path),
    )

    mask = None
    for label_value, roi_name in enumerate(roi_names, start=1):
        roi_mask = rtstruct.get_roi_mask_by_name(roi_name)  # boolean array, image grid shape
        if mask is None:
            mask = np.zeros(roi_mask.shape, dtype=np.int64)
        mask[roi_mask] = label_value

    if mask is None:
        raise ValueError("roi_names was empty, nothing to rasterize")
    return mask
