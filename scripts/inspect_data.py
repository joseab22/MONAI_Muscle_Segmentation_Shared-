"""
inspect_data.py

Run this against a handful of real cases BEFORE writing/trusting the dataset
loader. Medical imaging data comes in enough different shapes (NIfTI, DICOM
series, PNG slice stacks, RTSTRUCT contour files, pre-rasterized NumPy masks)
that guessing the format ahead of time is a good way to build the wrong
loader. This script reports what's actually on disk, per case, so the next
step (src/data_utils.py, src/dataset.py) can be written to match reality.

Usage:
    python scripts/inspect_data.py --images data/images --labels data/labels

Expected layout (either shape works for a given case):
    data/images/<case_id>/*.dcm        (DICOM series)
    data/images/<case_id>.nii.gz       (NIfTI)
    data/images/<case_id>/*.png        (PNG slice stack)
    data/labels/<case_id>...           (same case_id, any of: NIfTI mask,
                                         DICOM RTSTRUCT, NumPy mask)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy is not installed yet. Run: pip install -r requirements.txt")
    sys.exit(1)

# Optional-at-this-stage imports: we don't fail hard if one is missing,
# because we don't yet know which formats are actually in use. Each
# inspector below checks for its own dependency and reports clearly if
# it's missing instead of crashing the whole script.
try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import nrrd
except ImportError:
    nrrd = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NIFTI_SUFFIXES = (".nii", ".nii.gz")
NUMPY_EXTS = {".npy", ".npz"}
NRRD_EXTS = {".nrrd", ".nhdr"}
# ".seg.nrrd" is 3D Slicer's default extension when saving a Segmentation
# node directly - strip it as one unit or the case id ends up as "P04.seg"
# instead of "P04", breaking pairing against the image's "P04.nrrd".
MULTI_PART_SUFFIXES = (".nii.gz", ".seg.nrrd")


@dataclass
class CaseReport:
    case_id: str
    role: str  # "image" or "label"
    path: Path
    kind: str = "unknown"
    details: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suf) for suf in NIFTI_SUFFIXES)


def _list_case_entries(root: Path) -> dict[str, Path]:
    """Map case_id -> path, where a case is either a single file directly
    under `root`, or a subdirectory under `root` (e.g. a DICOM series or a
    PNG slice stack)."""
    cases: dict[str, Path] = {}
    if not root.exists():
        return cases
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            cases[entry.name] = entry
        elif entry.is_file():
            # strip known multi-part suffixes (.nii.gz, .seg.nrrd) before
            # falling back to a plain single-extension stem
            stem = entry.name
            lower = stem.lower()
            for suf in MULTI_PART_SUFFIXES:
                if lower.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            else:
                stem = entry.stem
            cases.setdefault(stem, entry)
    return cases


def inspect_nifti(case_id: str, path: Path, role: str) -> CaseReport:
    report = CaseReport(case_id=case_id, role=role, path=path, kind="nifti")
    if nib is None:
        report.problems.append("nibabel not installed (pip install -r requirements.txt)")
        return report
    try:
        img = nib.load(str(path))
        data = img.get_fdata()
        report.details["shape"] = data.shape
        report.details["dtype"] = str(data.dtype)
        report.details["voxel_spacing"] = tuple(round(float(z), 4) for z in img.header.get_zooms())
        if role == "label":
            unique_vals = np.unique(data)
            report.details["unique_values"] = unique_vals.tolist()
            report.details["num_classes_guess"] = len(unique_vals)
    except Exception as exc:  # noqa: BLE001 - surfacing to the user, not handling
        report.problems.append(f"failed to read NIfTI: {exc}")
    return report


def inspect_nrrd(case_id: str, path: Path, role: str) -> CaseReport:
    report = CaseReport(case_id=case_id, role=role, path=path, kind="nrrd")
    if nrrd is None:
        report.problems.append("pynrrd not installed (pip install -r requirements.txt)")
        return report
    try:
        data, header = nrrd.read(str(path))
        report.details["shape"] = data.shape
        report.details["dtype"] = str(data.dtype)

        space_directions = header.get("space directions")
        if space_directions is not None:
            try:
                spacing = tuple(
                    round(float(np.linalg.norm([v for v in row if v is not None])), 4)
                    for row in space_directions
                    if row is not None
                )
                report.details["voxel_spacing"] = spacing
            except (TypeError, ValueError):
                report.details["space_directions_raw"] = space_directions

        if role == "label":
            unique_vals = np.unique(data)
            report.details["unique_values"] = unique_vals.tolist()
            report.details["num_classes_guess"] = len(unique_vals)

            # 3D Slicer segmentation exports often embed the integer label
            # value -> segment name mapping directly in custom header
            # fields (Segment0_Name, Segment0_LabelValue, Segment1_Name, ...).
            # This is the authoritative source for which integer means
            # which muscle - prefer it over assuming label order.
            segment_fields = {k: v for k, v in header.items() if k.lower().startswith("segment")}
            if segment_fields:
                report.details["segment_metadata"] = segment_fields
            else:
                report.details["note"] = (
                    "no Segment*_Name/LabelValue fields found in header - "
                    "confirm the label-value-to-muscle mapping some other way "
                    "(e.g. the tool/script that produced these masks) before "
                    "assuming label order"
                )
    except Exception as exc:  # noqa: BLE001
        report.problems.append(f"failed to read NRRD: {exc}")
    return report


def inspect_numpy(case_id: str, path: Path, role: str) -> CaseReport:
    report = CaseReport(case_id=case_id, role=role, path=path, kind="numpy")
    try:
        arr = np.load(path, allow_pickle=False)
        if hasattr(arr, "files"):  # npz archive
            report.details["archive_keys"] = list(arr.files)
            arr = arr[arr.files[0]]
        report.details["shape"] = arr.shape
        report.details["dtype"] = str(arr.dtype)
        if role == "label":
            unique_vals = np.unique(arr)
            report.details["unique_values"] = unique_vals.tolist()
            report.details["num_classes_guess"] = len(unique_vals)
    except Exception as exc:  # noqa: BLE001
        report.problems.append(f"failed to read NumPy file: {exc}")
    return report


def inspect_png_stack(path: Path, role: str) -> CaseReport:
    report = CaseReport(case_id=path.name, role=role, path=path, kind="png_stack")
    slices = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    report.details["num_slices"] = len(slices)
    if not slices:
        report.problems.append("directory contains no recognizable image slices")
        return report
    if Image is None:
        report.problems.append("Pillow not installed (pip install -r requirements.txt)")
        return report
    try:
        with Image.open(slices[0]) as im:
            report.details["slice_size"] = im.size
            report.details["slice_mode"] = im.mode
        if role == "label":
            # Labels as PNG are usually single-channel with small integer
            # values burned into pixel intensity. Sample a few slices.
            uniques: set[int] = set()
            for s in slices[: min(5, len(slices))]:
                with Image.open(s) as im:
                    uniques.update(np.array(im).flatten().tolist())
            report.details["unique_values_sampled"] = sorted(uniques)
    except Exception as exc:  # noqa: BLE001
        report.problems.append(f"failed to read PNG slice: {exc}")
    return report


def inspect_dicom_dir(path: Path, role: str) -> CaseReport:
    report = CaseReport(case_id=path.name, role=role, path=path, kind="dicom_series")
    dcm_files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".dcm" or p.suffix == "")
    report.details["num_files"] = len(dcm_files)
    if not dcm_files:
        report.problems.append("directory contains no .dcm files")
        return report
    if pydicom is None:
        report.problems.append("pydicom not installed (pip install -r requirements.txt)")
        return report
    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        modality = getattr(ds, "Modality", "unknown")
        report.details["modality"] = modality
        if modality == "RTSTRUCT":
            report.kind = "dicom_rtstruct"
            ds_full = pydicom.dcmread(str(dcm_files[0]))
            roi_names = [roi.ROIName for roi in getattr(ds_full, "StructureSetROISequence", [])]
            report.details["roi_names"] = roi_names
            report.details["num_rois"] = len(roi_names)
        else:
            report.details["rows"] = getattr(ds, "Rows", None)
            report.details["columns"] = getattr(ds, "Columns", None)
            report.details["pixel_spacing"] = getattr(ds, "PixelSpacing", None)
            report.details["slice_thickness"] = getattr(ds, "SliceThickness", None)
    except Exception as exc:  # noqa: BLE001
        report.problems.append(f"failed to read DICOM metadata: {exc}")
    return report


def inspect_case(case_id: str, path: Path, role: str) -> CaseReport:
    if path.is_file():
        if _is_nifti(path):
            return inspect_nifti(case_id, path, role)
        if path.suffix.lower() in NRRD_EXTS:
            return inspect_nrrd(case_id, path, role)
        if path.suffix.lower() in NUMPY_EXTS:
            return inspect_numpy(case_id, path, role)
        return CaseReport(case_id=case_id, role=role, path=path, kind="unknown_file",
                           problems=[f"unrecognized file extension: {path.suffix}"])

    if path.is_dir():
        contents = list(path.iterdir())
        has_dicom = any(p.suffix.lower() == ".dcm" for p in contents)
        has_images = any(p.suffix.lower() in IMAGE_EXTS for p in contents)
        if has_dicom:
            return inspect_dicom_dir(path, role)
        if has_images:
            return inspect_png_stack(path, role)
        return CaseReport(case_id=case_id, role=role, path=path, kind="unknown_dir",
                           problems=["directory has no .dcm or image-slice files"])

    return CaseReport(case_id=case_id, role=role, path=path, kind="missing",
                       problems=["path does not exist"])


def print_report(report: CaseReport) -> None:
    header = f"[{report.role}] {report.case_id}  ->  kind={report.kind}  path={report.path}"
    print(header)
    for key, val in report.details.items():
        print(f"    {key}: {val}")
    for problem in report.problems:
        print(f"    !! {problem}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels"))
    args = parser.parse_args()

    image_cases = _list_case_entries(args.images)
    label_cases = _list_case_entries(args.labels)

    if not image_cases:
        print(f"No cases found under {args.images}. Drop some data in first.")
    if not label_cases:
        print(f"No cases found under {args.labels}. Drop some data in first.")

    all_ids = sorted(set(image_cases) | set(label_cases))
    print(f"Found {len(image_cases)} image case(s), {len(label_cases)} label case(s), "
          f"{len(all_ids)} unique case id(s) total.\n")

    for case_id in all_ids:
        if case_id in image_cases:
            print_report(inspect_case(case_id, image_cases[case_id], "image"))
        else:
            print(f"[image] {case_id}  -> MISSING (no matching image for this label)")

        if case_id in label_cases:
            print_report(inspect_case(case_id, label_cases[case_id], "label"))
        else:
            print(f"[label] {case_id}  -> MISSING (no matching label for this image)")
        print()


if __name__ == "__main__":
    main()
