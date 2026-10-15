# MONAI Muscle Segmentation

A deep-learning model, built with [MONAI](https://monai.io/) and PyTorch, that
automatically segments thigh muscles in mouse MRI scans. It outlines three
muscle groups: the **adductors**, **quadriceps**, and **hamstrings**.

## Why

Hand-contouring each muscle slice by slice is slow and is the main bottleneck
in analyzing these scans. The goal of this model is to take over most of that
annotation work, so many more scans can be measured consistently. Researchers
then review and correct its output instead of drawing every contour from
scratch.

## What the annotations are for

With reliable per-muscle segmentations, we can track how mouse muscle responds
after injury over time, including:

- **Loss of muscle mass**: changes in the volume of each muscle group
- **Fat infiltration**: fat building up within and around the muscle
- Other structural changes in muscle quality during recovery

## The bigger picture

The long-term aim is to carry these findings over to people, especially
injured athletes. By understanding how muscle degenerates and recovers in the
mouse model, we hope to help design better, more specialized rehabilitation
plans that lead to full or near-full recovery from complex injuries such as
**ACL tears** and **Achilles tendon ruptures**.

## Repository

- `src/`: dataset pipeline, 3D UNet model, and training loop
- `scripts/inspect_data.py`: checks image and label files before training
- `data/images/`, `data/labels/`: scans and matching masks, paired by case
  ID (not tracked in git)

Train with `python -m src.train` after installing `requirements.txt`.
