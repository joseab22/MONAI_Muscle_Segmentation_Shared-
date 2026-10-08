# MONAI Muscle Segmentation

Training pipeline for segmenting muscles in medical image stacks (MRI/CT) using
[MONAI](https://monai.io/) + PyTorch.

Status: format confirmed — NRRD images, NRRD label masks, 3 muscle classes
(adductor, quadriceps, hamstring) + background. `src/config.py`,
`src/dataset.py`, and `src/model.py` are wired for this. One thing still
worth double-checking with `scripts/inspect_data.py`: the exact integer
value each muscle maps to in your mask files (see the `LabelScheme` note in
`src/config.py`) — don't assume label order without confirming it.

## Directory layout

```
MONAI_Muscle_Segmentation/
├── data/
│   ├── images/      # one subfolder or file per case: the image stack
│   └── labels/      # matching subfolder or file per case: the muscle contours/masks
├── src/
│   ├── config.py        # paths, hyperparameters, device selection
│   ├── data_utils.py    # format detection + conversion (DICOM/NIfTI/PNG stacks, RTSTRUCT -> mask)
│   ├── dataset.py       # MONAI Dataset + transform pipeline
│   ├── model.py         # network definition (3D UNet by default)
│   └── train.py         # training loop
├── scripts/
│   └── inspect_data.py  # run this first — reports what's actually in data/
├── outputs/          # checkpoints + logs land here (not created yet)
└── requirements.txt
```

## Step 0 — environment setup

Run these yourself in a terminal (not through me, so you can watch it and
it isn't fighting my sandbox):

```
cd ~/Desktop/MONAI_Muscle_Segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you're on Apple Silicon, PyTorch will install with MPS (Metal) GPU support
automatically — no CUDA needed. Verify with:

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Step 1 — drop in a few real cases

Put a handful of real cases (not the whole dataset yet) into:

- `data/images/<case_id>/...` — the image stack for that case
- `data/labels/<case_id>/...` — the matching contour/mask file(s)

`<case_id>` should be the same string on both sides so we can pair them up,
e.g. `data/images/case001.nrrd` + `data/labels/case001.nrrd`.

## Producing the data in 3D Slicer

Your raw scans are DICOM series right now. The workflow below converts each
case's DICOM to NRRD and creates the matching 3-muscle label NRRD, both
landing with the same `<case_id>` so the pairing logic in Step 1 lines up.
Repeat this per case.

**A. Load the DICOM series**

1. Open 3D Slicer, switch to the **DICOM** module (toolbar icon, or search
   "DICOM" in the module search box).
2. Click **Import DICOM files/folders** and point it at the case's folder
   of `.dcm` files. It indexes into Slicer's local DICOM database.
3. In the browser, select the patient/study/series and click **Load** —
   this creates a Volume node in the scene.

**B. Export the volume as NRRD**

4. **File → Save Data...**. In the save dialog, find the row for your
   volume node.
5. Set its format to **NRRD (.nrrd)**, set the directory to this project's
   `data/images/`, and set the filename to exactly `<case_id>.nrrd`
   (e.g. `P04.nrrd`) — not whatever Slicer auto-suggests.
6. Click **Save**.

**C. Create the 3-muscle segmentation**

7. Switch to the **Segment Editor** module.
8. At the top: set the segmentation to "Create new Segmentation as..." and
   set the **source/master volume** to the volume you just loaded — this
   locks the segmentation's geometry (spacing, orientation) to match the
   image exactly.
9. Click **Add** three times, then double-click each new segment's name
   and rename them exactly: `adductor`, `quadriceps`, `hamstring`, in that
   order, every time, for every case. Consistent naming/order is what
   keeps label values 1/2/3 meaning the same muscle across all cases.
   (`scripts/inspect_data.py` double-checks this for you either way — see
   step 13.)
10. Actually contouring each muscle — see **Segmenting a muscle, step by
    step** below. Do this once per segment (adductor, then quadriceps,
    then hamstring).
11. Repeat for the other two segments.

### Segmenting a muscle, step by step

You'll work mostly in the **axial** slice view (the one looking down the
leg, showing a cross-section — it's usually the Red view by default), since
that's the orientation where the 3 muscles are most visually distinct from
each other.

**Before you start contouring:**

- Adjust contrast so muscle boundaries are easy to see: in the **Volumes**
  module, use the Window/Level controls (or just right-click-drag on the
  image in a slice view, which is Slicer's shortcut for adjusting
  window/level interactively) until the fascial planes (the thin
  boundaries between adjacent muscles) are clearly visible as distinct
  lines.
- Zoom in (scroll wheel) so you're not trying to trace a tiny thigh
  cross-section at a few pixels wide.
- Scroll through the full slice range first (mouse wheel over the slice
  view) just to see where the adductor/quadriceps/hamstring group are each
  actually present — you don't necessarily need to label every single one
  of the 134-140 slices, only the ones where that muscle is visibly there.

**Contouring one segment:**

1. In the segment list, click the segment you want to work on (e.g.
   `quadriceps`) to make it active — the tool you use next will paint
   into *that* segment specifically.
2. Pick a tool from the Segment Editor's effects list:
   - **Draw** (sometimes labeled "Draw tube"/closed surface) — click
     points around the muscle's boundary on the current slice to form a
     polygon, then press Enter/double-click to close and fill it. This
     gives the cleanest, most precise contour and is the one I'd default
     to for well-defined muscle boundaries.
   - **Paint** — a round brush you drag across the image. Faster for
     rough work, but easy to bleed across a fascial plane into the
     adjacent muscle if the brush is too big or you're not careful.
   - **Threshold** — restricts painting to an intensity range. Muscle
     tissue has a fairly narrow, consistent intensity range on this
     sequence, so this can help you quickly rough in a muscle's extent,
     but it won't by itself separate two adjacent muscles that have
     similar intensity — you still need to manually erase/correct at the
     shared boundary.
3. Contour that muscle on the current slice, then move to another slice
   a handful of frames away (not necessarily the very next one) and
   contour it again there.
4. Once you have the muscle contoured on, say, every 5th-10th slice
   across its visible range, use the **Fill between slices** effect — it
   interpolates a full contour for every slice in between automatically,
   which is a big time-saver across ~100+ slices.
5. Scroll through the *entire* stack for that segment afterward and check
   the interpolated slices actually look right — interpolation can go
   wrong where the muscle's cross-sectional shape changes quickly (e.g.
   near where it tapers to a tendon). Manually fix any slice that looks
   off, then re-run Fill between slices if you added more manual slices.
6. Clean up: the **Smoothing** effect rounds off jagged pixel-level edges
   from manual painting; the **Islands** effect removes small
   disconnected blobs that leaked outside the muscle (keep only the
   largest island if you accidentally painted a stray dot elsewhere).
7. Sanity-check in 3D: toggle on the segment's visibility in the 3D view
   (eyeball icon next to it) and rotate around. A real muscle should look
   like one smooth, continuous, anatomically plausible volume — floating
   disconnected blobs or a shape that looks nothing like a muscle belly
   means something leaked or was missed.

Repeat this whole sequence for `adductor` and `hamstring`, switching the
active segment each time. It's fine (expected, even) for muscles to touch
each other at their shared fascial plane — just don't let one segment's
paint overlap into the other's territory at that boundary.

**D. Export the segmentation as NRRD**

12. **File → Save Data...** again. Find the Segmentation node's row, set
    its format to NRRD, directory to `data/labels/`, and — same rule as
    the image — set the filename to exactly `<case_id>.nrrd`
    (e.g. `P04.nrrd`), not the auto-suggested `<name>.seg.nrrd`. (If you do
    end up with `.seg.nrrd`, the pairing code now strips that correctly
    too, but matching filenames exactly is simpler to reason about.)

**E. Confirm it worked**

13. Run `python scripts/inspect_data.py` (Step 2 below). For the label
    file, check that `unique_values` is `[0, 1, 2, 3]` and look at
    `segment_metadata` for `SegmentN_Name` / `SegmentN_LabelValue` fields —
    that's the authoritative confirmation of which integer is which
    muscle, independent of segment creation order.

Note: you don't have to finish converting every case's images to NRRD
before moving forward — the training pipeline reads DICOM series
directories directly too (same ITK reader), so a mix of NRRD and
not-yet-converted DICOM cases while you catch up on labeling is fine.

## Step 2 — inspect the real data

```
python scripts/inspect_data.py --images data/images --labels data/labels
```

This reports, per case: file type (NIfTI / DICOM series / PNG stack / RTSTRUCT /
NumPy), shape, voxel spacing if available, and the unique label values found.
We use this output to:

- pick the right MONAI readers in `src/data_utils.py`
- decide whether labels need converting first (e.g. RTSTRUCT → raster mask)
- set `NUM_CLASSES` and the label-to-muscle-name mapping in `src/config.py`

Send me the output (or just tell me what it prints) and we'll fill in the
remaining pieces together rather than guessing now.

## Step 3 — train

`src/dataset.py` needs at least 2 fully paired cases (image + label both
present) to form a train/val split — it'll raise a clear error otherwise.
With only P04 labeled so far, this step isn't runnable yet; once you have
the 3 more cases labeled this week (4 total), it becomes technically
runnable, though 4 cases is still very small for a 3D UNet — treat an
early run with that little data as a pipeline smoke test (does it run
end-to-end without crashing, does loss go down at all), not a real result.

Once `src/config.py` is confirmed against real data:

```
python -m src.train
```

Checkpoints and TensorBoard logs go to `outputs/`. View logs with:

```
tensorboard --logdir outputs/
```
