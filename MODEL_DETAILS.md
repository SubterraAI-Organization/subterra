# Model Details (v2 extraction)

This project keeps the **current backend’s segmentation models** and their inference logic, separated from Django.

## Supported models

### 1) U-Net (`--model unet`)

Implementation: `subterra_v2/subterra_model/models/unet.py`

- Input: RGB image (`3` channels), converted to float tensor in `[0, 1]`
- Shape handling: input is resized to the nearest multiple of `16` (same behavior as the current backend UNet)
- Output: `1` channel mask with `sigmoid`, then thresholded at `> 0.5` to produce a binary mask

Checkpoint loading:
- Supports both:
  - Lightning-style checkpoints with `state_dict` (keys like `model.*`)
  - Direct `state_dict` saves

### 2) YOLOv8 segmentation (`--model yolo`)

Implementation: Ultralytics `YOLO()` (checkpoint `.pt`)

- Inference uses `model.predict(...)`
- If multiple instance masks are returned, they are combined using `max(axis=0)` to produce one binary mask
- Default confidence threshold in the CLI is `0.3` (matches current backend behavior)

## Post-processing

After either model produces a binary mask:
- Small regions can be filtered via `--threshold-area` using contour-area thresholding (`subterra_v2/subterra_model/utils/masks.py`)

## Metrics

Metrics are calculated from the final (post-processed) mask:
- `root_count`
- `average_root_diameter`
- `total_root_length`
- `total_root_area`
- `total_root_volume`

Implementation: `subterra_v2/subterra_model/utils/root_analysis.py`

