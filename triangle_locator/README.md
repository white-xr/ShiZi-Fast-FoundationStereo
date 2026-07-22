# Triangle Locator

The production path uses synchronized Gemini 305 dual RGB frames only. Native
depth is not consumed.

Before localization, configure both files:

1. Export SDK calibration with
   `python scripts/export_orbbec_stereo_calib.py`, or calibrate a board with
   `python scripts/calibrate_stereo.py ...`.
2. Put a measured/CAD plate-frame screw offset in
   `configs/screw_offset.yaml`, or calibrate it from train/validation views with
   `python scripts/calibrate_screw_offset.py ...`.

An unconfigured stereo calibration stops startup with
`RECTIFICATION_INVALID`. An unconfigured screw offset still permits plate pose
output, but every result has `screw_valid=false` and
`SCREW_OFFSET_NOT_CONFIGURED`.

Offline:

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/offline_triangle_locator.yaml
```

Realtime:

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/realtime_triangle_locator.yaml
```

FFS-only debug without YOLO:

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/ffs_debug_no_yolo.yaml
```
