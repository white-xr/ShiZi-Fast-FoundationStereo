# Simplified Triangle Screw Locator

The production path uses one Gemini 305 in `Dual Color Streams` mode. It uses
the device's left and right RGB sensors; native depth is never enabled or read.

## Realtime

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/realtime_triangle_locator.yaml
```

Startup selects `LEFT_COLOR_SENSOR` and `RIGHT_COLOR_SENSOR` at
1280x800@30 FPS, reads K/D/R/T from those exact SDK profiles, creates the
OpenCV rectification maps, and then starts the latest-frame inference loop.
`configs/stereo_calib.yaml` is only a factory-SDK fallback if profile
calibration cannot be queried. No chessboard calibration UI is part of normal
startup.

Each processed frame runs this path:

```text
synchronized Dual RGB
  -> stereo rectification and epipolar check
  -> target_plate YOLO-Seg on rectified left RGB
  -> shared left/right ROI FFS disparity
  -> robust disparity plane and 3D triangle vertices
  -> fixed plate coordinate frame
  -> optional fixed screw offset
  -> rectified and raw left-RGB screw pixels
```

Realtime results are written to `results.jsonl` and the newest result is also
available as `latest_result.json`. Realtime defaults do not save disparity,
depth, point clouds, PLY files, HTML, or preview images.

## Screw Offset

`configs/screw_offset.yaml` is deliberately unconfigured until a measured
plate-frame offset is supplied:

```yaml
configured: false
offset_m:
  dx: 0.0
  dy: 0.0
  dz: 0.0
```

With this state, plate pose remains available while screw output is invalid
with `SCREW_OFFSET_NOT_CONFIGURED`. Normal inference never asks the operator to
click a screw and never emits a placeholder screw pixel.

## Offline Check

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/offline_triangle_locator.yaml
```

Local-image pose visualization in a Windows browser:

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/triangle_pose_web.py \
  --config configs/offline_triangle_locator.yaml --host 0.0.0.0 --port 7860
```

The viewer maps the recovered A/B/C vertices and plate-frame XYZ axes back to
the original left image for display. A rectified-image tab is retained only
for calibration diagnostics. The viewer also shows the three camera-frame
vertices, origin, rotation matrix, plane quality and epipolar error.

Offline images use the cached calibration file because no live SDK profile is
available. The saved left/right images must come from the same Dual RGB profile
without resize or crop. `RECTIFICATION_INVALID` means the image pair and the
loaded calibration did not pass the configured epipolar check.

The Tk screw-offset GUI remains an engineering utility only. It is not
imported or launched by either production command or by the pose viewer.
