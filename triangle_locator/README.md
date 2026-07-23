# Simplified Triangle Screw Locator

The production path uses one Gemini 305 in `Dual Color Streams` mode. It uses
the device's left and right RGB sensors; native depth is never enabled or read.

## Version Requirements

Tested machine:

- Ubuntu/Linux x86_64
- NVIDIA RTX 3050 6GB, driver `595.71.05`
- Gemini 305, firmware `1.0.52`, serial `CV2L36000024`
- Orbbec SDK `2.8.6`
- Python `3.12.13`
- CUDA PyTorch `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`
- OpenCV `4.13.0`, NumPy `2.4.4`, SciPy `1.17.1`
- Ultralytics `8.4.78`, Flask `3.1.3`, Open3D `0.19.0`

Minimum setup on another Linux PC:

```bash
git clone https://github.com/white-xr/ShiZi-Fast-FoundationStereo.git
cd ShiZi-Fast-FoundationStereo
git checkout codex/triangle-locator

conda create -n ffs python=3.12 -y
conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Realtime Gemini 305 capture also needs Orbbec SDK 2.8.6 and a C++ compiler:

```bash
sudo apt-get install -y g++
export ORBBEC_SDK_DIR=/path/to/OrbbecSDK_v2.8.6_linux_x86_64
export PYORBBECSDK_LIB=/path/to/pyorbbecsdk/install_286/lib
```

`ORBBEC_SDK_DIR` must contain `include/` and `lib/libOrbbecSDK.so`.
`PYORBBECSDK_LIB` is only needed when querying live SDK calibration from Python.
This repository includes the currently used FFS and YOLO weights under
`weights/`, so a fresh clone has the model files required by the configs.

## Realtime

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/run_config.py \
  --config configs/realtime_triangle_locator.yaml
```

Startup selects `LEFT_COLOR_SENSOR` and `RIGHT_COLOR_SENSOR` at
1280x800@30 FPS. The current realtime config keeps inference and display in
raw-left image coordinates (`rectification.enabled=false`) because that is the
view used by the operator and by the YOLO labels. `configs/stereo_calib.yaml`
is the factory-SDK calibration used for metric stereo geometry; live SDK
calibration can be queried when the Python SDK is available. No chessboard
calibration UI is part of normal startup.

Each processed frame runs this path:

```text
synchronized Dual RGB
  -> target_plate YOLO-Seg on raw left RGB
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

Realtime camera visualization in a Windows browser:

```bash
/home/depthai/miniconda3/envs/ffs/bin/python scripts/camera_preview_web.py \
  --config configs/realtime_triangle_locator.yaml --host 0.0.0.0 --port 7861
```

The viewer draws YOLO mask, recovered A/B/C vertices, and C pixel/depth on the
original left image. It also shows the three camera-frame vertices, origin,
rotation matrix and plane quality.

Offline images use the cached calibration file because no live SDK profile is
available. The saved left/right images must come from the same Dual RGB profile
without resize or crop.

The Tk screw-offset GUI remains an engineering utility only. It is not
imported or launched by either production command or by the pose viewer.
