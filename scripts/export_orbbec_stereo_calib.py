import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

repo_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_dir))

from scripts.run_camera import load_orbbec_sdk, pick_profile


def parse_args():
  parser = argparse.ArgumentParser(description='Export Gemini dual-color calibration from Orbbec SDK.')
  parser.add_argument('--output', default='configs/stereo_calib.yaml')
  parser.add_argument('--width', type=int, default=1280)
  parser.add_argument('--height', type=int, default=800)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--preset', default='Dual Color Streams')
  return parser.parse_args()


def intrinsic_matrix(value):
  return [
    [float(value.fx), 0.0, float(value.cx)],
    [0.0, float(value.fy), float(value.cy)],
    [0.0, 0.0, 1.0],
  ]


def distortion_vector(value):
  return [
    float(value.k1), float(value.k2), float(value.p1), float(value.p2),
    float(value.k3), float(value.k4), float(value.k5), float(value.k6),
  ]


def main():
  args = parse_args()
  ob = load_orbbec_sdk()
  pipeline = ob.Pipeline()
  device = pipeline.get_device()
  device.load_preset(args.preset)
  left = pick_profile(
    pipeline, ob.OBSensorType.LEFT_COLOR_SENSOR, args.width, args.height, args.fps,
    ['BGR', 'RGB', 'YUYV', 'MJPG'], ob,
  ).as_video_stream_profile()
  right = pick_profile(
    pipeline, ob.OBSensorType.RIGHT_COLOR_SENSOR, args.width, args.height, args.fps,
    ['BGR', 'RGB', 'YUYV', 'MJPG'], ob,
  ).as_video_stream_profile()
  left_intrinsic = left.get_intrinsic()
  right_intrinsic = right.get_intrinsic()
  extrinsic = left.get_extrinsic_to(right)
  translation = np.asarray(extrinsic.transform, dtype=np.float64).reshape(3)
  sdk_translation_unit = 'm'
  if np.linalg.norm(translation) > 1.0:
    translation /= 1000.0
    sdk_translation_unit = 'mm_converted_to_m'
  baseline = float(np.linalg.norm(translation))
  if not 0.001 <= baseline <= 1.0:
    raise RuntimeError(f'Orbbec SDK returned an implausible stereo baseline: {baseline} m')
  payload = {
    'configured': True,
    'source': 'Orbbec SDK dual-color stream profiles',
    'sdk_translation_unit': sdk_translation_unit,
    'image_width': int(left_intrinsic.width),
    'image_height': int(left_intrinsic.height),
    'K1': intrinsic_matrix(left_intrinsic),
    'D1': distortion_vector(left.get_distortion()),
    'K2': intrinsic_matrix(right_intrinsic),
    'D2': distortion_vector(right.get_distortion()),
    'R': np.asarray(extrinsic.rot, dtype=np.float64).reshape(3, 3).tolist(),
    'T': translation.reshape(3, 1).tolist(),
    'baseline_m': baseline,
  }
  output = Path(args.output)
  if not output.is_absolute():
    output = repo_dir / output
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
  print(f'Wrote {output} with baseline_m={baseline:.9f}')


if __name__ == '__main__':
  main()
