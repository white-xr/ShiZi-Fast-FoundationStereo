import argparse
import sys
from pathlib import Path

import yaml

repo_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_dir))

from scripts.run_camera import load_orbbec_sdk, pick_profile
from triangle_locator.calibration import stereo_calibration_from_sdk_profiles, stereo_calibration_payload


def parse_args():
  parser = argparse.ArgumentParser(description='Export Gemini dual-color calibration from Orbbec SDK.')
  parser.add_argument('--output', default='configs/stereo_calib.yaml')
  parser.add_argument('--width', type=int, default=1280)
  parser.add_argument('--height', type=int, default=800)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--preset', default='Dual Color Streams')
  return parser.parse_args()


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
  calibration = stereo_calibration_from_sdk_profiles(left, right)
  payload = stereo_calibration_payload(calibration, fps=args.fps, preset=args.preset)
  output = Path(args.output)
  if not output.is_absolute():
    output = repo_dir / output
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
  print(f'Wrote {output} with baseline_m={calibration.baseline_m:.9f}')


if __name__ == '__main__':
  main()
