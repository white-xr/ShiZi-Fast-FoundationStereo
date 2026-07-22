import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from triangle_locator.screw_locator import calibrate_offset_from_observations


def parse_args():
  parser = argparse.ArgumentParser(description='Calibrate a fixed plate-frame screw offset from multi-view 2D annotations.')
  parser.add_argument('--observations', required=True, help='JSON list containing R, t, K, uv and split per view')
  parser.add_argument('--output', default='configs/screw_offset.yaml')
  parser.add_argument('--max_validation_rmse_px', type=float, default=3.0)
  return parser.parse_args()


def reprojection_errors(observations, offset):
  errors = []
  for item in observations:
    R = np.asarray(item['R'], dtype=np.float64).reshape(3, 3)
    t = np.asarray(item['t'], dtype=np.float64).reshape(3)
    K = np.asarray(item['K'], dtype=np.float64).reshape(3, 3)
    uv = np.asarray(item['uv'], dtype=np.float64).reshape(2)
    point = R @ offset + t
    if point[2] <= 0:
      errors.append(float('inf'))
      continue
    projected = np.array([
      K[0, 0] * point[0] / point[2] + K[0, 2],
      K[1, 1] * point[1] / point[2] + K[1, 2],
    ])
    errors.append(float(np.linalg.norm(projected - uv)))
  return np.asarray(errors)


def main():
  args = parse_args()
  observations = json.loads(Path(args.observations).read_text(encoding='utf-8'))
  train = [item for item in observations if str(item.get('split', 'train')).lower() == 'train']
  validation = [item for item in observations if str(item.get('split', '')).lower() in {'val', 'validation'}]
  if len(train) < 3:
    raise RuntimeError('at least three training views are required')
  if not validation:
    raise RuntimeError('at least one independent validation view is required')
  offset, result = calibrate_offset_from_observations(train)
  train_errors = reprojection_errors(train, offset)
  validation_errors = reprojection_errors(validation, offset)
  validation_rmse = float(np.sqrt(np.mean(np.square(validation_errors))))
  if not np.isfinite(validation_rmse) or validation_rmse > args.max_validation_rmse_px:
    raise RuntimeError(
      f'validation RMSE {validation_rmse:.3f}px exceeds {args.max_validation_rmse_px:.3f}px; offset not saved'
    )
  payload = {
    'configured': True,
    'source': 'robust multi-view 2D reprojection calibration',
    'offset_m': offset.tolist(),
    'train_views': len(train),
    'train_inlier_views': int(np.count_nonzero(result.inlier_mask)),
    'train_rmse_px': float(np.sqrt(np.mean(np.square(train_errors)))),
    'validation_views': len(validation),
    'validation_rmse_px': validation_rmse,
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
  print(f'Wrote {output}; validation_rmse_px={validation_rmse:.3f}')


if __name__ == '__main__':
  main()
