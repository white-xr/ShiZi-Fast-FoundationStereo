import json
from pathlib import Path

import numpy as np
import yaml


def load_screw_offset(path):
  if path in {None, '', 'null'}:
    return None
  path = Path(path)
  if not path.exists():
    return None
  with path.open('r', encoding='utf-8') as stream:
    data = json.load(stream) if path.suffix.lower() == '.json' else (yaml.safe_load(stream) or {})
  if not bool(data.get('configured', False)):
    return None
  value = data.get('offset_m')
  if value is None:
    value = [data.get('dx'), data.get('dy'), data.get('dz')]
  if any(item is None for item in value):
    raise ValueError('configured screw offset must provide dx/dy/dz in meters')
  offset = np.asarray(value, dtype=np.float64).reshape(3)
  if not np.isfinite(offset).all() or np.linalg.norm(offset) <= 1e-12:
    raise ValueError('configured screw offset must be finite and cannot be an all-zero placeholder')
  return offset


def locate_screw(R, t, offset_m, K, image_size):
  if offset_m is None:
    return {
      'valid': False,
      'invalid_reason': 'SCREW_OFFSET_NOT_CONFIGURED',
      'xyz_camera_m': None,
      'u': None,
      'v': None,
    }
  point = np.asarray(R, dtype=np.float64) @ np.asarray(offset_m, dtype=np.float64) + np.asarray(t, dtype=np.float64)
  if not np.isfinite(point).all() or point[2] <= 0:
    return {'valid': False, 'invalid_reason': 'PROJECTION_OUT_OF_IMAGE', 'xyz_camera_m': point, 'u': None, 'v': None}
  K = np.asarray(K, dtype=np.float64)
  u = float(K[0, 0] * point[0] / point[2] + K[0, 2])
  v = float(K[1, 1] * point[1] / point[2] + K[1, 2])
  width, height = image_size
  valid = 0 <= u < width and 0 <= v < height
  return {
    'valid': valid,
    'invalid_reason': None if valid else 'PROJECTION_OUT_OF_IMAGE',
    'xyz_camera_m': point,
    'u': u if valid else None,
    'v': v if valid else None,
  }


def calibrate_offset_from_observations(observations, initial=None):
  from scipy.optimize import least_squares

  if len(observations) < 3:
    raise ValueError('at least three views are required to calibrate screw offset')

  def residuals(offset):
    errors = []
    for observation in observations:
      R = np.asarray(observation['R'], dtype=np.float64).reshape(3, 3)
      t = np.asarray(observation['t'], dtype=np.float64).reshape(3)
      K = np.asarray(observation['K'], dtype=np.float64).reshape(3, 3)
      uv = np.asarray(observation['uv'], dtype=np.float64).reshape(2)
      point = R @ offset + t
      if point[2] <= 1e-9:
        errors.extend([1e3, 1e3])
      else:
        projected = np.array([
          K[0, 0] * point[0] / point[2] + K[0, 2],
          K[1, 1] * point[1] / point[2] + K[1, 2],
        ])
        errors.extend((projected - uv).tolist())
    return np.asarray(errors)

  result = least_squares(
    residuals,
    np.zeros(3) if initial is None else np.asarray(initial, dtype=np.float64),
    loss='huber',
    f_scale=2.0,
  )
  view_errors = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
  median = float(np.median(view_errors))
  mad = float(1.4826 * np.median(np.abs(view_errors - median)))
  threshold = median + max(3.0 * mad, 2.0)
  inlier_mask = view_errors <= threshold
  if np.count_nonzero(inlier_mask) >= 3 and not np.all(inlier_mask):
    inliers = [item for item, keep in zip(observations, inlier_mask) if keep]
    result = least_squares(
      lambda value: np.concatenate([
        (lambda point, K, uv: np.array([
          K[0, 0] * point[0] / point[2] + K[0, 2] - uv[0],
          K[1, 1] * point[1] / point[2] + K[1, 2] - uv[1],
        ]))(
          np.asarray(item['R']).reshape(3, 3) @ value + np.asarray(item['t']).reshape(3),
          np.asarray(item['K']).reshape(3, 3),
          np.asarray(item['uv']).reshape(2),
        ) for item in inliers
      ]),
      result.x,
      loss='huber',
      f_scale=2.0,
    )
  result.inlier_mask = inlier_mask
  return result.x, result
