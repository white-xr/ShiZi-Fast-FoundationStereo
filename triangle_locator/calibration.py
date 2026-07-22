from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


class CalibrationError(ValueError):
  invalid_reason = 'RECTIFICATION_INVALID'


@dataclass(frozen=True)
class StereoCalibration:
  image_width: int
  image_height: int
  K1: np.ndarray
  D1: np.ndarray
  K2: np.ndarray
  D2: np.ndarray
  R: np.ndarray
  T: np.ndarray
  baseline_m: float

  @property
  def image_size(self):
    return self.image_width, self.image_height


def _matrix(data, key, shape):
  value = data.get(key)
  if value is None:
    raise CalibrationError(f'stereo calibration field {key} is not configured')
  array = np.asarray(value, dtype=np.float64)
  if array.size != int(np.prod(shape)):
    raise CalibrationError(f'stereo calibration field {key} must contain {int(np.prod(shape))} values')
  return array.reshape(shape)


def load_stereo_calibration(path):
  path = Path(path)
  if not path.exists():
    raise CalibrationError(f'stereo calibration file not found: {path}')
  with path.open('r', encoding='utf-8') as stream:
    data = yaml.safe_load(stream) or {}
  if not bool(data.get('configured', True)):
    raise CalibrationError(f'stereo calibration is not configured: {path}')

  width = int(data.get('image_width') or 0)
  height = int(data.get('image_height') or 0)
  if width <= 0 or height <= 0:
    raise CalibrationError('stereo calibration image_width/image_height must be positive')
  K1 = _matrix(data, 'K1', (3, 3))
  K2 = _matrix(data, 'K2', (3, 3))
  D1 = np.asarray(data.get('D1'), dtype=np.float64).reshape(-1) if data.get('D1') is not None else None
  D2 = np.asarray(data.get('D2'), dtype=np.float64).reshape(-1) if data.get('D2') is not None else None
  if D1 is None or D1.size < 4 or D2 is None or D2.size < 4:
    raise CalibrationError('stereo calibration D1/D2 must each contain at least 4 values')
  R = _matrix(data, 'R', (3, 3))
  T = _matrix(data, 'T', (3, 1))
  measured_baseline = float(np.linalg.norm(T))
  baseline = data.get('baseline_m')
  if baseline is None:
    raise CalibrationError('stereo calibration baseline_m is not configured')
  baseline = float(baseline)
  if baseline <= 0 or measured_baseline <= 0:
    raise CalibrationError('stereo calibration baseline and T must be non-zero and expressed in meters')
  tolerance = max(0.0005, baseline * 0.02)
  if abs(measured_baseline - baseline) > tolerance:
    raise CalibrationError(
      f'baseline_m={baseline:.9f} disagrees with norm(T)={measured_baseline:.9f}; '
      'both must use meters'
    )
  return StereoCalibration(width, height, K1, D1, K2, D2, R, T, baseline)


def adjust_intrinsics_for_roi(K, roi_xy, scale=1.0):
  adjusted = np.asarray(K, dtype=np.float64).copy()
  adjusted[0, :] *= float(scale)
  adjusted[1, :] *= float(scale)
  adjusted[2, :] = np.asarray(K, dtype=np.float64)[2, :]
  adjusted[0, 2] -= float(roi_xy[0]) * float(scale)
  adjusted[1, 2] -= float(roi_xy[1]) * float(scale)
  return adjusted


class StereoRectifier:
  def __init__(self, calibration, alpha=0.0, map_type=cv2.CV_32FC1):
    self.calibration = calibration
    size = calibration.image_size
    self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
      calibration.K1,
      calibration.D1,
      calibration.K2,
      calibration.D2,
      size,
      calibration.R,
      calibration.T,
      flags=cv2.CALIB_ZERO_DISPARITY,
      alpha=float(alpha),
    )
    self.left_map = cv2.initUndistortRectifyMap(
      calibration.K1, calibration.D1, self.R1, self.P1, size, map_type,
    )
    self.right_map = cv2.initUndistortRectifyMap(
      calibration.K2, calibration.D2, self.R2, self.P2, size, map_type,
    )
    self.K = self.P1[:, :3].astype(np.float64)
    rectified_baseline = abs(float(self.P2[0, 3] / self.P2[0, 0]))
    if abs(rectified_baseline - calibration.baseline_m) > max(0.0005, calibration.baseline_m * 0.02):
      raise ValueError('rectified projection matrix baseline disagrees with stereo calibration')
    self.baseline_m = rectified_baseline

  @classmethod
  def from_file(cls, path, alpha=0.0):
    return cls(load_stereo_calibration(path), alpha=alpha)

  def rectify(self, left, right):
    expected = (self.calibration.image_height, self.calibration.image_width)
    if left.shape[:2] != expected or right.shape[:2] != expected:
      raise ValueError(
        f'input image size must be {expected[1]}x{expected[0]}, '
        f'got left={left.shape[1]}x{left.shape[0]} right={right.shape[1]}x{right.shape[0]}'
      )
    left_rect = cv2.remap(left, self.left_map[0], self.left_map[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, self.right_map[0], self.right_map[1], cv2.INTER_LINEAR)
    return left_rect, right_rect


def measure_vertical_epipolar_error(left, right, config=None):
  config = config or {}
  max_features = int(config.get('max_features', 1600))
  min_matches = int(config.get('min_matches', 20))
  ratio = float(config.get('ratio_test', 0.75))
  gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
  gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
  detector = cv2.ORB_create(nfeatures=max_features)
  kp_left, desc_left = detector.detectAndCompute(gray_left, None)
  kp_right, desc_right = detector.detectAndCompute(gray_right, None)
  if desc_left is None or desc_right is None:
    return {'valid': False, 'matches': 0, 'median_px': None, 'p95_px': None}
  matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_left, desc_right, k=2)
  errors = []
  for pair in matches:
    if len(pair) != 2 or pair[0].distance >= ratio * pair[1].distance:
      continue
    left_pt = kp_left[pair[0].queryIdx].pt
    right_pt = kp_right[pair[0].trainIdx].pt
    if left_pt[0] < right_pt[0]:
      continue
    errors.append(abs(left_pt[1] - right_pt[1]))
  if len(errors) < min_matches:
    return {'valid': False, 'matches': len(errors), 'median_px': None, 'p95_px': None}
  errors = np.asarray(errors, dtype=np.float64)
  median = float(np.median(errors))
  p95 = float(np.percentile(errors, 95))
  valid = (
    median <= float(config.get('max_median_px', 0.5))
    and p95 <= float(config.get('max_p95_px', 1.5))
  )
  return {'valid': valid, 'matches': len(errors), 'median_px': median, 'p95_px': p95}
