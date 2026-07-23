from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from triangle_locator.calibration import (
  CalibrationError,
  StereoCalibration,
  StereoRectifier,
  adjust_intrinsics_for_roi,
  load_stereo_calibration,
  stereo_calibration_from_sdk_profiles,
)


def test_roi_intrinsics_shift_principal_point_only():
  K = np.array([[600.0, 0.0, 640.0], [0.0, 610.0, 400.0], [0.0, 0.0, 1.0]])
  adjusted = adjust_intrinsics_for_roi(K, (123, 45))
  assert adjusted[0, 0] == pytest.approx(600.0)
  assert adjusted[1, 1] == pytest.approx(610.0)
  assert adjusted[0, 2] == pytest.approx(517.0)
  assert adjusted[1, 2] == pytest.approx(355.0)


def test_missing_calibration_is_rejected(tmp_path):
  path = tmp_path / 'stereo_calib.yaml'
  path.write_text(yaml.safe_dump({'configured': False}), encoding='utf-8')
  with pytest.raises(CalibrationError, match='not configured') as error:
    load_stereo_calibration(path)
  assert error.value.invalid_reason == 'RECTIFICATION_INVALID'


def test_baseline_and_translation_units_must_agree(tmp_path):
  payload = {
    'configured': True,
    'image_width': 640,
    'image_height': 480,
    'K1': np.eye(3).tolist(),
    'D1': [0, 0, 0, 0, 0],
    'K2': np.eye(3).tolist(),
    'D2': [0, 0, 0, 0, 0],
    'R': np.eye(3).tolist(),
    'T': [[-18.0], [0.0], [0.0]],
    'baseline_m': 0.018,
  }
  path = tmp_path / 'bad_units.yaml'
  path.write_text(yaml.safe_dump(payload), encoding='utf-8')
  with pytest.raises(ValueError, match='must use meters'):
    load_stereo_calibration(path)


def test_sdk_profiles_preserve_left_to_right_translation_and_opencv_distortion_order():
  class Value:
    pass

  left_intrinsic = Value()
  left_intrinsic.width, left_intrinsic.height = 1280, 800
  left_intrinsic.fx, left_intrinsic.fy = 613.5, 613.3
  left_intrinsic.cx, left_intrinsic.cy = 632.0, 398.1
  right_intrinsic = Value()
  right_intrinsic.width, right_intrinsic.height = 1280, 800
  right_intrinsic.fx, right_intrinsic.fy = 614.5, 614.3
  right_intrinsic.cx, right_intrinsic.cy = 636.1, 397.7
  distortion = Value()
  distortion.k1, distortion.k2, distortion.k3 = -1.2, 0.8, -0.2
  distortion.k4, distortion.k5, distortion.k6 = 0.01, 0.02, 0.03
  distortion.p1, distortion.p2 = -0.0001, 0.0002
  extrinsic = Value()
  extrinsic.rot = np.eye(3, dtype=np.float32).reshape(-1)
  extrinsic.transform = np.array([-18.018206, -0.067, 0.046], dtype=np.float32)

  class Profile:
    def __init__(self, intrinsic):
      self.intrinsic = intrinsic

    def get_intrinsic(self):
      return self.intrinsic

    def get_distortion(self):
      return distortion

    def get_extrinsic_to(self, other):
      assert other is right
      return extrinsic

  left = Profile(left_intrinsic)
  right = Profile(right_intrinsic)
  calibration = stereo_calibration_from_sdk_profiles(left, right)
  assert calibration.T[0, 0] == pytest.approx(-0.018018206, abs=1e-9)
  assert calibration.baseline_m == pytest.approx(np.linalg.norm(calibration.T))
  np.testing.assert_allclose(
    calibration.D1,
    [-1.2, 0.8, -0.0001, 0.0002, -0.2, 0.01, 0.02, 0.03],
  )


def test_sdk_profiles_reject_non_horizontal_dual_rgb_translation():
  intrinsic = SimpleNamespace(
    width=1280, height=800, fx=613.5, fy=613.3, cx=632.0, cy=398.1,
  )
  distortion = SimpleNamespace(
    k1=0.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0,
  )
  extrinsic = SimpleNamespace(
    rot=np.eye(3, dtype=np.float32),
    transform=np.array([-18.018206, -18.018206, -18.018206], dtype=np.float32),
  )

  class Profile:
    def get_intrinsic(self):
      return intrinsic

    def get_distortion(self):
      return distortion

    def get_extrinsic_to(self, _other):
      return extrinsic

  with pytest.raises(CalibrationError, match='predominantly horizontal'):
    stereo_calibration_from_sdk_profiles(Profile(), Profile())


def test_rectified_left_pixels_map_back_to_raw_left_pixels():
  calibration = StereoCalibration(
    640,
    480,
    np.array([[600.0, 0, 320.0], [0, 602.0, 240.0], [0, 0, 1.0]]),
    np.array([-0.1, 0.02, 0.001, -0.002, 0.0]),
    np.array([[601.0, 0, 319.0], [0, 603.0, 241.0], [0, 0, 1.0]]),
    np.array([-0.11, 0.03, -0.001, 0.001, 0.0]),
    np.eye(3),
    np.array([[-0.018], [0.0], [0.0]]),
    0.018,
  )
  rectifier = StereoRectifier(calibration)
  raw = np.array([[[120.0, 100.0]], [[320.0, 240.0]], [[500.0, 350.0]]])
  rectified = cv2.undistortPoints(
    raw,
    calibration.K1,
    calibration.D1,
    R=rectifier.R1,
    P=rectifier.P1,
  ).reshape(-1, 2)
  recovered = rectifier.left_rectified_to_raw(rectified)
  np.testing.assert_allclose(recovered, raw.reshape(-1, 2), atol=1e-5)


def test_disabled_rectification_passes_images_masks_and_pixels_through():
  calibration = StereoCalibration(
    8,
    6,
    np.array([[100.0, 0, 4.0], [0, 101.0, 3.0], [0, 0, 1.0]]),
    np.array([-1.2, 0.8, 0.0, 0.0, -0.2]),
    np.array([[102.0, 0, 4.0], [0, 103.0, 3.0], [0, 0, 1.0]]),
    np.array([-1.1, 0.7, 0.0, 0.0, -0.1]),
    np.eye(3),
    np.array([[-0.018], [0.0], [0.0]]),
    0.018,
  )
  rectifier = StereoRectifier(calibration, enabled=False)
  left = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
  right = np.flip(left, axis=1).copy()
  mask = left[:, :, 0] > 30
  left_out, right_out = rectifier.rectify(left, right)
  np.testing.assert_array_equal(left_out, left)
  np.testing.assert_array_equal(right_out, right)
  np.testing.assert_array_equal(rectifier.rectify_left_mask(mask), mask)
  pixels = np.array([[1.0, 2.0], [5.0, 4.0]])
  np.testing.assert_array_equal(rectifier.left_rectified_to_raw(pixels), pixels)
  np.testing.assert_array_equal(rectifier.K, calibration.K1)
