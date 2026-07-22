import numpy as np
import pytest
import yaml

from triangle_locator.calibration import CalibrationError, adjust_intrinsics_for_roi, load_stereo_calibration


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
