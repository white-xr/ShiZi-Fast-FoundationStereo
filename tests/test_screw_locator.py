import json

import numpy as np
import pytest

from triangle_locator.screw_locator import calibrate_offset_from_observations, load_screw_offset, locate_screw


def test_known_offset_transform_and_projection():
  angle = np.deg2rad(20)
  R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
  t = np.array([0.02, -0.01, 0.45])
  offset = np.array([0.03, 0.01, 0.005])
  K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
  result = locate_screw(R, t, offset, K, (640, 480))
  expected = R @ offset + t
  assert result['valid']
  np.testing.assert_allclose(result['xyz_camera_m'], expected)
  assert result['u'] == pytest.approx(600 * expected[0] / expected[2] + 320)
  assert result['v'] == pytest.approx(600 * expected[1] / expected[2] + 240)


def test_missing_or_placeholder_offset_is_not_a_valid_screw(tmp_path):
  path = tmp_path / 'offset.json'
  path.write_text(json.dumps({'configured': False, 'offset_m': [0, 0, 0]}), encoding='utf-8')
  assert load_screw_offset(path) is None
  result = locate_screw(np.eye(3), np.array([0, 0, 0.5]), None, np.eye(3), (640, 480))
  assert not result['valid']
  assert result['invalid_reason'] == 'SCREW_OFFSET_NOT_CONFIGURED'


def test_multiview_huber_calibration_rejects_bad_annotation():
  truth = np.array([0.025, -0.012, 0.008])
  K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
  observations = []
  for index, x in enumerate(np.linspace(-0.05, 0.05, 7)):
    R = np.eye(3)
    t = np.array([x, 0.01 * np.sin(index), 0.45 + 0.02 * index])
    point = R @ truth + t
    uv = np.array([K[0, 0] * point[0] / point[2] + K[0, 2], K[1, 1] * point[1] / point[2] + K[1, 2]])
    if index == 3:
      uv += np.array([80.0, -60.0])
    observations.append({'R': R, 't': t, 'K': K, 'uv': uv})
  offset, result = calibrate_offset_from_observations(observations)
  np.testing.assert_allclose(offset, truth, atol=2e-3)
  assert np.count_nonzero(result.inlier_mask) < len(observations)
