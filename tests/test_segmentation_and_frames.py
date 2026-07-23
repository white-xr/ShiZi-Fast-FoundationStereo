from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from scripts.seg_pointcloud import select_target_candidate, target_roi
from triangle_locator import pipeline
from triangle_locator.pipeline import TriangleLocator


def candidate(x0, y0, x1, y1, confidence, shape=(100, 160)):
  mask = np.zeros(shape, dtype=bool)
  mask[y0:y1, x0:x1] = True
  return {'mask': mask, 'confidence': confidence, 'class_name': 'target_plate'}


def test_ambiguous_target_plates_are_not_or_merged():
  candidates = [candidate(10, 10, 50, 50, 0.8), candidate(90, 15, 130, 55, 0.78)]
  selected, reason, filtered = select_target_candidate(candidates, (100, 160), {
    'min_area_px': 100,
    'ambiguity_conf_delta': 0.05,
    'ambiguity_min_area_ratio': 0.5,
  })
  assert selected is None
  assert reason == 'MULTIPLE_PLATES'
  assert len(filtered) == 2


def test_clear_best_target_is_selected_without_union():
  first = candidate(10, 10, 50, 50, 0.92)
  second = candidate(90, 15, 130, 55, 0.6)
  selected, reason, _ = select_target_candidate([first, second], (100, 160), {'min_area_px': 100})
  assert reason is None
  assert selected['mask'].sum() == first['mask'].sum()
  assert not selected['mask'][20, 100]


def test_roi_adds_disparity_context_on_left_and_is_shared():
  mask = candidate(80, 30, 120, 70, 0.9)['mask']
  roi = target_roi(mask, max_disp=32, context_padding_px=8, min_size_px=0)
  assert roi == (40, 22, 128, 78)
  left = np.zeros((100, 160, 3), dtype=np.uint8)
  right = np.ones_like(left)
  x0, y0, x1, y1 = roi
  assert left[y0:y1, x0:x1].shape == right[y0:y1, x0:x1].shape


def test_timestamp_mismatch_is_rejected_before_reusing_any_result():
  locator = TriangleLocator.__new__(TriangleLocator)
  locator.config = {'synchronization': {'max_delta_ms': 1.0}}
  frame = {
    'frame_id': 42,
    'left_timestamp_ms': 100.0,
    'right_timestamp_ms': 103.0,
    'timestamp_delta_ms': 3.0,
  }
  result = locator.process(frame)
  assert result['frame_id'] == 42
  assert not result['plate_valid']
  assert result['invalid_reason'] == 'STEREO_NOT_SYNCHRONIZED'
  for key in ('detection_valid', 'screw_valid', 'plate_confidence', 'quality_score', 'timings_ms'):
    assert key in result
  for key in ('screw_u_rect', 'screw_v_rect', 'screw_u_raw', 'screw_v_raw'):
    assert result[key] is None


def test_left_right_frame_index_mismatch_is_rejected():
  locator = TriangleLocator.__new__(TriangleLocator)
  locator.config = {'synchronization': {'max_delta_ms': 1.0, 'max_frame_index_delta': 0}}
  frame = {
    'frame_id': 43,
    'left_frame_id': 100,
    'right_frame_id': 101,
    'left_timestamp_ms': 100.0,
    'right_timestamp_ms': 100.2,
    'timestamp_delta_ms': 0.2,
  }
  result = locator.process(frame)
  assert result['invalid_reason'] == 'STEREO_NOT_SYNCHRONIZED'


def test_successful_result_exposes_rectified_and_raw_screw_pixels(monkeypatch):
  class Rectifier:
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    baseline_m = 0.018
    calibration = SimpleNamespace(image_width=640, image_height=480)

    @staticmethod
    def rectify(left, right):
      return left + 17, right + 17

    @staticmethod
    def left_rectified_to_raw(points):
      return np.asarray(points, dtype=np.float64) + np.array([2.0, 3.0])

    @staticmethod
    def rectify_left_mask(mask):
      return np.asarray(mask, dtype=bool)

  mask = np.zeros((480, 640), dtype=bool)
  mask[150:330, 220:430] = True
  yolo_input = {}
  monkeypatch.setattr(pipeline, 'measure_vertical_epipolar_error', lambda *args: {
    'valid': True, 'matches': 100, 'median_px': 0.1, 'p95_px': 0.3,
  })
  def fake_yolo(image, *args, **kwargs):
    yolo_input['mean'] = float(np.asarray(image).mean())
    return {
      'target_plate': mask,
      'plate_confidence': 0.9,
      'detection_valid': True,
      'invalid_reason': None,
    }

  monkeypatch.setattr(pipeline.seg_pointcloud, 'run_yolo_seg_array', fake_yolo)
  monkeypatch.setattr(pipeline.seg_pointcloud, 'target_roi', lambda *args, **kwargs: (0, 0, 640, 480))
  monkeypatch.setattr(pipeline.run_demo, 'run_pair_arrays', lambda *args, **kwargs: {
    'disp': np.full((480, 640), 24.0, dtype=np.float32),
    'K': Rectifier.K.copy(),
    'scale': 1.0,
    'valid_disparity_mask': np.ones((480, 640), dtype=bool),
    'valid_disparity_ratio': 1.0,
    'timings_ms': {'ffs_ms': 5.0},
  })
  monkeypatch.setattr(pipeline, 'fit_disparity_plane', lambda *args, **kwargs: SimpleNamespace(
    valid=True,
    invalid_reason=None,
    coefficients=np.array([0.0, 0.0, 24.0]),
    camera_plane=np.array([0.0, 0.0, 1.0, -0.45]),
    metrics={
      'valid_ratio': 1.0,
      'inlier_ratio': 0.95,
      'disparity_rmse_px': 0.1,
      'depth_rmse_mm': 1.0,
      'spatial_coverage': 1.0,
      'quality_score': 0.9,
    },
  ))
  monkeypatch.setattr(pipeline, 'estimate_plate_pose', lambda *args, **kwargs: SimpleNamespace(
    valid=True,
    invalid_reason=None,
    R=np.eye(3),
    t=np.array([0.0, 0.0, 0.5]),
    vertices_xyz=np.array([[-0.1, 0, 0.5], [0.1, 0, 0.5], [0, 0.1, 0.5]]),
    vertices_uv=np.array([[200, 200], [440, 200], [320, 350]], dtype=np.float64),
    metrics={'triangle_iou': 0.9},
  ))

  locator = TriangleLocator.__new__(TriangleLocator)
  locator.model = object()
  locator.model_args = OmegaConf.create({'max_disp': 128, 'scale': 1})
  locator.rectifier = Rectifier()
  locator.config = {
    'synchronization': {'max_delta_ms': 1.0, 'max_frame_index_delta': 0},
    'rectification_quality': {'enabled': True},
    'segmentation': {},
    'roi': {},
    'plane': {},
    'pose': {},
    'debug': {},
  }
  locator.repo_dir = None
  locator.previous_rotation = None
  locator.screw_offset = np.array([0.01, 0.0, 0.0])
  result = locator.process({
    'frame_id': 9,
    'left_frame_id': 9,
    'right_frame_id': 9,
    'left_timestamp_ms': 100.0,
    'right_timestamp_ms': 100.1,
    'timestamp_delta_ms': 0.1,
    'left_bgr': np.zeros((480, 640, 3), dtype=np.uint8),
    'right_bgr': np.zeros((480, 640, 3), dtype=np.uint8),
  })
  assert result['plate_valid']
  assert result['screw_valid']
  assert result['invalid_reason'] == ''
  assert result['screw_u_rect'] == 332.0
  assert result['screw_v_rect'] == 240.0
  assert result['screw_u_raw'] == 334.0
  assert result['screw_v_raw'] == 243.0
  assert yolo_input['mean'] == 0.0

  def unexpected_screw_call(*args, **kwargs):
    raise AssertionError('disabled screw stage must not call locate_screw')

  monkeypatch.setattr(pipeline, 'locate_screw', unexpected_screw_call)
  locator.screw_enabled = False
  plate_only = locator.process({
    'frame_id': 10,
    'left_frame_id': 10,
    'right_frame_id': 10,
    'left_timestamp_ms': 101.0,
    'right_timestamp_ms': 101.1,
    'timestamp_delta_ms': 0.1,
    'left_bgr': np.zeros((480, 640, 3), dtype=np.uint8),
    'right_bgr': np.zeros((480, 640, 3), dtype=np.uint8),
  })
  assert plate_only['plate_valid']
  assert not plate_only['screw_valid']
  assert plate_only['invalid_reason'] == ''
