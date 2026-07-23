from types import SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf

from scripts.seg_pointcloud import select_target_candidate, target_roi
from triangle_locator import pipeline
from triangle_locator.pipeline import (
  TriangleLocator,
  _expand_roi_for_vertices,
  _requires_temporal_pose,
  _stereo_geometry_for_roi,
  select_tracked_target_candidate,
  track_triangle_vertices,
)


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


def test_raw_stereo_geometry_tracks_roi_and_input_scale():
  calibration = SimpleNamespace(
    K1=np.array([[600.0, 0.0, 630.0], [0.0, 610.0, 400.0], [0.0, 0.0, 1.0]]),
    K2=np.array([[602.0, 0.0, 635.0], [0.0, 612.0, 398.0], [0.0, 0.0, 1.0]]),
    R=np.eye(3),
    T=np.array([-0.018, 0.0, 0.0]),
  )
  geometry = _stereo_geometry_for_roi(calibration, (120, 50), scale=2.0)
  assert np.allclose(geometry['K1'][0], [1200.0, 0.0, 1020.0])
  assert np.allclose(geometry['K1'][1], [0.0, 1220.0, 700.0])
  assert np.allclose(geometry['K2'][0], [1204.0, 0.0, 1030.0])
  assert np.allclose(geometry['T'], [-0.018, 0.0, 0.0])


def test_tracking_vertices_expand_shared_roi_with_disparity_context():
  bbox = (180, 100, 280, 220)
  vertices = np.array([[100, 70], [250, 75], [175, 260]], dtype=np.float64)
  expanded = _expand_roi_for_vertices(
    bbox,
    vertices,
    (300, 400, 3),
    max_disp=64,
    context_padding_px=12,
  )
  assert expanded == (24, 58, 280, 273)


def test_residual_mask_candidate_is_replaced_when_it_disagrees_with_tracking():
  config = {
    'max_direct_prediction_error_ratio': 0.30,
    'min_direct_visible_fraction': 0.20,
  }
  assert _requires_temporal_pose(True, 0.331, 0.59, config)
  assert not _requires_temporal_pose(True, 0.12, 0.59, config)
  assert _requires_temporal_pose(True, 0.12, 0.10, config)
  assert _requires_temporal_pose(False, None, 0.80, config)


def test_shrunken_mask_area_uses_temporal_pose_even_when_shape_is_plausible():
  config = {
    'max_direct_prediction_error_ratio': 0.05,
    'min_direct_visible_fraction': 0.75,
    'min_direct_mask_area_ratio': 0.75,
  }
  assert _requires_temporal_pose(True, 0.02, 0.90, config, mask_area_ratio=0.60)
  assert not _requires_temporal_pose(True, 0.02, 0.90, config, mask_area_ratio=0.95)


def test_tracking_resolves_clear_multiple_plate_candidate_without_mask_union():
  predicted = np.array([[40, 20], [120, 25], [80, 100]], dtype=np.float64)
  full = np.zeros((130, 160), dtype=bool)
  cv2.fillConvexPoly(full.view(np.uint8), predicted.astype(np.int32), 1)
  residual = full.copy()
  residual[:, 85:] = False
  selected, diagnostics = select_tracked_target_candidate(
    [
      {'mask': residual, 'confidence': 0.92},
      {'mask': full, 'confidence': 0.91},
    ],
    predicted,
    {},
    {
      'min_candidate_visible_fraction': 0.25,
      'min_candidate_mask_containment': 0.70,
      'min_candidate_score': 0.15,
      'min_candidate_score_margin': 0.08,
    },
  )
  assert selected is not None
  assert diagnostics['ambiguity_resolved']
  assert diagnostics['selected_index'] == 1
  np.testing.assert_array_equal(selected['mask'], full)


def test_tracking_keeps_multiple_plate_invalid_when_candidates_are_equally_plausible():
  predicted = np.array([[40, 20], [120, 25], [80, 100]], dtype=np.float64)
  mask = np.zeros((130, 160), dtype=np.uint8)
  cv2.fillConvexPoly(mask, predicted.astype(np.int32), 1)
  selected, diagnostics = select_tracked_target_candidate(
    [
      {'mask': mask.astype(bool), 'confidence': 0.92},
      {'mask': mask.astype(bool), 'confidence': 0.91},
    ],
    predicted,
    {},
    {'min_candidate_score_margin': 0.08},
  )
  assert selected is None
  assert not diagnostics['ambiguity_resolved']


def test_optical_flow_updates_triangle_with_current_image_motion():
  rng = np.random.default_rng(7)
  previous = rng.integers(0, 256, (240, 320), dtype=np.uint8)
  translation = np.array([7.0, 5.0])
  current = cv2.warpAffine(
    previous,
    np.array([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]]),
    (320, 240),
  )
  current = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
  vertices = np.array([[80, 55], [220, 60], [150, 195]], dtype=np.float64)
  predicted, metrics = track_triangle_vertices(previous, current, vertices, {
    'temporal': {'enabled': True, 'max_rotation_deg': 10.0},
  })
  assert metrics['valid'], metrics
  assert metrics['inlier_ratio'] > 0.9
  np.testing.assert_allclose(predicted, vertices + translation, atol=0.5)


def test_optical_flow_prefers_plate_mask_over_differently_moving_background():
  rng = np.random.default_rng(11)
  previous = rng.integers(0, 256, (240, 320), dtype=np.uint8)
  vertices = np.array([[90, 60], [230, 65], [160, 195]], dtype=np.float64)
  plate_mask = np.zeros(previous.shape, dtype=np.uint8)
  cv2.fillConvexPoly(plate_mask, vertices.astype(np.int32), 1)
  plate_translation = np.array([6.0, 4.0])
  background = cv2.warpAffine(
    previous,
    np.array([[1.0, 0.0, -9.0], [0.0, 1.0, 1.0]]),
    (320, 240),
  )
  moved_plate = cv2.warpAffine(
    previous,
    np.array([[1.0, 0.0, plate_translation[0]], [0.0, 1.0, plate_translation[1]]]),
    (320, 240),
  )
  moved_mask = cv2.warpAffine(
    plate_mask,
    np.array([[1.0, 0.0, plate_translation[0]], [0.0, 1.0, plate_translation[1]]]),
    (320, 240),
    flags=cv2.INTER_NEAREST,
  ).astype(bool)
  current = background
  current[moved_mask] = moved_plate[moved_mask]
  predicted, metrics = track_triangle_vertices(
    previous,
    cv2.cvtColor(current, cv2.COLOR_GRAY2BGR),
    vertices,
    {'temporal': {'enabled': True, 'mask_feature_padding_px': 0}},
    plate_mask.astype(bool),
  )
  assert metrics['valid'], metrics
  np.testing.assert_allclose(predicted, vertices + plate_translation, atol=0.75)


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


def test_temporal_pose_uses_current_frame_then_expires(monkeypatch):
  class Rectifier:
    K = np.array([[600.0, 0, 160.0], [0, 600.0, 120.0], [0, 0, 1.0]])
    baseline_m = 0.018
    calibration = SimpleNamespace(image_width=320, image_height=240)

    @staticmethod
    def rectify(left, right):
      return left, right

  mask = np.zeros((240, 320), dtype=bool)
  mask[70:190, 80:240] = True
  monkeypatch.setattr(pipeline.seg_pointcloud, 'run_yolo_seg_array', lambda *args, **kwargs: {
    'target_plate': mask,
    'plate_confidence': 0.9,
    'detection_valid': True,
    'invalid_reason': None,
  })
  monkeypatch.setattr(pipeline.seg_pointcloud, 'target_roi', lambda *args, **kwargs: (0, 0, 320, 240))
  monkeypatch.setattr(pipeline.run_demo, 'run_pair_arrays', lambda *args, **kwargs: {
    'disp': np.full((240, 320), 24.0, dtype=np.float32),
    'K': Rectifier.K.copy(),
    'scale': 1.0,
    'valid_disparity_mask': np.ones((240, 320), dtype=bool),
    'valid_disparity_ratio': 1.0,
    'timings_ms': {'ffs_ms': 1.0},
  })
  plane = SimpleNamespace(
    valid=True,
    invalid_reason=None,
    coefficients=np.array([0.0, 0.0, 24.0]),
    camera_plane=np.array([0.0, 0.0, 1.0, -0.45]),
    metrics={
      'valid_ratio': 1.0,
      'inlier_ratio': 0.9,
      'disparity_rmse_px': 0.1,
      'depth_rmse_mm': 1.0,
      'spatial_coverage': 1.0,
      'quality_score': 0.9,
    },
  )
  monkeypatch.setattr(pipeline, 'fit_disparity_plane', lambda *args, **kwargs: plane)
  monkeypatch.setattr(pipeline, 'estimate_plate_pose', lambda *args, **kwargs: SimpleNamespace(
    valid=False,
    invalid_reason='TRIANGLE_FIT_FAILED',
    vertices_uv=None,
    metrics={'triangle_iou': 0.2},
  ))
  monkeypatch.setattr(pipeline, 'track_triangle_vertices', lambda _gray, _image, vertices, _config, _mask: (
    np.asarray(vertices) + np.array([3.0, 2.0]),
    {
      'valid': True,
      'reason': '',
      'inlier_ratio': 0.9,
      'direct_prediction_error_ratio': None,
    },
  ))

  def temporal_pose(vertices, *_args, metrics=None, **_kwargs):
    vertices = np.asarray(vertices, dtype=np.float64)
    return SimpleNamespace(
      valid=True,
      invalid_reason=None,
      R=np.eye(3),
      t=np.array([0.0, 0.0, 0.5]),
      vertices_xyz=np.column_stack((vertices / 1000.0, np.full(3, 0.5))),
      vertices_uv=vertices,
      metrics=metrics or {},
    )

  monkeypatch.setattr(pipeline, 'estimate_pose_from_vertices', temporal_pose)
  locator = TriangleLocator.__new__(TriangleLocator)
  locator.model = object()
  locator.model_args = OmegaConf.create({'max_disp': 128, 'scale': 1})
  locator.rectifier = Rectifier()
  locator.config = {
    'synchronization': {'require_timestamps': False},
    'rectification_quality': {'enabled': False},
    'segmentation': {'input_coordinate': 'rectified_left'},
    'roi': {},
    'plane': {},
    'pose': {'temporal': {'enabled': True, 'max_estimated_frames': 4}},
    'debug': {},
  }
  locator.repo_dir = None
  locator.screw_enabled = False
  locator.screw_offset = None
  locator.reset_tracking()
  locator.previous_vertices_left_uv = np.array([[80, 60], [220, 65], [150, 190]], dtype=np.float64)
  locator.previous_left_gray = np.zeros((240, 320), dtype=np.uint8)
  locator.previous_frame_id = 1
  locator.previous_timestamp_ms = None
  locator.previous_rotation = np.eye(3)

  def frame(frame_id):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
      'frame_id': frame_id,
      'left_frame_id': frame_id,
      'right_frame_id': frame_id,
      'left_bgr': image,
      'right_bgr': image.copy(),
    }

  estimated = locator.process(frame(2))
  assert estimated['plate_valid']
  assert estimated['pose_source'] == 'TEMPORAL'
  assert estimated['temporal_age'] == 1
  assert 0.0 < estimated['pose_confidence'] < plane.metrics['quality_score']
  # A temporal estimate is projected from the reliable direct anchor; it must
  # not replace that anchor and accumulate mask drift on the next frame.
  assert locator.previous_frame_id == 1
  np.testing.assert_allclose(
    locator.previous_vertices_left_uv,
    np.array([[80, 60], [220, 65], [150, 190]], dtype=np.float64),
  )

  locator.temporal_age = 4
  expired = locator.process(frame(3))
  assert not expired['plate_valid']
  assert expired['pose_source'] == 'INVALID'
  assert expired['invalid_reason'] == 'TEMPORAL_ESTIMATE_EXPIRED'
