import numpy as np

from scripts.seg_pointcloud import select_target_candidate, target_roi
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
