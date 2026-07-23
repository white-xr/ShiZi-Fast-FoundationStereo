import json
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from scripts import run_demo, seg_pointcloud
from .calibration import adjust_intrinsics_for_roi, measure_vertical_epipolar_error
from .plate_pose import (
  estimate_plate_pose,
  estimate_pose_from_vertices,
  support_triangle_vertices,
  triangle_geometry_metrics,
)
from .screw_locator import load_screw_offset, locate_screw
from .stereo_plane import fit_disparity_plane


INVALID_REASONS = {
  'NO_PLATE',
  'MULTIPLE_PLATES',
  'STEREO_NOT_SYNCHRONIZED',
  'RECTIFICATION_INVALID',
  'INSUFFICIENT_DISPARITY',
  'PLANE_FIT_FAILED',
  'PLANE_QUALITY_LOW',
  'TRIANGLE_FIT_FAILED',
  'TRIANGLE_DIRECTION_INVALID',
  'POSE_DISCONTINUITY',
  'POSE_JUMP_REJECTED',
  'TEMPORAL_ESTIMATE_EXPIRED',
  'SCREW_OFFSET_NOT_CONFIGURED',
  'PROJECTION_OUT_OF_IMAGE',
}


def _json_value(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {key: _json_value(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_value(item) for item in value]
  return value


def write_result_json(path, result):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(_json_value(result), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append_result_jsonl(path, result):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(_json_value(result), ensure_ascii=False, separators=(',', ':')) + '\n')


def _base_result(frame):
  left_ts = frame.get('left_timestamp_ms')
  right_ts = frame.get('right_timestamp_ms')
  timestamp = frame.get('timestamp_ms')
  if timestamp is None and left_ts is not None and right_ts is not None:
    timestamp = (float(left_ts) + float(right_ts)) / 2.0
  return {
    'frame_id': int(frame.get('frame_id', 0)),
    'timestamp_ms': timestamp,
    'left_timestamp_ms': left_ts,
    'right_timestamp_ms': right_ts,
    'timestamp_delta_ms': frame.get('timestamp_delta_ms'),
    'left_frame_id': frame.get('left_frame_id'),
    'right_frame_id': frame.get('right_frame_id'),
    'detection_valid': False,
    'plate_valid': False,
    'screw_valid': False,
    'plate_confidence': 0.0,
    'screw_u_rect': None,
    'screw_v_rect': None,
    'screw_u_raw': None,
    'screw_v_raw': None,
    'screw_xyz_camera_m': None,
    'plate_origin_camera_m': None,
    'plate_rotation_camera': None,
    'plate_vertices_camera_m': None,
    'plate_vertices_left_uv': None,
    'candidate_vertices_left_uv': None,
    'pose_source': 'INVALID',
    'pose_confidence': 0.0,
    'temporal_age': 0,
    'triangle_metrics': None,
    'tracking': None,
    'plane': None,
    'valid_ratio': 0.0,
    'inlier_ratio': 0.0,
    'plane_rmse_px': None,
    'depth_rmse_mm': None,
    'spatial_coverage': 0.0,
    'quality_score': 0.0,
    'ffs_valid_disparity_ratio': 0.0,
    'rectification': None,
    'timings_ms': {
      'capture_ms': float(frame.get('capture_ms', 0.0) or 0.0),
      'rectify_ms': 0.0,
      'yolo_ms': 0.0,
      'ffs_ms': 0.0,
      'plane_ms': 0.0,
      'pose_ms': 0.0,
      'total_ms': 0.0,
    },
    'invalid_reason': '',
  }


def _frame_timestamp_ms(frame):
  timestamp = frame.get('timestamp_ms')
  if timestamp is not None:
    return float(timestamp)
  left = frame.get('left_timestamp_ms')
  right = frame.get('right_timestamp_ms')
  if left is not None and right is not None:
    return (float(left) + float(right)) / 2.0
  return None


def _vertex_error_ratio(first, second):
  first = np.asarray(first, dtype=np.float64).reshape(3, 2)
  second = np.asarray(second, dtype=np.float64).reshape(3, 2)
  first_base = np.linalg.norm(first[1] - first[0])
  second_base = np.linalg.norm(second[1] - second[0])
  scale = max((first_base + second_base) / 2.0, 1e-6)
  return float(np.sqrt(np.mean(np.sum(np.square(first - second), axis=1))) / scale)


def _requires_temporal_pose(
  direct_valid,
  prediction_error,
  visible_fraction,
  config,
  mask_area_ratio=None,
):
  if not direct_valid:
    return True
  return (
    prediction_error is not None
    and prediction_error > float(config.get('max_direct_prediction_error_ratio', 0.30))
  ) or (
    visible_fraction is not None
    and visible_fraction < float(config.get('min_direct_visible_fraction', 0.20))
  ) or (
    mask_area_ratio is not None
    and mask_area_ratio < float(config.get('min_direct_mask_area_ratio', 0.0))
  )


def _triangle_mask_coverage(mask, vertices):
  mask = np.asarray(mask, dtype=bool)
  triangle = np.zeros(mask.shape, dtype=np.uint8)
  cv2.fillConvexPoly(triangle, np.round(vertices).astype(np.int32), 1)
  triangle = triangle.astype(bool)
  intersection = int(np.count_nonzero(mask & triangle))
  triangle_area = int(np.count_nonzero(triangle))
  mask_area = int(np.count_nonzero(mask))
  return {
    'prediction_visible_fraction': float(intersection / triangle_area) if triangle_area else 0.0,
    'prediction_mask_containment': float(intersection / mask_area) if mask_area else 0.0,
  }


def select_tracked_target_candidate(
  candidates,
  predicted_vertices,
  segmentation_config,
  temporal_config,
  mask_transform=None,
):
  diagnostics = {
    'candidate_count': len(candidates),
    'ambiguity_resolved': False,
    'selected_index': None,
    'best_score': None,
    'score_margin': None,
  }
  if predicted_vertices is None or not candidates:
    return None, diagnostics
  scored = []
  for index, candidate in enumerate(candidates):
    source_mask = seg_pointcloud.postprocess_mask(candidate['mask'], segmentation_config)
    evaluation_mask = mask_transform(source_mask) if mask_transform is not None else source_mask
    coverage = _triangle_mask_coverage(evaluation_mask, predicted_vertices)
    score = coverage['prediction_visible_fraction'] * coverage['prediction_mask_containment']
    scored.append({
      'index': index,
      'candidate': candidate,
      'source_mask': source_mask,
      'score': float(score),
      **coverage,
    })
  scored.sort(
    key=lambda item: (item['score'], float(item['candidate']['confidence'])),
    reverse=True,
  )
  best = scored[0]
  second_score = scored[1]['score'] if len(scored) > 1 else 0.0
  margin = best['score'] - second_score
  diagnostics.update({
    'selected_index': best['index'],
    'best_score': best['score'],
    'score_margin': margin,
    'prediction_visible_fraction': best['prediction_visible_fraction'],
    'prediction_mask_containment': best['prediction_mask_containment'],
  })
  valid = (
    best['prediction_visible_fraction'] >= float(
      temporal_config.get('min_candidate_visible_fraction', 0.25)
    )
    and best['prediction_mask_containment'] >= float(
      temporal_config.get('min_candidate_mask_containment', 0.70)
    )
    and best['score'] >= float(temporal_config.get('min_candidate_score', 0.15))
    and margin >= float(temporal_config.get('min_candidate_score_margin', 0.08))
  )
  if not valid:
    diagnostics['selected_index'] = None
    return None, diagnostics
  diagnostics['ambiguity_resolved'] = True
  selected = dict(best['candidate'])
  selected['mask'] = best['source_mask']
  return selected, diagnostics


def _expand_roi_for_vertices(bbox, vertices, image_shape, max_disp, context_padding_px):
  if vertices is None:
    return bbox
  height, width = image_shape[:2]
  vertices = np.asarray(vertices, dtype=np.float64).reshape(3, 2)
  context = max(0, int(context_padding_px))
  tracking_x0 = int(np.floor(vertices[:, 0].min() - float(max_disp) - context))
  tracking_y0 = int(np.floor(vertices[:, 1].min() - context))
  tracking_x1 = int(np.ceil(vertices[:, 0].max() + context)) + 1
  tracking_y1 = int(np.ceil(vertices[:, 1].max() + context)) + 1
  x0, y0, x1, y1 = bbox
  return (
    max(0, min(x0, tracking_x0)),
    max(0, min(y0, tracking_y0)),
    min(width, max(x1, tracking_x1)),
    min(height, max(y1, tracking_y1)),
  )


def _stereo_geometry_for_roi(calibration, roi_xy, scale):
  """Express raw stereo calibration in the scaled, cropped FFS coordinates."""
  return {
    'K1': adjust_intrinsics_for_roi(calibration.K1, roi_xy, scale=scale),
    'K2': adjust_intrinsics_for_roi(calibration.K2, roi_xy, scale=scale),
    'R': np.asarray(calibration.R, dtype=np.float64),
    'T': np.asarray(calibration.T, dtype=np.float64),
  }


def track_triangle_vertices(
  previous_gray,
  current_bgr,
  vertices,
  pose_config=None,
  previous_target_mask=None,
):
  pose_config = pose_config or {}
  config = pose_config.get('temporal') or {}
  metrics = {
    'valid': False,
    'reason': 'DISABLED' if not bool(config.get('enabled', True)) else 'UNAVAILABLE',
    'detected_points': 0,
    'tracked_points': 0,
    'inlier_points': 0,
    'inlier_ratio': 0.0,
    'median_fb_error_px': None,
    'scale': None,
    'rotation_deg': None,
    'direct_prediction_error_ratio': None,
    'prediction_visible_fraction': None,
    'prediction_mask_containment': None,
  }
  if not bool(config.get('enabled', True)) or previous_gray is None or vertices is None:
    return None, metrics

  previous_gray = np.asarray(previous_gray, dtype=np.uint8)
  current = np.asarray(current_bgr)
  current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current.astype(np.uint8)
  vertices = np.asarray(vertices, dtype=np.float32).reshape(3, 2)
  triangle_mask = np.zeros(previous_gray.shape, dtype=np.uint8)
  cv2.fillConvexPoly(triangle_mask, np.round(vertices).astype(np.int32), 255)
  if previous_target_mask is not None and np.asarray(previous_target_mask).shape == previous_gray.shape:
    feature_mask = np.asarray(previous_target_mask, dtype=bool).astype(np.uint8) * 255
    feature_mask = cv2.bitwise_and(feature_mask, triangle_mask)
    padding = max(0, int(config.get('mask_feature_padding_px', 4)))
  else:
    feature_mask = triangle_mask
    padding = max(0, int(config.get('feature_padding_px', 28)))
  if padding:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
    feature_mask = cv2.dilate(feature_mask, kernel)

  points = cv2.goodFeaturesToTrack(
    previous_gray,
    maxCorners=int(config.get('max_features', 120)),
    qualityLevel=float(config.get('feature_quality', 0.01)),
    minDistance=float(config.get('feature_min_distance_px', 5.0)),
    mask=feature_mask,
    blockSize=int(config.get('feature_block_size', 7)),
  )
  if points is None:
    metrics['reason'] = 'NO_FEATURES'
    return None, metrics
  metrics['detected_points'] = int(len(points))
  window = max(9, int(config.get('lk_window_px', 31)))
  if window % 2 == 0:
    window += 1
  lk_options = {
    'winSize': (window, window),
    'maxLevel': int(config.get('lk_max_level', 4)),
    'criteria': (
      cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
      int(config.get('lk_iterations', 30)),
      float(config.get('lk_epsilon', 0.01)),
    ),
  }
  current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
    previous_gray, current_gray, points, None, **lk_options,
  )
  if current_points is None or forward_status is None:
    metrics['reason'] = 'FORWARD_FLOW_FAILED'
    return None, metrics
  backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
    current_gray, previous_gray, current_points, None, **lk_options,
  )
  if backward_points is None or backward_status is None:
    metrics['reason'] = 'BACKWARD_FLOW_FAILED'
    return None, metrics

  forward_status = forward_status.reshape(-1).astype(bool)
  backward_status = backward_status.reshape(-1).astype(bool)
  fb_error = np.linalg.norm(points.reshape(-1, 2) - backward_points.reshape(-1, 2), axis=1)
  good = forward_status & backward_status & np.isfinite(fb_error)
  good &= fb_error <= float(config.get('max_fb_error_px', 1.5))
  source_points = points.reshape(-1, 2)[good]
  destination_points = current_points.reshape(-1, 2)[good]
  metrics['tracked_points'] = int(len(source_points))
  min_tracks = int(config.get('min_tracked_points', 8))
  if len(source_points) < min_tracks:
    metrics['reason'] = 'INSUFFICIENT_TRACKS'
    return None, metrics

  transform, inliers = cv2.estimateAffinePartial2D(
    source_points,
    destination_points,
    method=cv2.RANSAC,
    ransacReprojThreshold=float(config.get('ransac_threshold_px', 2.0)),
    maxIters=int(config.get('ransac_iterations', 500)),
    confidence=float(config.get('ransac_confidence', 0.99)),
    refineIters=int(config.get('ransac_refine_iterations', 10)),
  )
  if transform is None or inliers is None:
    metrics['reason'] = 'AFFINE_FIT_FAILED'
    return None, metrics
  inlier_count = int(np.count_nonzero(inliers))
  inlier_ratio = float(inlier_count / len(source_points))
  metrics['inlier_points'] = inlier_count
  metrics['inlier_ratio'] = inlier_ratio
  metrics['median_fb_error_px'] = float(np.median(fb_error[good]))
  if (
    inlier_count < int(config.get('min_inlier_points', 6))
    or inlier_ratio < float(config.get('min_inlier_ratio', 0.5))
  ):
    metrics['reason'] = 'TRACK_QUALITY_LOW'
    return None, metrics

  scale = float(np.hypot(transform[0, 0], transform[1, 0]))
  rotation_deg = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
  metrics['scale'] = scale
  metrics['rotation_deg'] = rotation_deg
  if not (
    float(config.get('min_scale', 0.65)) <= scale <= float(config.get('max_scale', 1.55))
    and abs(rotation_deg) <= float(config.get('max_rotation_deg', 35.0))
  ):
    metrics['reason'] = 'TRACK_TRANSFORM_IMPLAUSIBLE'
    return None, metrics

  predicted = cv2.transform(vertices.reshape(1, 3, 2), transform).reshape(3, 2).astype(np.float64)
  geometry = triangle_geometry_metrics(predicted, pose_config)
  if not geometry['valid']:
    metrics['reason'] = 'TRACK_DIRECTION_INVALID'
    return None, metrics
  metrics['valid'] = True
  metrics['reason'] = ''
  return geometry['vertices'], metrics


class TriangleLocator:
  def __init__(self, model, model_args, rectifier, config, repo_dir):
    self.model = model
    self.model_args = model_args
    self.rectifier = rectifier
    self.config = config
    self.repo_dir = Path(repo_dir)
    screw_config = config.get('screw') or {}
    self.screw_enabled = bool(screw_config.get('enabled', True))
    if self.screw_enabled:
      offset_path = screw_config.get('offset_file')
      if offset_path and not Path(offset_path).is_absolute():
        offset_path = self.repo_dir / offset_path
      self.screw_offset = load_screw_offset(offset_path)
    else:
      self.screw_offset = None
    self.last_target_mask = None
    self.last_target_mask_coordinate = None
    self.reset_tracking()

  def reset_tracking(self):
    self.previous_rotation = None
    self.previous_vertices_left_uv = None
    self.previous_left_gray = None
    self.previous_target_mask = None
    self.previous_frame_id = None
    self.previous_timestamp_ms = None
    self.previous_target_mask_area_px = None
    self.temporal_age = 0

  def _tracking_prediction(self, frame, left_rect, pose_config):
    temporal = pose_config.get('temporal') or {}
    metrics = {
      'valid': False,
      'reason': 'NO_HISTORY',
      'detected_points': 0,
      'tracked_points': 0,
      'inlier_points': 0,
      'inlier_ratio': 0.0,
      'median_fb_error_px': None,
      'scale': None,
      'rotation_deg': None,
      'direct_prediction_error_ratio': None,
      'prediction_visible_fraction': None,
      'prediction_mask_containment': None,
    }
    if (
      getattr(self, 'previous_vertices_left_uv', None) is None
      or getattr(self, 'previous_left_gray', None) is None
    ):
      return None, metrics
    frame_id = int(frame.get('frame_id', 0))
    frame_gap = frame_id - int(self.previous_frame_id)
    metrics['frame_gap'] = frame_gap
    if frame_gap <= 0 or frame_gap > int(temporal.get('max_frame_gap', 60)):
      metrics['reason'] = 'FRAME_GAP'
      self.reset_tracking()
      return None, metrics
    timestamp = _frame_timestamp_ms(frame)
    if timestamp is not None and self.previous_timestamp_ms is not None:
      interval = timestamp - float(self.previous_timestamp_ms)
      metrics['interval_ms'] = interval
      if interval <= 0 or interval > float(temporal.get('max_interval_ms', 1500.0)):
        metrics['reason'] = 'TIME_GAP'
        self.reset_tracking()
        return None, metrics

    predicted, tracking = track_triangle_vertices(
      self.previous_left_gray,
      left_rect,
      self.previous_vertices_left_uv,
      pose_config,
      self.previous_target_mask if bool(temporal.get('anchor_use_target_mask', False)) else None,
    )
    tracking['frame_gap'] = frame_gap
    if timestamp is not None and self.previous_timestamp_ms is not None:
      tracking['interval_ms'] = timestamp - float(self.previous_timestamp_ms)
    if predicted is None:
      self.reset_tracking()
    return predicted, tracking

  def _update_tracking(self, frame, left_rect, target_mask, vertices, rotation, temporal_age):
    self.previous_vertices_left_uv = np.asarray(vertices, dtype=np.float64).reshape(3, 2).copy()
    self.previous_left_gray = cv2.cvtColor(np.asarray(left_rect), cv2.COLOR_BGR2GRAY)
    self.previous_target_mask = np.asarray(target_mask, dtype=bool).copy()
    self.previous_frame_id = int(frame.get('frame_id', 0))
    self.previous_timestamp_ms = _frame_timestamp_ms(frame)
    self.previous_target_mask_area_px = int(np.count_nonzero(target_mask))
    self.previous_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
    self.temporal_age = int(temporal_age)

  def _update_temporal_pose(self, rotation, temporal_age):
    self.previous_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3).copy()
    self.temporal_age = int(temporal_age)

  def _finish(self, result, started, reason=None):
    if reason is not None:
      if reason not in INVALID_REASONS:
        raise ValueError(f'unknown invalid_reason: {reason}')
      result['invalid_reason'] = reason
    result['timings_ms']['total_ms'] = (time.perf_counter() - started) * 1000.0
    return _json_value(result)

  def process(self, frame, output_dir=None, debug=False, save_preview=False):
    started = time.perf_counter()
    result = _base_result(frame)
    self.last_target_mask = None
    self.last_target_mask_coordinate = None
    synchronization = self.config.get('synchronization') or {}
    delta = frame.get('timestamp_delta_ms')
    require_timestamps = bool(synchronization.get('require_timestamps', True))
    if (require_timestamps and delta is None) or (
      delta is not None and float(delta) > float(synchronization.get('max_delta_ms', 2.0))
    ):
      return self._finish(result, started, 'STEREO_NOT_SYNCHRONIZED')
    left_frame_id = frame.get('left_frame_id')
    right_frame_id = frame.get('right_frame_id')
    if left_frame_id is not None and right_frame_id is not None:
      if abs(int(left_frame_id) - int(right_frame_id)) > int(synchronization.get('max_frame_index_delta', 0)):
        return self._finish(result, started, 'STEREO_NOT_SYNCHRONIZED')

    left_bgr = np.asarray(frame['left_bgr'])
    right_bgr = np.asarray(frame['right_bgr'])
    stage = time.perf_counter()
    try:
      left_rect, right_rect = self.rectifier.rectify(left_bgr, right_bgr)
    except (ValueError, cv2.error):
      return self._finish(result, started, 'RECTIFICATION_INVALID')
    result['timings_ms']['rectify_ms'] = (time.perf_counter() - stage) * 1000.0

    rectification_config = self.config.get('rectification_quality') or {}
    if bool(rectification_config.get('enabled', True)):
      rectification = measure_vertical_epipolar_error(left_rect, right_rect, rectification_config)
      result['rectification'] = {
        'matches': rectification['matches'],
        'vertical_error_median_px': rectification['median_px'],
        'vertical_error_p95_px': rectification['p95_px'],
      }
      if not rectification['valid']:
        return self._finish(result, started, 'RECTIFICATION_INVALID')

    pose_config = self.config.get('pose') or {}
    raw_stereo_calibration = (
      None if bool(getattr(self.rectifier, 'enabled', True)) else self.rectifier.calibration
    )
    predicted_vertices_full, tracking = self._tracking_prediction(frame, left_rect, pose_config)
    result['tracking'] = tracking
    previous_rotation = getattr(self, 'previous_rotation', None)
    previous_temporal_age = int(getattr(self, 'temporal_age', 0))

    segmentation = self.config.get('segmentation') or {}
    segmentation_coordinate = str(segmentation.get('input_coordinate', 'raw_left')).strip().lower()
    if segmentation_coordinate not in {'raw_left', 'rectified_left'}:
      raise ValueError(f'unsupported segmentation.input_coordinate: {segmentation_coordinate}')
    segmentation_image = left_bgr if segmentation_coordinate == 'raw_left' else left_rect
    stage = time.perf_counter()
    masks = seg_pointcloud.run_yolo_seg_array(
      segmentation_image,
      output_dir,
      segmentation,
      self.repo_dir,
      write_debug=bool(debug and segmentation.get('debug_vis', True)),
    )
    result['timings_ms']['yolo_ms'] = (time.perf_counter() - stage) * 1000.0
    target_mask_source = np.asarray(masks[seg_pointcloud.TARGET_CLASS], dtype=bool)
    detection_valid = bool(masks['detection_valid'])
    plate_confidence = float(masks['plate_confidence'])
    if (
      not detection_valid
      and masks['invalid_reason'] == 'MULTIPLE_PLATES'
      and predicted_vertices_full is not None
    ):
      mask_transform = (
        self.rectifier.rectify_left_mask if segmentation_coordinate == 'raw_left' else None
      )
      selected, ambiguity = select_tracked_target_candidate(
        masks.get('target_candidates') or [],
        support_triangle_vertices(predicted_vertices_full, pose_config),
        segmentation,
        pose_config.get('temporal') or {},
        mask_transform=mask_transform,
      )
      result['tracking']['ambiguity'] = ambiguity
      if selected is not None:
        target_mask_source = np.asarray(selected['mask'], dtype=bool)
        plate_confidence = float(selected['confidence'])
        detection_valid = True
    result['plate_confidence'] = plate_confidence
    result['detection_valid'] = detection_valid
    self.last_target_mask = target_mask_source.copy()
    self.last_target_mask_coordinate = segmentation_coordinate
    if not detection_valid:
      return self._finish(result, started, masks['invalid_reason'])

    target_mask = target_mask_source
    if segmentation_coordinate == 'raw_left':
      target_mask = self.rectifier.rectify_left_mask(target_mask)
    if predicted_vertices_full is not None:
      result['tracking'].update(_triangle_mask_coverage(target_mask, predicted_vertices_full))
      anchor_area = getattr(self, 'previous_target_mask_area_px', None)
      tracking_scale = result['tracking'].get('scale')
      if anchor_area and tracking_scale is not None and float(tracking_scale) > 0:
        expected_area = float(anchor_area) * float(tracking_scale) ** 2
        result['tracking']['expected_mask_area_px'] = expected_area
        result['tracking']['mask_area_ratio'] = float(
          np.count_nonzero(target_mask) / max(expected_area, 1.0)
        )

    roi_config = self.config.get('roi') or {}
    max_disp_input = float(self.model_args.max_disp) / float(self.model_args.scale)
    context_padding = int(roi_config.get('context_padding_px', 32))
    bbox = seg_pointcloud.target_roi(
      target_mask,
      max_disp=max_disp_input,
      context_padding_px=context_padding,
      padding_px=int(roi_config.get('padding_px', 0)),
      min_size_px=int(roi_config.get('min_size_px', 160)),
    )
    if bbox is None:
      return self._finish(result, started, 'NO_PLATE')
    tracking_support_vertices = None
    if predicted_vertices_full is not None:
      tracking_support_vertices = support_triangle_vertices(predicted_vertices_full, pose_config)
    bbox = _expand_roi_for_vertices(
      bbox,
      tracking_support_vertices,
      left_rect.shape,
      max_disp_input,
      context_padding,
    )
    x0, y0, x1, y1 = bbox
    left_crop_bgr = left_rect[y0:y1, x0:x1]
    right_crop_bgr = right_rect[y0:y1, x0:x1]
    target_crop = target_mask[y0:y1, x0:x1]
    K_roi = adjust_intrinsics_for_roi(self.rectifier.K, (x0, y0))

    sample_args = OmegaConf.create(OmegaConf.to_container(self.model_args, resolve=True))
    debug_config = self.config.get('debug') or {}
    sample_args.save_inputs = int(bool(debug and debug_config.get('save_inputs', False)))
    sample_args.save_disp = int(bool(debug and debug_config.get('save_disp', False)))
    sample_args.save_disp_vis = int(bool(debug and debug_config.get('save_disp_vis', False)))
    sample_args.save_depth = int(bool(debug and debug_config.get('save_depth', False)))
    sample_args.save_report = 0
    sample_args.get_pc = int(bool(debug and debug_config.get('save_point_cloud', False)))
    sample_args.show = 0
    ffs_outputs = run_demo.run_pair_arrays(
      self.model,
      sample_args,
      cv2.cvtColor(left_crop_bgr, cv2.COLOR_BGR2RGB),
      cv2.cvtColor(right_crop_bgr, cv2.COLOR_BGR2RGB),
      output_dir,
      clean_out_dir=False,
      intrinsic_matrix=K_roi,
      baseline_m=self.rectifier.baseline_m,
    )
    result['timings_ms']['ffs_ms'] = ffs_outputs['timings_ms']['ffs_ms']
    result['ffs_valid_disparity_ratio'] = ffs_outputs['valid_disparity_ratio']
    if target_crop.shape != ffs_outputs['disp'].shape:
      target_crop = cv2.resize(
        target_crop.astype(np.uint8),
        (ffs_outputs['disp'].shape[1], ffs_outputs['disp'].shape[0]),
        interpolation=cv2.INTER_NEAREST,
      ).astype(bool)

    stereo_geometry = None
    if raw_stereo_calibration is not None:
      stereo_geometry = _stereo_geometry_for_roi(
        raw_stereo_calibration,
        (x0, y0),
        float(ffs_outputs['scale']),
      )

    stage = time.perf_counter()
    plane_config = dict(self.config.get('plane') or {})
    plane_config.setdefault('max_disp', float(self.model_args.max_disp))
    plane = fit_disparity_plane(
      ffs_outputs['disp'],
      target_crop,
      ffs_outputs['K'],
      self.rectifier.baseline_m,
      plane_config,
      validity_mask=ffs_outputs['valid_disparity_mask'],
      stereo_geometry=stereo_geometry,
    )
    result['timings_ms']['plane_ms'] = (time.perf_counter() - stage) * 1000.0
    result['valid_ratio'] = plane.metrics['valid_ratio']
    result['inlier_ratio'] = plane.metrics['inlier_ratio']
    result['plane_rmse_px'] = plane.metrics['disparity_rmse_px']
    result['depth_rmse_mm'] = plane.metrics['depth_rmse_mm']
    result['spatial_coverage'] = plane.metrics['spatial_coverage']
    result['quality_score'] = plane.metrics['quality_score']
    if plane.coefficients is not None:
      result['plane'] = {
        'disparity_coefficients_roi': plane.coefficients,
        'camera_plane': plane.camera_plane,
        'roi_xyxy': [x0, y0, x1, y1],
      }
    if not plane.valid:
      return self._finish(result, started, plane.invalid_reason)

    stage = time.perf_counter()
    direct_pose = estimate_plate_pose(
      target_crop,
      plane,
      ffs_outputs['K'],
      self.rectifier.baseline_m,
      pose_config,
      previous_rotation=previous_rotation,
      stereo_geometry=stereo_geometry,
    )
    result['triangle_metrics'] = direct_pose.metrics
    scale = float(ffs_outputs['scale'])
    roi_origin = np.array([x0, y0], dtype=np.float64)
    direct_vertices_full = None
    if direct_pose.vertices_uv is not None:
      direct_vertices_full = direct_pose.vertices_uv / scale + roi_origin
      result['candidate_vertices_left_uv'] = direct_vertices_full

    temporal_config = pose_config.get('temporal') or {}
    selected_pose = direct_pose
    pose_source = 'DIRECT' if direct_pose.valid else 'INVALID'
    temporal_age = 0
    temporal_requested = False
    direct_prediction_error = None
    if predicted_vertices_full is not None:
      prediction_visible_fraction = result['tracking'].get('prediction_visible_fraction')
      if direct_pose.valid and direct_vertices_full is not None:
        direct_prediction_error = _vertex_error_ratio(direct_vertices_full, predicted_vertices_full)
        result['tracking']['direct_prediction_error_ratio'] = direct_prediction_error
        temporal_requested = _requires_temporal_pose(
          True,
          direct_prediction_error,
          prediction_visible_fraction,
          temporal_config,
          result['tracking'].get('mask_area_ratio'),
        )
      else:
        temporal_requested = _requires_temporal_pose(
          False,
          None,
          prediction_visible_fraction,
          temporal_config,
          result['tracking'].get('mask_area_ratio'),
        )

    temporal_failure_reason = None
    if temporal_requested:
      temporal_age = previous_temporal_age + 1
      max_temporal_frames = int(temporal_config.get('max_estimated_frames', 4))
      if temporal_age > max_temporal_frames:
        temporal_failure_reason = 'TEMPORAL_ESTIMATE_EXPIRED'
      else:
        predicted_roi = (np.asarray(predicted_vertices_full, dtype=np.float64) - roi_origin) * scale
        height, width = ffs_outputs['disp'].shape
        margin = float(temporal_config.get('max_vertex_outside_roi_px', 24.0))
        inside_roi = (
          np.all(predicted_roi[:, 0] >= -margin)
          and np.all(predicted_roi[:, 0] <= width - 1 + margin)
          and np.all(predicted_roi[:, 1] >= -margin)
          and np.all(predicted_roi[:, 1] <= height - 1 + margin)
        )
        if inside_roi:
          temporal_metrics = dict(direct_pose.metrics)
          temporal_metrics.update({
            'pose_source': 'TEMPORAL',
            'temporal_age': temporal_age,
            'direct_invalid_reason': direct_pose.invalid_reason,
            'direct_prediction_error_ratio': direct_prediction_error,
          })
          temporal_pose = estimate_pose_from_vertices(
            predicted_roi,
            plane,
            ffs_outputs['K'],
            self.rectifier.baseline_m,
            pose_config,
            previous_rotation=previous_rotation,
            metrics=temporal_metrics,
            stereo_geometry=stereo_geometry,
          )
          if temporal_pose.valid:
            selected_pose = temporal_pose
            pose_source = 'TEMPORAL'
          else:
            temporal_failure_reason = temporal_pose.invalid_reason
        else:
          temporal_failure_reason = 'POSE_JUMP_REJECTED'

    result['timings_ms']['pose_ms'] = (time.perf_counter() - stage) * 1000.0
    if pose_source == 'INVALID':
      reason = temporal_failure_reason or direct_pose.invalid_reason
      if reason == 'TEMPORAL_ESTIMATE_EXPIRED':
        self.reset_tracking()
      return self._finish(result, started, reason)
    if temporal_requested and pose_source != 'TEMPORAL':
      reason = temporal_failure_reason or 'POSE_JUMP_REJECTED'
      if reason == 'TEMPORAL_ESTIMATE_EXPIRED':
        self.reset_tracking()
      return self._finish(result, started, reason)

    pose = selected_pose
    result['plate_valid'] = True
    result['pose_source'] = pose_source
    result['temporal_age'] = temporal_age if pose_source == 'TEMPORAL' else 0
    result['triangle_metrics'] = pose.metrics
    result['plate_origin_camera_m'] = pose.t
    result['plate_rotation_camera'] = pose.R
    result['plate_vertices_camera_m'] = pose.vertices_xyz
    full_vertices = pose.vertices_uv / scale + roi_origin
    result['plate_vertices_left_uv'] = full_vertices
    if pose_source == 'DIRECT':
      visibility_factor = result['tracking'].get('prediction_visible_fraction')
      if visibility_factor is None:
        visibility_factor = pose.metrics.get('triangle_visible_fraction', 1.0)
      pose_confidence = result['quality_score'] * float(np.clip(
        pose.metrics.get('triangle_iou', 0.0), 0.0, 1.0,
      )) * float(np.clip(visibility_factor, 0.0, 1.0)) * float(np.clip(
        pose.metrics.get('geometry_score', 1.0), 0.0, 1.0,
      ))
      temporal_age = 0
    else:
      max_temporal_frames = max(1, int(temporal_config.get('max_estimated_frames', 4)))
      age_factor = max(0.0, 1.0 - (temporal_age - 1) / max_temporal_frames)
      tracking_factor = float(np.clip(result['tracking'].get('inlier_ratio', 0.0), 0.0, 1.0))
      pose_confidence = (
        result['quality_score']
        * float(temporal_config.get('confidence_scale', 0.60))
        * age_factor
        * tracking_factor
      )
    result['pose_confidence'] = float(np.clip(pose_confidence, 0.0, 1.0))
    result['quality_score'] = result['pose_confidence']
    if pose_source == 'DIRECT':
      self._update_tracking(frame, left_rect, target_mask, full_vertices, pose.R, temporal_age)
    else:
      # Keep the last reliable direct frame as the optical-flow anchor.  Updating
      # it with an occluded estimate allows small mask errors to accumulate.
      self._update_temporal_pose(pose.R, temporal_age)

    if not getattr(self, 'screw_enabled', True):
      return self._finish(result, started)

    screw = locate_screw(
      pose.R,
      pose.t,
      self.screw_offset,
      self.rectifier.K,
      (self.rectifier.calibration.image_width, self.rectifier.calibration.image_height),
    )
    result['screw_xyz_camera_m'] = screw['xyz_camera_m']
    result['screw_valid'] = screw['valid']
    result['screw_u_rect'] = screw['u']
    result['screw_v_rect'] = screw['v']
    screw_reason = screw['invalid_reason']
    if screw['valid']:
      raw_uv = self.rectifier.left_rectified_to_raw([[screw['u'], screw['v']]])[0]
      raw_valid = (
        np.isfinite(raw_uv).all()
        and 0 <= raw_uv[0] < self.rectifier.calibration.image_width
        and 0 <= raw_uv[1] < self.rectifier.calibration.image_height
      )
      if raw_valid:
        result['screw_u_raw'] = float(raw_uv[0])
        result['screw_v_raw'] = float(raw_uv[1])
      else:
        result['screw_valid'] = False
        screw_reason = 'PROJECTION_OUT_OF_IMAGE'
    if (debug or save_preview) and output_dir is not None:
      Path(output_dir).mkdir(parents=True, exist_ok=True)
      preview = left_rect.copy()
      cv2.polylines(preview, [np.round(full_vertices).astype(np.int32)], True, (0, 255, 0), 2)
      if result['screw_valid']:
        cv2.drawMarker(preview, (round(screw['u']), round(screw['v'])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
      cv2.imwrite(str(Path(output_dir) / 'locator_preview.jpg'), preview)
    return self._finish(result, started, screw_reason)
