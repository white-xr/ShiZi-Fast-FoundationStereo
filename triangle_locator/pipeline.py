import json
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from scripts import run_demo, seg_pointcloud
from .calibration import adjust_intrinsics_for_roi, measure_vertical_epipolar_error
from .plate_pose import estimate_plate_pose
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
  'POSE_DISCONTINUITY',
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
    'screw_u': None,
    'screw_v': None,
    'screw_xyz_camera_m': None,
    'plate_origin_camera_m': None,
    'plate_rotation_camera': None,
    'plate_vertices_camera_m': None,
    'plate_vertices_left_uv': None,
    'plane': None,
    'valid_ratio': 0.0,
    'inlier_ratio': 0.0,
    'plane_rmse_px': None,
    'depth_rmse_mm': None,
    'spatial_coverage': 0.0,
    'quality_score': 0.0,
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
    'invalid_reason': None,
  }


class TriangleLocator:
  def __init__(self, model, model_args, rectifier, config, repo_dir):
    self.model = model
    self.model_args = model_args
    self.rectifier = rectifier
    self.config = config
    self.repo_dir = Path(repo_dir)
    offset_path = (config.get('screw') or {}).get('offset_file')
    if offset_path and not Path(offset_path).is_absolute():
      offset_path = self.repo_dir / offset_path
    self.screw_offset = load_screw_offset(offset_path)
    self.previous_rotation = None

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
    synchronization = self.config.get('synchronization') or {}
    delta = frame.get('timestamp_delta_ms')
    if delta is None or float(delta) > float(synchronization.get('max_delta_ms', 2.0)):
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

    segmentation = self.config.get('segmentation') or {}
    stage = time.perf_counter()
    masks = seg_pointcloud.run_yolo_seg_array(
      left_rect,
      output_dir,
      segmentation,
      self.repo_dir,
      write_debug=bool(debug and segmentation.get('debug_vis', True)),
    )
    result['timings_ms']['yolo_ms'] = (time.perf_counter() - stage) * 1000.0
    result['plate_confidence'] = masks['plate_confidence']
    result['detection_valid'] = masks['detection_valid']
    if not masks['detection_valid']:
      return self._finish(result, started, masks['invalid_reason'])

    roi_config = self.config.get('roi') or {}
    bbox = seg_pointcloud.target_roi(
      masks[seg_pointcloud.TARGET_CLASS],
      max_disp=float(self.model_args.max_disp) / float(self.model_args.scale),
      context_padding_px=int(roi_config.get('context_padding_px', 32)),
      padding_px=int(roi_config.get('padding_px', 0)),
      min_size_px=int(roi_config.get('min_size_px', 160)),
    )
    if bbox is None:
      return self._finish(result, started, 'NO_PLATE')
    x0, y0, x1, y1 = bbox
    left_crop_bgr = left_rect[y0:y1, x0:x1]
    right_crop_bgr = right_rect[y0:y1, x0:x1]
    target_crop = masks[seg_pointcloud.TARGET_CLASS][y0:y1, x0:x1]
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
    if target_crop.shape != ffs_outputs['disp'].shape:
      target_crop = cv2.resize(
        target_crop.astype(np.uint8),
        (ffs_outputs['disp'].shape[1], ffs_outputs['disp'].shape[0]),
        interpolation=cv2.INTER_NEAREST,
      ).astype(bool)

    stage = time.perf_counter()
    plane_config = dict(self.config.get('plane') or {})
    plane_config.setdefault('max_disp', float(self.model_args.max_disp))
    plane = fit_disparity_plane(
      ffs_outputs['disp'], target_crop, ffs_outputs['K'], self.rectifier.baseline_m, plane_config,
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
    pose = estimate_plate_pose(
      target_crop,
      plane,
      ffs_outputs['K'],
      self.rectifier.baseline_m,
      self.config.get('pose') or {},
      previous_rotation=self.previous_rotation,
    )
    result['timings_ms']['pose_ms'] = (time.perf_counter() - stage) * 1000.0
    if not pose.valid:
      return self._finish(result, started, pose.invalid_reason)
    self.previous_rotation = pose.R.copy()
    result['plate_valid'] = True
    result['plate_origin_camera_m'] = pose.t
    result['plate_rotation_camera'] = pose.R
    result['plate_vertices_camera_m'] = pose.vertices_xyz
    full_vertices = pose.vertices_uv / float(ffs_outputs['scale']) + np.array([x0, y0], dtype=np.float64)
    result['plate_vertices_left_uv'] = full_vertices
    result['quality_score'] *= float(np.clip(pose.metrics['triangle_iou'], 0.0, 1.0))

    screw = locate_screw(
      pose.R,
      pose.t,
      self.screw_offset,
      self.rectifier.K,
      (self.rectifier.calibration.image_width, self.rectifier.calibration.image_height),
    )
    result['screw_valid'] = screw['valid']
    result['screw_xyz_camera_m'] = screw['xyz_camera_m']
    result['screw_u'] = screw['u']
    result['screw_v'] = screw['v']
    if (debug or save_preview) and output_dir is not None:
      Path(output_dir).mkdir(parents=True, exist_ok=True)
      preview = left_rect.copy()
      cv2.polylines(preview, [np.round(full_vertices).astype(np.int32)], True, (0, 255, 0), 2)
      if screw['valid']:
        cv2.drawMarker(preview, (round(screw['u']), round(screw['v'])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
      cv2.imwrite(str(Path(output_dir) / 'locator_preview.jpg'), preview)
    return self._finish(result, started, screw['invalid_reason'])
