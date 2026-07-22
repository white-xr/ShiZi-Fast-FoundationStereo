import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from Utils import o3d, toOpen3dCloud


TARGET_CLASS = 'target_plate'
TIP_CLASS = 'screwdriver_tip'
_MODEL_CACHE = {}


def _require_ultralytics():
  try:
    from ultralytics import YOLO
  except ImportError as exc:
    raise RuntimeError(
      '缺少 ultralytics，无法运行 YOLO-Seg。请在当前 ffs 环境安装 ultralytics 后重试。'
    ) from exc
  return YOLO


def _resolve_path(value, base_dir):
  path = Path(str(value)).expanduser()
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def _class_name(names, cls_id):
  if isinstance(names, dict):
    return str(names.get(int(cls_id), int(cls_id)))
  if isinstance(names, (list, tuple)) and int(cls_id) < len(names):
    return str(names[int(cls_id)])
  return str(int(cls_id))


def _area_threshold(config, class_name):
  by_class = config.get('min_area_by_class') or {}
  if class_name in by_class:
    return int(by_class[class_name])
  return int(config.get('min_area_px', 0) or 0)


def _morph(mask, open_kernel=0, close_kernel=0):
  out = mask.astype(np.uint8)
  if open_kernel and open_kernel > 1:
    kernel = np.ones((int(open_kernel), int(open_kernel)), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
  if close_kernel and close_kernel > 1:
    kernel = np.ones((int(close_kernel), int(close_kernel)), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
  return out.astype(bool)


def _save_mask(path, mask):
  imageio.imwrite(path, (mask.astype(np.uint8) * 255))


def _overlay_mask(image, mask, color, alpha=0.45):
  if not mask.any():
    return image
  out = image.copy()
  color_arr = np.array(color, dtype=np.uint8)
  out[mask] = (out[mask].astype(np.float32) * (1 - alpha) + color_arr.astype(np.float32) * alpha).astype(np.uint8)
  return out


def _write_cloud(path, points, colors):
  path = Path(path)
  if len(points) == 0:
    path.write_text(
      'ply\n'
      'format ascii 1.0\n'
      'element vertex 0\n'
      'property float x\n'
      'property float y\n'
      'property float z\n'
      'property uchar red\n'
      'property uchar green\n'
      'property uchar blue\n'
      'end_header\n',
      encoding='ascii',
    )
    return

  if o3d is None:
    raise RuntimeError('缺少 open3d，无法写 ply 点云。')
  cloud = toOpen3dCloud(points, colors)
  o3d.io.write_point_cloud(str(path), cloud)


def write_empty_cropped_clouds(output_dir):
  output_dir = Path(output_dir)
  _write_cloud(output_dir / 'target_tip_cloud.ply', np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))


def select_target_candidate(candidates, image_shape, config):
  height, width = image_shape[:2]
  image_area = float(height * width)
  min_area = _area_threshold(config, TARGET_CLASS)
  min_ratio = float(config.get('min_area_ratio', 0.0) or 0.0)
  max_ratio = float(config.get('max_area_ratio', 1.0) or 1.0)
  min_aspect = float(config.get('min_bbox_aspect', 0.15) or 0.15)
  max_aspect = float(config.get('max_bbox_aspect', 6.0) or 6.0)
  filtered = []
  for candidate in candidates:
    mask = np.asarray(candidate['mask'], dtype=bool)
    area = int(mask.sum())
    ratio = area / image_area
    ys, xs = np.where(mask)
    if area < min_area or ratio < min_ratio or ratio > max_ratio or len(xs) == 0:
      continue
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    aspect = box_width / max(box_height, 1)
    if aspect < min_aspect or aspect > max_aspect:
      continue
    item = dict(candidate)
    item.update({'area': area, 'area_ratio': ratio, 'bbox_aspect': aspect})
    filtered.append(item)
  filtered.sort(key=lambda item: (float(item['confidence']), item['area']), reverse=True)
  if not filtered:
    return None, 'NO_PLATE', filtered
  if len(filtered) > 1:
    best, second = filtered[:2]
    confidence_close = float(second['confidence']) >= (
      float(best['confidence']) - float(config.get('ambiguity_conf_delta', 0.05))
    )
    area_close = second['area'] >= best['area'] * float(config.get('ambiguity_min_area_ratio', 0.5))
    if confidence_close and area_close:
      return None, 'MULTIPLE_PLATES', filtered
  return filtered[0], None, filtered


def run_yolo_seg_array(left_bgr, output_dir, config, repo_dir, write_debug=False):
  YOLO = _require_ultralytics()

  output_dir = Path(output_dir) if output_dir is not None else None
  model_path = _resolve_path(config.get('model_path', 'weights/triangle-metal.pt'), repo_dir)
  if not model_path.exists():
    raise FileNotFoundError(f'找不到 YOLO-Seg 模型：{model_path}')

  left_bgr = np.asarray(left_bgr)[..., :3]
  height, width = left_bgr.shape[:2]
  target_mask = np.zeros((height, width), dtype=bool)
  tip_mask = np.zeros((height, width), dtype=bool)
  target_candidates = []
  tip_candidates = []

  cache_key = str(model_path)
  if cache_key not in _MODEL_CACHE:
    loaded = YOLO(cache_key)
    if str(getattr(loaded, 'task', '')).lower() != 'segment':
      raise RuntimeError(f'YOLO model is not a segmentation model: {model_path}')
    model_names = getattr(loaded, 'names', {})
    name_values = model_names.values() if isinstance(model_names, dict) else model_names
    if TARGET_CLASS not in {str(value) for value in name_values}:
      raise RuntimeError(f'YOLO model does not contain class={TARGET_CLASS}: {model_path}')
    _MODEL_CACHE[cache_key] = loaded
  model = _MODEL_CACHE[cache_key]
  predict_kwargs = {
    'source': left_bgr,
    'imgsz': int(config.get('imgsz', 960)),
    'conf': float(config.get('conf_thres', 0.35)),
    'verbose': False,
  }
  device = str(config.get('device', '') or '').strip()
  if device and device.lower() not in {'auto', 'none', 'null'}:
    predict_kwargs['device'] = device
  results = model.predict(**predict_kwargs)
  if not results:
    logging.warning('YOLO-Seg did not return a result')
  else:
    result = results[0]
    masks = result.masks
    boxes = result.boxes
    if masks is not None and boxes is not None:
      mask_data = masks.data.detach().cpu().numpy()
      cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
      confs = boxes.conf.detach().cpu().numpy()
      names = result.names
      for idx, raw_mask in enumerate(mask_data):
        class_name = _class_name(names, cls_ids[idx])
        if class_name not in {TARGET_CLASS, TIP_CLASS}:
          continue
        if float(confs[idx]) < float(config.get('conf_thres', 0.35)):
          continue

        resized = cv2.resize(raw_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        min_area = _area_threshold(config, class_name)
        if int(resized.sum()) < min_area:
          logging.info(
            '跳过过小 mask：class=%s, area=%d, min_area=%d',
            class_name,
            int(resized.sum()),
            min_area,
          )
          continue

        candidate = {
          'mask': resized,
          'confidence': float(confs[idx]),
          'class_name': class_name,
        }
        if class_name == TARGET_CLASS:
          target_candidates.append(candidate)
        elif class_name == TIP_CLASS:
          tip_candidates.append(candidate)

  selected, invalid_reason, filtered_targets = select_target_candidate(
    target_candidates, left_bgr.shape, config,
  )
  if selected is not None:
    target_mask = selected['mask']
  if tip_candidates:
    tip_mask = max(tip_candidates, key=lambda item: item['confidence'])['mask']

  open_kernel = int(config.get('morph_open_kernel', 0) or 0)
  close_kernel = int(config.get('morph_close_kernel', 0) or 0)
  target_mask = _morph(target_mask, open_kernel, close_kernel)
  tip_mask = _morph(tip_mask, open_kernel, close_kernel)

  if output_dir is not None and write_debug:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_mask(output_dir / 'target_plate_mask.png', target_mask)
    _save_mask(output_dir / 'screwdriver_tip_mask.png', tip_mask)

  if output_dir is not None and write_debug and bool(config.get('debug_vis', True)):
    debug = left_bgr.copy()
    if bool(config.get('draw_masks', True)):
      if target_mask.any():
        debug = _overlay_mask(debug, target_mask, color=(0, 255, 0))
        if bool(config.get('draw_tip_when_target_exists', False)):
          debug = _overlay_mask(debug, tip_mask, color=(0, 128, 255))
      else:
        debug = _overlay_mask(debug, tip_mask, color=(0, 128, 255))
    cv2.imwrite(str(output_dir / 'debug_vis.png'), debug)

  logging.info(
    'YOLO result: target_area=%d, screwdriver_tip_area=%d, candidates=%d, reason=%s',
    int(target_mask.sum()),
    int(tip_mask.sum()),
    len(filtered_targets),
    invalid_reason,
  )
  return {
    TARGET_CLASS: target_mask,
    TIP_CLASS: tip_mask,
    'left_bgr': left_bgr,
    'left_rgb': cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB),
    'detection_valid': selected is not None,
    'invalid_reason': invalid_reason,
    'plate_confidence': float(selected['confidence']) if selected is not None else 0.0,
    'target_candidates': filtered_targets,
  }


def run_yolo_seg(left_image_path, output_dir, config, repo_dir):
  left_bgr = cv2.imread(str(left_image_path), cv2.IMREAD_COLOR)
  if left_bgr is None:
    raise FileNotFoundError(f'cannot read left image: {left_image_path}')
  return run_yolo_seg_array(left_bgr, output_dir, config, repo_dir, write_debug=True)


def _mask_for_xyz(mask, xyz_shape):
  height, width = xyz_shape[:2]
  if mask.shape == (height, width):
    return mask.astype(bool)
  resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
  return resized.astype(bool)


def union_bbox(masks, padding_px, min_size_px=64):
  union = masks[TARGET_CLASS]
  if not union.any():
    return None

  height, width = union.shape
  ys, xs = np.where(union)
  pad = int(padding_px or 0)
  x0 = max(0, int(xs.min()) - pad)
  y0 = max(0, int(ys.min()) - pad)
  x1 = min(width, int(xs.max()) + pad + 1)
  y1 = min(height, int(ys.max()) + pad + 1)

  min_size = int(min_size_px or 0)
  if min_size > 0:
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < min_size:
      extra = min_size - box_w
      x0 = max(0, x0 - extra // 2)
      x1 = min(width, x1 + extra - extra // 2)
    if box_h < min_size:
      extra = min_size - box_h
      y0 = max(0, y0 - extra // 2)
      y1 = min(height, y1 + extra - extra // 2)

  return x0, y0, x1, y1


def target_roi(mask, max_disp, context_padding_px=32, padding_px=0, min_size_px=64):
  mask = np.asarray(mask, dtype=bool)
  if not mask.any():
    return None
  height, width = mask.shape
  ys, xs = np.where(mask)
  context = int(context_padding_px or 0)
  pad = int(padding_px or 0)
  x0 = max(0, int(xs.min()) - int(np.ceil(max_disp)) - context - pad)
  y0 = max(0, int(ys.min()) - context - pad)
  x1 = min(width, int(xs.max()) + context + pad + 1)
  y1 = min(height, int(ys.max()) + context + pad + 1)
  min_size = int(min_size_px or 0)
  if x1 - x0 < min_size:
    extra = min_size - (x1 - x0)
    x0 = max(0, x0 - extra // 2)
    x1 = min(width, x1 + extra - extra // 2)
  if y1 - y0 < min_size:
    extra = min_size - (y1 - y0)
    y0 = max(0, y0 - extra // 2)
    y1 = min(height, y1 + extra - extra // 2)
  return x0, y0, x1, y1


def crop_image(image, bbox):
  x0, y0, x1, y1 = bbox
  return image[y0:y1, x0:x1]


def crop_masks(masks, bbox):
  return {
    TARGET_CLASS: crop_image(masks[TARGET_CLASS], bbox),
    TIP_CLASS: crop_image(masks[TIP_CLASS], bbox),
  }


def crop_clouds_from_masks(outputs, masks, output_dir, config):
  if o3d is None:
    raise RuntimeError('缺少 open3d，无法输出 target_plate_cloud.ply / screwdriver_tip_cloud.ply。')

  xyz_map = outputs.get('xyz_map')
  depth = outputs.get('depth')
  left_rgb = outputs.get('left_rgb')
  if xyz_map is None or depth is None or left_rgb is None:
    raise RuntimeError('FFS 没有返回 xyz_map/depth/left_rgb，无法按 mask 裁点云。')

  min_depth = float(config.get('min_depth', 0.0) or 0.0)
  max_depth_value = config.get('max_depth', None)
  max_depth = None if max_depth_value in {None, '', 'none', 'None'} else float(max_depth_value)

  if left_rgb.shape[:2] != xyz_map.shape[:2]:
    left_rgb_for_cloud = cv2.resize(left_rgb, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_LINEAR)
  else:
    left_rgb_for_cloud = left_rgb

  valid_depth = np.isfinite(depth) & (depth > 0)
  if min_depth > 0:
    valid_depth &= depth >= min_depth
  if max_depth is not None and max_depth > 0:
    valid_depth &= depth <= max_depth

  valid_xyz = np.isfinite(xyz_map).all(axis=2) & (xyz_map[:, :, 2] > 0)
  base_valid = valid_depth & valid_xyz

  outputs_spec = [
    (TARGET_CLASS, 'target_plate_cloud.ply'),
    (TIP_CLASS, 'screwdriver_tip_cloud.ply'),
  ]
  class_masks = {}
  save_individual = bool(config.get('save_individual_object_clouds', False))
  for class_name, filename in outputs_spec:
    mask = _mask_for_xyz(masks[class_name], xyz_map.shape)
    class_masks[class_name] = mask
    if save_individual:
      keep = mask & base_valid
      points = xyz_map[keep]
      colors = left_rgb_for_cloud[keep]
      _write_cloud(Path(output_dir) / filename, points, colors)
      logging.info('%s saved: points=%d', filename, len(points))

  if bool(config.get('save_merged_object_cloud', True)):
    merged_filename = str(config.get('merged_object_cloud_file', 'target_tip_cloud.ply'))
    merged_mask = (class_masks[TARGET_CLASS] | class_masks[TIP_CLASS]) & base_valid
    merged_points = xyz_map[merged_mask]
    merged_colors = left_rgb_for_cloud[merged_mask]
    _write_cloud(Path(output_dir) / merged_filename, merged_points, merged_colors)
    logging.info('%s saved: points=%d', merged_filename, len(merged_points))


def process_segmentation_pointcloud(left_image_path, ffs_outputs, output_dir, config, repo_dir):
  if not config or not config.get('enabled', False):
    return
  masks = run_yolo_seg(left_image_path, output_dir, config, repo_dir)
  crop_clouds_from_masks(ffs_outputs, masks, output_dir, config)
