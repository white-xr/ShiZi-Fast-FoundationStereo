#!/usr/bin/env python3
import argparse
import logging
import os
import socket
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException


code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from scripts import run_config, run_demo
from triangle_locator.calibration import StereoRectifier
from triangle_locator.pipeline import TriangleLocator


VERTEX_NAMES = ('A / base-left', 'B / base-right', 'C / tip')
VERTEX_COLORS = ((40, 190, 255), (255, 170, 40), (220, 80, 220))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description='Browser GUI for offline triangle 3D pose inspection.')
  parser.add_argument('--config', default='configs/offline_triangle_locator.yaml')
  parser.add_argument('--all-images', action='store_true', help='show every matched pair instead of configured batch ids')
  parser.add_argument('--host', default='0.0.0.0')
  parser.add_argument('--port', type=int, default=7860)
  return parser.parse_args(argv)


def _project(K, point):
  point = np.asarray(point, dtype=np.float64).reshape(3)
  if not np.isfinite(point).all() or point[2] <= 0:
    return None
  return np.array([
    K[0, 0] * point[0] / point[2] + K[0, 2],
    K[1, 1] * point[1] / point[2] + K[1, 2],
  ])


def _draw_label(image, text, point, color):
  anchor = tuple(np.round(point).astype(int) + np.array([7, -7]))
  cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 20, 24), 4, cv2.LINE_AA)
  cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _empty_frame_state(name):
  return {
    'name': name,
    'inferred': False,
    'plate_valid': False,
    'pose_source': 'INVALID',
    'pose_confidence': 0.0,
    'temporal_age': 0,
    'invalid_reason': '',
    'quality_score': 0.0,
    'plane_rmse_px': None,
    'inlier_ratio': 0.0,
    'valid_ratio': 0.0,
    'spatial_coverage': 0.0,
    'rectification': None,
    'vertices_camera_mm': None,
    'vertices_rect_uv': None,
    'vertices_raw_uv': None,
    'candidate_vertices_rect_uv': None,
    'origin_camera_mm': None,
    'origin_raw_uv': None,
    'rotation_camera': None,
    'triangle_metrics': None,
    'tracking': None,
  }


class PoseWebSession:
  def __init__(self, args):
    self.config_path = run_config.resolve_path(args.config)
    raw_config = run_config.load_yaml(self.config_path)
    self.config = run_config.apply_auto_dataset_paths(raw_config)

    pair_config = self.config
    if bool(getattr(args, 'all_images', False)) and pair_config.get('batch'):
      pair_config = dict(self.config)
      batch = dict(pair_config.get('batch') or {})
      for key in ('ids', 'start', 'end', 'limit'):
        batch.pop(key, None)
      pair_config['batch'] = batch
    self.pairs = run_config.collect_pairs(pair_config)
    if not self.pairs:
      raise RuntimeError('配置的数据目录中没有找到同名的左右图像对')
    self.pairs_by_name = {pair['name']: pair for pair in self.pairs}
    self.timestamps = run_config.load_pair_timestamps(self.config)

    calibration_path = self.config.get('stereo_calibration_file')
    if not calibration_path:
      raise RuntimeError('配置缺少 stereo_calibration_file')
    self.rectifier = StereoRectifier.from_file(
      run_config.resolve_path(calibration_path),
      alpha=float((self.config.get('rectification') or {}).get('alpha', 0.0)),
      enabled=bool((self.config.get('rectification') or {}).get('enabled', True)),
    )
    self.axis_length_m = float((self.config.get('visualization') or {}).get('axis_length_m', 0.03))
    self.records = {}
    self.yolo_masks = {}
    self.raw_cache = OrderedDict()
    self.rectified_cache = OrderedDict()
    self.model = None
    self.model_args = None
    self.locator = None
    self.lock = threading.RLock()
    self.inference_lock = threading.Lock()

  def pair(self, name):
    pair = self.pairs_by_name.get(str(name))
    if pair is None:
      raise KeyError(f'未知图像：{name}')
    return pair

  @staticmethod
  def _cache(cache, name, image):
    cache[name] = image
    cache.move_to_end(name)
    while len(cache) > 8:
      cache.popitem(last=False)

  def images(self, name):
    name = str(name)
    with self.lock:
      raw = self.raw_cache.get(name)
      rectified = self.rectified_cache.get(name)
      if raw is not None and rectified is not None:
        return raw.copy(), rectified.copy()
    pair = self.pair(name)
    left = cv2.imread(str(pair['left_file']), cv2.IMREAD_COLOR)
    right = cv2.imread(str(pair['right_file']), cv2.IMREAD_COLOR)
    if left is None or right is None:
      raise FileNotFoundError(f'无法读取左右图：{name}')
    left_rect, _ = self.rectifier.rectify(left, right)
    with self.lock:
      self._cache(self.raw_cache, name, left)
      self._cache(self.rectified_cache, name, left_rect)
    return left.copy(), left_rect.copy()

  def ensure_locator(self):
    if self.locator is not None:
      return
    config = run_config.apply_runtime_safety(
      self.config,
      require_gpu=True,
    )
    config = run_config.apply_auto_dataset_paths(config)
    runtime_config = config.get('runtime') or {}
    run_demo.configure_runtime(
      disable_torch_compile=bool(runtime_config.get('disable_torch_compile', False)),
    )
    model_args = run_demo.load_args(run_config.runtime_overrides(config))
    model_args.show = 0
    model_args.intrinsic_file = None
    model = run_demo.load_model(model_args)
    locator = TriangleLocator(model, model_args, self.rectifier, config, repo_dir)
    with self.lock:
      self.config = config
      self.model_args = model_args
      self.model = model
      self.locator = locator

  def load_dataset(self, data_dir):
    if not str(data_dir or '').strip():
      raise ValueError('请输入数据集路径')
    path = Path(str(data_dir).strip()).expanduser()
    if not path.is_absolute():
      path = repo_dir / path
    path = path.resolve()
    if not path.is_dir():
      raise FileNotFoundError(f'找不到数据集目录：{path}')

    config = dict(self.config)
    config['data_dir'] = str(path)
    config.pop('dataset_dir', None)
    config.pop('single', None)
    config.pop('pairs', None)
    batch = dict(config.get('batch') or {})
    for key in ('ids', 'start', 'end', 'limit', 'left_dir', 'right_dir', 'data_dir', 'dataset_dir'):
      batch.pop(key, None)
    batch.setdefault('pattern', '*.png')
    config['batch'] = batch
    config = run_config.apply_auto_dataset_paths(config)
    pairs = run_config.collect_pairs(config)
    if not pairs:
      raise ValueError(f'数据集中没有找到左右同名图像：{path}')

    if not self.inference_lock.acquire(blocking=False):
      raise RuntimeError('姿态推理正在运行，请完成后再切换数据集')
    try:
      with self.lock:
        self.config = config
        self.pairs = pairs
        self.pairs_by_name = {pair['name']: pair for pair in pairs}
        self.timestamps = run_config.load_pair_timestamps(config)
        self.records.clear()
        self.yolo_masks.clear()
        self.raw_cache.clear()
        self.rectified_cache.clear()
        if self.locator is not None:
          self.locator.config = config
          self.locator.reset_tracking()
    finally:
      self.inference_lock.release()
    return self.public_state()

  def infer(self, name):
    pair = self.pair(name)
    if not self.inference_lock.acquire(blocking=False):
      raise RuntimeError('已有一张图正在恢复姿态，请等待完成')
    try:
      left = cv2.imread(str(pair['left_file']), cv2.IMREAD_COLOR)
      right = cv2.imread(str(pair['right_file']), cv2.IMREAD_COLOR)
      if left is None or right is None:
        raise FileNotFoundError(f'无法读取左右图：{name}')
      left_rect, right_rect = self.rectifier.rectify(left, right)
      self.ensure_locator()
      metadata = self.timestamps.get(pair['name'], {})
      pair_index = next(index for index, item in enumerate(self.pairs) if item['name'] == name)
      numeric_id = int(name) if str(name).isdigit() else pair_index + 1
      frame = {
        'frame_id': numeric_id,
        'left_frame_id': numeric_id,
        'right_frame_id': numeric_id,
        'left_bgr': left,
        'right_bgr': right,
        **metadata,
      }
      result = self.locator.process(frame, output_dir=None, debug=False)
      with self.lock:
        self.records[name] = result
        target_mask = self.locator.last_target_mask
        if target_mask is not None:
          self.yolo_masks[name] = (
            target_mask.copy(),
            self.locator.last_target_mask_coordinate,
          )
        else:
          self.yolo_masks.pop(name, None)
        self._cache(self.raw_cache, name, left)
        self._cache(self.rectified_cache, name, left_rect)
      return self.frame_state(name)
    finally:
      self.inference_lock.release()

  def frame_state(self, name):
    self.pair(name)
    result = self.records.get(name)
    if result is None:
      return _empty_frame_state(name)
    vertices_xyz = result.get('plate_vertices_camera_m')
    origin = result.get('plate_origin_camera_m')
    rotation = result.get('plate_rotation_camera')
    vertices_rect_uv = result.get('plate_vertices_left_uv')
    vertices_raw_uv = None
    if vertices_rect_uv is not None:
      vertices_raw_uv = self.rectifier.left_rectified_to_raw(vertices_rect_uv)
    origin_raw_uv = None
    if origin is not None:
      origin_rect_uv = _project(self.rectifier.K, origin)
      if origin_rect_uv is not None:
        origin_raw_uv = self.rectifier.left_rectified_to_raw([origin_rect_uv])[0]
    return {
      'name': name,
      'inferred': True,
      'plate_valid': bool(result.get('plate_valid')),
      'pose_source': result.get('pose_source') or 'INVALID',
      'pose_confidence': float(result.get('pose_confidence') or 0.0),
      'temporal_age': int(result.get('temporal_age') or 0),
      'invalid_reason': result.get('invalid_reason') or '',
      'quality_score': float(result.get('quality_score') or 0.0),
      'plane_rmse_px': result.get('plane_rmse_px'),
      'inlier_ratio': float(result.get('inlier_ratio') or 0.0),
      'valid_ratio': float(result.get('valid_ratio') or 0.0),
      'spatial_coverage': float(result.get('spatial_coverage') or 0.0),
      'rectification': result.get('rectification'),
      'vertices_camera_mm': (np.asarray(vertices_xyz) * 1000.0).tolist() if vertices_xyz is not None else None,
      'vertices_rect_uv': vertices_rect_uv,
      'vertices_raw_uv': vertices_raw_uv.tolist() if vertices_raw_uv is not None else None,
      'candidate_vertices_rect_uv': result.get('candidate_vertices_left_uv'),
      'origin_camera_mm': (np.asarray(origin) * 1000.0).tolist() if origin is not None else None,
      'origin_raw_uv': origin_raw_uv.tolist() if origin_raw_uv is not None else None,
      'rotation_camera': rotation,
      'triangle_metrics': result.get('triangle_metrics'),
      'tracking': result.get('tracking'),
    }

  def public_state(self):
    selected = self.pairs[0]['name']
    return {
      'images': [pair['name'] for pair in self.pairs],
      'selected': selected,
      'config_file': str(self.config_path),
      'data_dir': str(run_config.get_data_dir(self.config) or ''),
      'valid_iters': int(self.config.get('valid_iters', 0) or 0),
      'max_disp': int(self.config.get('max_disp', 0) or 0),
      'image_width': self.rectifier.calibration.image_width,
      'image_height': self.rectifier.calibration.image_height,
      'axis_length_mm': self.axis_length_m * 1000.0,
      'vertex_names': VERTEX_NAMES,
    }

  def render_jpeg(self, name, view='raw_pose'):
    raw, rectified = self.images(name)
    raw_display = view == 'raw_pose'
    image = raw if raw_display else rectified
    result = self.records.get(name)
    if result is not None:
      candidate_rect = result.get('candidate_vertices_left_uv')
      show_candidate = candidate_rect is not None and (
        not result.get('plate_valid') or result.get('pose_source') == 'TEMPORAL'
      )
      if show_candidate:
        candidate_rect = np.asarray(candidate_rect, dtype=np.float64)
        candidate = (
          self.rectifier.left_rectified_to_raw(candidate_rect)
          if raw_display else candidate_rect
        )
        cv2.polylines(
          image,
          [np.round(candidate).astype(np.int32)],
          True,
          (60, 60, 225),
          2,
          cv2.LINE_AA,
        )
        _draw_label(image, 'MASK CANDIDATE', candidate[0], (60, 60, 225))
    if result is not None and result.get('plate_valid'):
      pose_source = result.get('pose_source') or 'DIRECT'
      outline_color = (40, 175, 255) if pose_source == 'TEMPORAL' else (60, 220, 90)
      vertices_rect = np.asarray(result['plate_vertices_left_uv'], dtype=np.float64)
      vertices = self.rectifier.left_rectified_to_raw(vertices_rect) if raw_display else vertices_rect
      cv2.polylines(image, [np.round(vertices).astype(np.int32)], True, outline_color, 3, cv2.LINE_AA)
      for index, (point, color) in enumerate(zip(vertices, VERTEX_COLORS)):
        center = tuple(np.round(point).astype(int))
        cv2.circle(image, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, 10, (20, 24, 28), 2, cv2.LINE_AA)
        _draw_label(image, VERTEX_NAMES[index].split(' ')[0], point, color)
      _draw_label(image, pose_source, vertices[0] + np.array([0.0, -18.0]), outline_color)

      origin_xyz = np.asarray(result['plate_origin_camera_m'], dtype=np.float64)
      origin_rect_uv = _project(self.rectifier.K, origin_xyz)
      if origin_rect_uv is not None:
        origin_uv = (
          self.rectifier.left_rectified_to_raw([origin_rect_uv])[0]
          if raw_display else origin_rect_uv
        )
        origin_point = tuple(np.round(origin_uv).astype(int))
        cv2.drawMarker(image, origin_point, (0, 220, 255), cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)
        _draw_label(image, 'O', origin_uv, (0, 220, 255))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
      raise RuntimeError('图像编码失败')
    return encoded.tobytes()

  def render_yolo_jpeg(self, name):
    self.pair(name)
    with self.lock:
      entry = self.yolo_masks.get(name)
      if entry is not None:
        mask = np.asarray(entry[0], dtype=bool).copy()
      else:
        mask = None
    if mask is None:
      raw, _ = self.images(name)
      image = np.zeros_like(raw)
    else:
      image = np.zeros((*mask.shape, 3), dtype=np.uint8)
      image[mask] = (40, 220, 100)
      contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
      )
      cv2.drawContours(image, contours, -1, (170, 255, 220), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
      raise RuntimeError('YOLO 分割图编码失败')
    return encoded.tobytes()


def create_app(session):
  app = Flask(__name__, template_folder=str(code_dir / 'templates'))

  @app.errorhandler(Exception)
  def handle_error(exc):
    if isinstance(exc, HTTPException):
      return jsonify({'ok': False, 'error': exc.description}), exc.code
    logging.exception('Pose Web GUI request failed')
    status = 404 if isinstance(exc, KeyError) else 400 if isinstance(exc, ValueError) else 500
    return jsonify({'ok': False, 'error': str(exc)}), status

  @app.get('/')
  def index():
    return render_template('screw_offset_web.html')

  @app.get('/favicon.ico')
  def favicon():
    return Response(status=204)

  @app.get('/api/state')
  def state():
    return jsonify({'ok': True, **session.public_state()})

  @app.post('/api/dataset')
  def dataset():
    payload = request.get_json(silent=True) or {}
    return jsonify({'ok': True, **session.load_dataset(payload.get('data_dir'))})

  @app.get('/api/frame/<name>')
  def frame_state(name):
    return jsonify({'ok': True, **session.frame_state(name)})

  @app.get('/api/image/<name>.jpg')
  def image(name):
    view = request.args.get('view', 'raw_pose')
    if view not in {'raw_pose', 'rectified'}:
      raise ValueError(f'未知图像视图：{view}')
    return Response(
      session.render_jpeg(name, view=view),
      mimetype='image/jpeg',
      headers={'Cache-Control': 'no-store'},
    )

  @app.get('/api/image/<name>.yolo.jpg')
  def yolo_image(name):
    return Response(
      session.render_yolo_jpeg(name),
      mimetype='image/jpeg',
      headers={'Cache-Control': 'no-store'},
    )

  @app.post('/api/infer/<name>')
  def infer(name):
    return jsonify({'ok': True, **session.infer(name)})

  return app


def local_ip():
  ssh_connection = os.environ.get('SSH_CONNECTION', '').split()
  if len(ssh_connection) >= 3:
    return ssh_connection[2]
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    sock.connect(('8.8.8.8', 80))
    return sock.getsockname()[0]
  except OSError:
    return '127.0.0.1'
  finally:
    sock.close()


def main(argv=None):
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = parse_args(argv)
  session = PoseWebSession(args)
  app = create_app(session)
  logging.info('Open on Windows: http://%s:%d', local_ip(), args.port)
  app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
  main()
