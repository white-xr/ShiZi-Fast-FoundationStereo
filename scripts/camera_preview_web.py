#!/usr/bin/env python3
import argparse
import atexit
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template


code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from scripts import run_config, run_demo
from scripts.run_camera import CppOrbbecStereoSource, rectifier_for_source
from triangle_locator.pipeline import TriangleLocator


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description='Web preview for Gemini 305 Dual RGB.')
  parser.add_argument('--config', default='configs/realtime_triangle_locator.yaml')
  parser.add_argument('--host', default='0.0.0.0')
  parser.add_argument('--port', type=int, default=7861)
  return parser.parse_args(argv)


class CameraPreview:
  def __init__(self, config):
    preview_config = run_config.apply_runtime_safety(dict(config), require_gpu=True)
    camera = dict(preview_config.get('camera') or {})
    camera['query_sdk_calibration'] = False
    preview_config['camera'] = camera
    self.source = CppOrbbecStereoSource(preview_config)
    self.rectifier = rectifier_for_source(self.source, preview_config)
    runtime = preview_config.get('runtime') or {}
    run_demo.configure_runtime(
      disable_torch_compile=bool(runtime.get('disable_torch_compile', False)),
    )
    model_args = run_demo.load_args(run_config.runtime_overrides(preview_config))
    model_args.show = 0
    model_args.intrinsic_file = None
    model = run_demo.load_model(model_args)
    self.locator = TriangleLocator(model, model_args, self.rectifier, preview_config, repo_dir)
    self.condition = threading.Condition()
    self.frame = None
    self.capture_sequence = 0
    self.processed_sequence = 0
    self.captured = 0
    self.processed = 0
    self.jpeg = None
    self.state = None
    self.capture_error = None
    self.inference_error = None
    self.closed = False
    self.capture_thread = threading.Thread(target=self._capture, name='camera-web-capture', daemon=True)
    self.inference_thread = threading.Thread(target=self._infer, name='camera-web-inference', daemon=True)
    self.capture_thread.start()
    self.inference_thread.start()

  def _capture(self):
    try:
      while True:
        with self.condition:
          if self.closed:
            return
        frame = self.source.read()
        with self.condition:
          self.frame = frame
          self.capture_sequence += 1
          self.captured += 1
          self.condition.notify_all()
    except Exception as exc:
      with self.condition:
        if not self.closed:
          self.capture_error = str(exc)
        self.condition.notify_all()

  @staticmethod
  def _draw_label(image, text, point, color):
    anchor = tuple(np.round(point).astype(int) + np.array([7, -7]))
    cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 20, 24), 4, cv2.LINE_AA)
    cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

  def _render(self, frame, result, target_mask):
    left = frame['left_bgr'].copy()
    right = frame['right_bgr'].copy()
    if target_mask is not None and target_mask.shape == left.shape[:2]:
      mask = np.asarray(target_mask, dtype=bool)
      tint = np.zeros_like(left)
      tint[mask] = (35, 210, 90)
      left = cv2.addWeighted(left, 1.0, tint, 0.34, 0.0)
      contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      cv2.drawContours(left, contours, -1, (140, 255, 185), 2, cv2.LINE_AA)

    vertices_raw = None
    vertices_camera = result.get('plate_vertices_camera_m')
    if result.get('plate_valid') and result.get('plate_vertices_left_uv') is not None:
      vertices_rect = np.asarray(result['plate_vertices_left_uv'], dtype=np.float64)
      vertices_raw = self.rectifier.left_rectified_to_raw(vertices_rect)
      cv2.polylines(left, [np.round(vertices_raw).astype(np.int32)], True, (50, 230, 90), 3, cv2.LINE_AA)
      colors = ((40, 190, 255), (255, 170, 40), (220, 80, 220))
      xyz_mm = np.asarray(vertices_camera, dtype=np.float64) * 1000.0
      for index, (point, color) in enumerate(zip(vertices_raw, colors)):
        center = tuple(np.round(point).astype(int))
        cv2.circle(left, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(left, center, 10, (20, 24, 28), 2, cv2.LINE_AA)
        self._draw_label(left, 'ABC'[index], point, color)
      c_pixel = vertices_raw[2]
      c_text = f'C: u={c_pixel[0]:.1f}  v={c_pixel[1]:.1f}  depth={xyz_mm[2, 2]:.1f} mm'
      (text_width, text_height), _ = cv2.getTextSize(
        c_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
      )
      cv2.rectangle(left, (12, 12), (28 + text_width, 28 + text_height), (20, 28, 32), -1)
      cv2.putText(left, c_text, (20, 22 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 235, 240), 2, cv2.LINE_AA)
    else:
      reason = result.get('invalid_reason') or 'WAITING'
      cv2.putText(left, reason, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 220), 3, cv2.LINE_AA)

    combined = cv2.hconcat([left, right])
    ok, encoded = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
      raise RuntimeError('Dual RGB inference preview JPEG encoding failed')
    state = {
      'ok': True,
      'display_frame_id': int(frame['frame_id']),
      'left_frame_id': int(frame['left_frame_id']),
      'right_frame_id': int(frame['right_frame_id']),
      'timestamp_delta_ms': float(frame['timestamp_delta_ms']),
      'width': int(left.shape[1]),
      'height': int(left.shape[0]),
      'plate_valid': bool(result.get('plate_valid')),
      'pose_source': result.get('pose_source') or 'INVALID',
      'quality_score': float(result.get('quality_score') or 0.0),
      'invalid_reason': result.get('invalid_reason') or '',
      'vertices_raw_uv': vertices_raw.tolist() if vertices_raw is not None else None,
      'vertices_camera_mm': (
        (np.asarray(vertices_camera, dtype=np.float64) * 1000.0).tolist()
        if vertices_camera is not None else None
      ),
      'timings_ms': result.get('timings_ms') or {},
    }
    return encoded.tobytes(), state

  def _infer(self):
    last_sequence = 0
    try:
      while True:
        with self.condition:
          self.condition.wait_for(
            lambda: self.closed or self.capture_error is not None or self.capture_sequence > last_sequence,
            timeout=3.0,
          )
          if self.closed or self.capture_error is not None:
            return
          if self.capture_sequence <= last_sequence:
            continue
          sequence = self.capture_sequence
          frame = dict(self.frame)
        result = self.locator.process(frame, output_dir=None, debug=False, save_preview=False)
        target_mask = self.locator.last_target_mask
        if target_mask is not None and self.locator.last_target_mask_coordinate != 'raw_left':
          target_mask = None
        jpeg, state = self._render(frame, result, target_mask)
        with self.condition:
          last_sequence = sequence
          self.processed_sequence = sequence
          self.processed += 1
          state.update({
            'captured': self.captured,
            'processed': self.processed,
            'dropped': max(0, self.captured - self.processed - 1),
          })
          self.jpeg = jpeg
          self.state = state
          self.condition.notify_all()
    except Exception as exc:
      with self.condition:
        if not self.closed:
          self.inference_error = str(exc)
        self.condition.notify_all()

  def wait(self, timeout=3.0):
    with self.condition:
      if self.jpeg is None and self.capture_error is None and self.inference_error is None:
        self.condition.wait(timeout=timeout)
      error = self.capture_error or self.inference_error
      if error is not None:
        raise RuntimeError(error)
      if self.jpeg is None:
        raise RuntimeError('camera preview timed out')
      return self.jpeg, dict(self.state)

  def close(self):
    with self.condition:
      if self.closed:
        return
      self.closed = True
      self.condition.notify_all()
    self.source.release()
    self.capture_thread.join(timeout=2.0)
    self.inference_thread.join(timeout=3.0)


def create_app(preview):
  app = Flask(__name__)

  @app.get('/')
  def index():
    return render_template('camera_preview_web.html')

  @app.get('/api/frame.jpg')
  def frame_jpeg():
    jpeg, _ = preview.wait()
    return Response(jpeg, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

  @app.get('/api/state')
  def state():
    _, state_value = preview.wait()
    return jsonify(state_value)

  return app


def main(argv=None):
  args = parse_args(argv)
  config = run_config.load_yaml(run_config.resolve_path(args.config))
  preview = CameraPreview(config)
  atexit.register(preview.close)
  app = create_app(preview)
  print(f'Open on Windows: http://192.168.1.57:{args.port}', flush=True)
  try:
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
  finally:
    preview.close()


if __name__ == '__main__':
  main()
