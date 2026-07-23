#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import sys
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from scripts import run_config, run_demo
from triangle_locator.calibration import StereoRectifier
from triangle_locator.pipeline import TriangleLocator
from triangle_locator.screw_locator import (
  calibrate_offset_from_observations,
  load_screw_offset,
  locate_screw,
  save_screw_offset_mm,
)


AXIS_COLORS = ((40, 40, 230), (40, 190, 40), (220, 100, 30))


def configure_local_display():
  if os.environ.get('DISPLAY'):
    return False
  socket_dir = Path('/tmp/.X11-unix')
  if not socket_dir.is_dir():
    return False
  sockets = sorted(
    path for path in socket_dir.glob('X*')
    if path.name[1:].isdigit() and path.stat().st_uid == os.getuid()
  )
  if not sockets:
    return False
  display_number = sockets[0].name[1:]
  runtime_dir = Path(os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}')
  auth_candidates = [runtime_dir / 'gdm' / 'Xauthority', Path.home() / '.Xauthority']
  authority = next(
    (path for path in auth_candidates if path.is_file() and path.stat().st_uid == os.getuid()),
    None,
  )
  if authority is None:
    return False
  os.environ['DISPLAY'] = f':{display_number}'
  os.environ['XAUTHORITY'] = str(authority)
  logging.info('Detected local desktop display %s', os.environ['DISPLAY'])
  return True


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description='Visually tune the plate-local screw-cap offset in millimeters.')
  parser.add_argument('--config', default='configs/offline_triangle_locator.yaml')
  parser.add_argument('--offset-file', default=None, help='override screw.offset_file from the locator config')
  parser.add_argument(
    '--configured-only',
    action='store_true',
    help='show only batch.ids/start/end/limit selected by the YAML; default is every paired image',
  )
  parser.add_argument('--range-mm', type=float, default=300.0, help='slider range in both directions')
  return parser.parse_args(argv)


def _project(K, point):
  point = np.asarray(point, dtype=np.float64).reshape(3)
  if not np.isfinite(point).all() or point[2] <= 0:
    return None
  return np.array([
    K[0, 0] * point[0] / point[2] + K[0, 2],
    K[1, 1] * point[1] / point[2] + K[1, 2],
  ])


def _draw_cross(image, uv, color, size=14, thickness=2):
  if uv is None:
    return
  point = tuple(np.round(uv).astype(int))
  cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, size, thickness, cv2.LINE_AA)


class ScrewOffsetGui:
  def __init__(self, root, args):
    self.root = root
    self.args = args
    self.config_path = run_config.resolve_path(args.config)
    raw_config = run_config.load_yaml(self.config_path)
    self.config = run_config.apply_auto_dataset_paths(raw_config)

    pair_config = copy.deepcopy(self.config)
    if not args.configured_only and pair_config.get('batch'):
      batch = dict(pair_config['batch'])
      for key in ('ids', 'start', 'end', 'limit'):
        batch.pop(key, None)
      pair_config['batch'] = batch
    self.pairs = run_config.collect_pairs(pair_config)
    if not self.pairs:
      raise RuntimeError('配置的数据目录中没有找到同名的左右图像对')
    self.timestamps = run_config.load_pair_timestamps(self.config)

    calibration_path = self.config.get('stereo_calibration_file')
    if not calibration_path:
      raise RuntimeError('配置缺少 stereo_calibration_file')
    self.rectifier = StereoRectifier.from_file(
      run_config.resolve_path(calibration_path),
      alpha=float((self.config.get('rectification') or {}).get('alpha', 0.0)),
    )
    configured_offset = args.offset_file or (self.config.get('screw') or {}).get('offset_file')
    if not configured_offset:
      raise RuntimeError('配置缺少 screw.offset_file')
    self.offset_path = run_config.resolve_path(configured_offset)

    self.locator = None
    self.model = None
    self.model_args = None
    self.records = {}
    self.annotations = {}
    self.current_index = 0
    self.base_bgr = None
    self.photo = None
    self.zoom_photo = None
    self.display_transform = None
    self.busy = False
    self.gpu_run_confirmed = False
    self.suppress_offset_events = False

    self.status_var = tk.StringVar(value='选择图像后点击“运行定位”')
    self.metrics_var = tk.StringVar(value='尚无标注')
    self.pixel_var = tk.StringVar(value='像素：-')
    self.offset_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
    self.pair_var = tk.StringVar()

    self._build_ui()
    self._load_saved_offset(show_message=False)
    self._select_initial_pair(raw_config)
    self.root.after(20, self._show_selected_image)

  def _build_ui(self):
    self.root.title('三角板螺丝帽偏移标定（毫米）')
    self.root.minsize(1120, 760)
    self.root.geometry('1380x880')

    toolbar = ttk.Frame(self.root, padding=(10, 8))
    toolbar.pack(fill=tk.X)
    ttk.Label(toolbar, text='离线图像').pack(side=tk.LEFT)
    self.pair_combo = ttk.Combobox(
      toolbar,
      textvariable=self.pair_var,
      values=[pair['name'] for pair in self.pairs],
      state='readonly',
      width=22,
    )
    self.pair_combo.pack(side=tk.LEFT, padx=(8, 6))
    self.pair_combo.bind('<<ComboboxSelected>>', lambda _event: self._pair_changed())
    self.prev_button = ttk.Button(toolbar, text='上一张', command=lambda: self._step_pair(-1))
    self.prev_button.pack(side=tk.LEFT, padx=3)
    self.next_button = ttk.Button(toolbar, text='下一张', command=lambda: self._step_pair(1))
    self.next_button.pack(side=tk.LEFT, padx=3)
    self.run_button = ttk.Button(toolbar, text='运行当前图定位', command=self._run_current)
    self.run_button.pack(side=tk.LEFT, padx=(12, 3))
    ttk.Label(toolbar, text=f'共 {len(self.pairs)} 对').pack(side=tk.LEFT, padx=10)

    body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
    body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    image_frame = ttk.Frame(body)
    controls = ttk.Frame(body, padding=(12, 4))
    body.add(image_frame, weight=4)
    body.add(controls, weight=1)

    self.canvas = tk.Canvas(image_frame, background='#16191d', highlightthickness=0, cursor='crosshair')
    self.canvas.pack(fill=tk.BOTH, expand=True)
    self.canvas.bind('<Configure>', lambda _event: self._render())
    self.canvas.bind('<Button-1>', self._canvas_clicked)
    ttk.Label(image_frame, textvariable=self.pixel_var, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    ttk.Label(controls, text='三角板局部坐标偏移', font=('', 13, 'bold')).pack(anchor=tk.W)
    ttk.Label(
      controls,
      text='X/Y 沿板面坐标轴，Z 沿板面法向。界面和文件均使用毫米。',
      wraplength=300,
      justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(3, 10))

    names = (('dx', '局部 X'), ('dy', '局部 Y'), ('dz', '局部 Z'))
    slider_range = abs(float(self.args.range_mm))
    for index, (short_name, description) in enumerate(names):
      row = ttk.Frame(controls)
      row.pack(fill=tk.X, pady=(2, 8))
      ttk.Label(row, text=f'{short_name}  {description}', width=14).pack(side=tk.LEFT)
      spinbox = ttk.Spinbox(
        row,
        from_=-slider_range,
        to=slider_range,
        increment=0.1,
        textvariable=self.offset_vars[index],
        width=10,
        command=self._offset_changed,
      )
      spinbox.pack(side=tk.RIGHT)
      spinbox.bind('<KeyRelease>', lambda _event: self._offset_changed())
      spinbox.bind('<FocusOut>', lambda _event: self._offset_changed())
      scale = tk.Scale(
        controls,
        from_=-slider_range,
        to=slider_range,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        showvalue=False,
        variable=self.offset_vars[index],
        command=lambda _value: self._offset_changed(),
        highlightthickness=0,
      )
      scale.pack(fill=tk.X, pady=(0, 3))

    action_row = ttk.Frame(controls)
    action_row.pack(fill=tk.X, pady=(5, 8))
    ttk.Button(action_row, text='保存毫米参数', command=self._save_offset).pack(side=tk.LEFT)
    ttk.Button(action_row, text='恢复已保存', command=self._load_saved_offset).pack(side=tk.LEFT, padx=6)

    ttk.Separator(controls).pack(fill=tk.X, pady=8)
    ttk.Label(controls, text='真实螺丝帽中心', font=('', 11, 'bold')).pack(anchor=tk.W)
    ttk.Label(
      controls,
      text='在左图点击真实中心。至少标注并定位 3 张图后，可自动拟合固定偏移。',
      wraplength=300,
      justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(3, 7))
    annotation_row = ttk.Frame(controls)
    annotation_row.pack(fill=tk.X)
    ttk.Button(annotation_row, text='自动拟合', command=self._auto_fit).pack(side=tk.LEFT)
    ttk.Button(annotation_row, text='清除当前标注', command=self._clear_annotation).pack(side=tk.LEFT, padx=6)
    ttk.Label(controls, textvariable=self.metrics_var, wraplength=300, justify=tk.LEFT).pack(
      anchor=tk.W, pady=(7, 8), fill=tk.X,
    )

    ttk.Label(controls, text='局部放大', font=('', 11, 'bold')).pack(anchor=tk.W, pady=(3, 3))
    self.zoom_canvas = tk.Canvas(controls, width=290, height=210, background='#16191d', highlightthickness=0)
    self.zoom_canvas.pack(fill=tk.X)

    ttk.Separator(controls).pack(fill=tk.X, pady=8)
    ttk.Label(controls, textvariable=self.status_var, wraplength=300, justify=tk.LEFT).pack(
      anchor=tk.W, fill=tk.X,
    )

  def _select_initial_pair(self, raw_config):
    requested = ((raw_config.get('batch') or {}).get('ids') or [])
    requested = str(requested[0]) if requested else None
    names = [pair['name'] for pair in self.pairs]
    if requested in names:
      self.current_index = names.index(requested)
    self.pair_var.set(self.pairs[self.current_index]['name'])

  def _current_pair(self):
    return self.pairs[self.current_index]

  def _pair_changed(self):
    names = [pair['name'] for pair in self.pairs]
    try:
      self.current_index = names.index(self.pair_var.get())
    except ValueError:
      return
    self._show_selected_image()

  def _step_pair(self, delta):
    if self.busy:
      return
    self.current_index = (self.current_index + delta) % len(self.pairs)
    self.pair_var.set(self._current_pair()['name'])
    self._show_selected_image()

  def _show_selected_image(self):
    pair = self._current_pair()
    record = self.records.get(pair['name'])
    if record is not None:
      self.base_bgr = record['left_rect'].copy()
      self.status_var.set(self._result_status(record['result']))
      self._render()
      return
    left = cv2.imread(str(pair['left_file']), cv2.IMREAD_COLOR)
    right = cv2.imread(str(pair['right_file']), cv2.IMREAD_COLOR)
    if left is None or right is None:
      self.status_var.set(f'读取失败：{pair["name"]}')
      return
    try:
      self.base_bgr, _ = self.rectifier.rectify(left, right)
    except (ValueError, cv2.error) as exc:
      self.base_bgr = left
      self.status_var.set(f'校正失败：{exc}')
    else:
      self.status_var.set('图像已校正；尚未运行当前图定位')
    self._render()

  def _set_busy(self, busy):
    self.busy = busy
    state = tk.DISABLED if busy else tk.NORMAL
    self.run_button.configure(state=state)
    self.prev_button.configure(state=state)
    self.next_button.configure(state=state)
    self.pair_combo.configure(state='disabled' if busy else 'readonly')

  def _ensure_locator(self):
    if self.locator is not None:
      return
    config = run_config.apply_runtime_safety(
      run_config.load_yaml(self.config_path),
      require_gpu=True,
    )
    config = run_config.apply_auto_dataset_paths(config)
    model_args = run_demo.load_args(run_config.runtime_overrides(config))
    model_args.show = 0
    model_args.intrinsic_file = None
    model = run_demo.load_model(model_args)
    locator = TriangleLocator(model, model_args, self.rectifier, config, repo_dir)
    self.config = config
    self.model_args = model_args
    self.model = model
    self.locator = locator

  def _run_current(self):
    if self.busy:
      return
    if not self.gpu_run_confirmed:
      valid_iters = int(self.config.get('valid_iters', 0) or 0)
      max_disp = int(self.config.get('max_disp', 0) or 0)
      confirmed = messagebox.askyesno(
        '确认启动 GPU 推理',
        f'即将加载 FFS 并运行 GPU 推理。\n\n'
        f'valid_iters={valid_iters}，max_disp={max_disp}\n\n'
        '这台机器以前出现过推理期间整机重启。请确认 BIOS/供电/显卡状态稳定后再继续。',
      )
      if not confirmed:
        return
      self.gpu_run_confirmed = True
    pair = self._current_pair().copy()
    self._set_busy(True)
    self.status_var.set('正在加载模型并定位，首次运行会较慢...')

    def worker():
      try:
        self._ensure_locator()
        left = cv2.imread(str(pair['left_file']), cv2.IMREAD_COLOR)
        right = cv2.imread(str(pair['right_file']), cv2.IMREAD_COLOR)
        if left is None or right is None:
          raise FileNotFoundError(f'无法读取左右图：{pair["name"]}')
        left_rect, _ = self.rectifier.rectify(left, right)
        metadata = self.timestamps.get(pair['name'], {})
        numeric_id = int(pair['name']) if str(pair['name']).isdigit() else self.current_index + 1
        frame = {
          'frame_id': numeric_id,
          'left_frame_id': numeric_id,
          'right_frame_id': numeric_id,
          'left_bgr': left,
          'right_bgr': right,
          **metadata,
        }
        self.locator.previous_rotation = None
        result = self.locator.process(frame, output_dir=None, debug=False)
      except Exception as exc:
        details = traceback.format_exc()
        self.root.after(0, lambda error=exc, trace=details: self._inference_failed(error, trace))
        return
      self.root.after(
        0,
        lambda pair_name=pair['name'], image=left_rect, output=result:
          self._inference_finished(pair_name, image, output),
      )

    threading.Thread(target=worker, name='screw-offset-inference', daemon=True).start()

  def _inference_failed(self, exc, details):
    self._set_busy(False)
    self.status_var.set(f'定位失败：{exc}')
    logging.error('GUI localization failed:\n%s', details)
    messagebox.showerror('定位失败', str(exc))

  def _inference_finished(self, name, left_rect, result):
    self.records[name] = {'left_rect': left_rect, 'result': result}
    self._set_busy(False)
    if self._current_pair()['name'] == name:
      self.base_bgr = left_rect.copy()
      self.status_var.set(self._result_status(result))
    self._render()

  @staticmethod
  def _result_status(result):
    if result.get('plate_valid'):
      return (
        f'三角板姿态有效；质量 {float(result.get("quality_score", 0.0)):.3f}，'
        f'平面 RMSE {float(result.get("plane_rmse_px") or 0.0):.3f} px'
      )
    return f'三角板姿态无效：{result.get("invalid_reason") or "未知原因"}'

  def _offset_mm(self):
    try:
      return np.array([value.get() for value in self.offset_vars], dtype=np.float64)
    except (tk.TclError, ValueError):
      return None

  def _set_offset_mm(self, offset):
    self.suppress_offset_events = True
    try:
      for variable, value in zip(self.offset_vars, np.asarray(offset).reshape(3)):
        variable.set(round(float(value), 4))
    finally:
      self.suppress_offset_events = False
    self._offset_changed()

  def _offset_changed(self):
    if self.suppress_offset_events:
      return
    self._render()

  def _prediction(self, result):
    offset_mm = self._offset_mm()
    if offset_mm is None or not result.get('plate_valid'):
      return None
    return locate_screw(
      result['plate_rotation_camera'],
      result['plate_origin_camera_m'],
      offset_mm / 1000.0,
      self.rectifier.K,
      self.rectifier.calibration.image_size,
    )

  def _draw_pose(self, image, result):
    vertices = result.get('plate_vertices_left_uv')
    if vertices is not None:
      cv2.polylines(image, [np.round(vertices).astype(np.int32)], True, (30, 220, 50), 2, cv2.LINE_AA)
    if not result.get('plate_valid'):
      return
    R = np.asarray(result['plate_rotation_camera'], dtype=np.float64)
    t = np.asarray(result['plate_origin_camera_m'], dtype=np.float64)
    origin_uv = _project(self.rectifier.K, t)
    _draw_cross(image, origin_uv, (0, 220, 255), 12, 2)
    for index, color in enumerate(AXIS_COLORS):
      end_uv = _project(self.rectifier.K, t + R[:, index] * 0.02)
      if origin_uv is not None and end_uv is not None:
        cv2.arrowedLine(
          image,
          tuple(np.round(origin_uv).astype(int)),
          tuple(np.round(end_uv).astype(int)),
          color,
          2,
          cv2.LINE_AA,
          tipLength=0.2,
        )

  def _render(self):
    if self.base_bgr is None or self.root.winfo_exists() == 0:
      return
    image = self.base_bgr.copy()
    name = self._current_pair()['name']
    record = self.records.get(name)
    prediction = None
    if record is not None:
      result = record['result']
      self._draw_pose(image, result)
      prediction = self._prediction(result)
      if prediction is not None and prediction['valid']:
        predicted_uv = np.array([prediction['u'], prediction['v']])
        _draw_cross(image, predicted_uv, (20, 20, 240), 22, 3)
        cv2.circle(image, tuple(np.round(predicted_uv).astype(int)), 9, (20, 20, 240), 2, cv2.LINE_AA)

    annotated_uv = self.annotations.get(name)
    if annotated_uv is not None:
      _draw_cross(image, annotated_uv, (255, 220, 0), 22, 3)
      if prediction is not None and prediction['valid']:
        predicted_uv = np.array([prediction['u'], prediction['v']])
        cv2.line(
          image,
          tuple(np.round(predicted_uv).astype(int)),
          tuple(np.round(annotated_uv).astype(int)),
          (255, 220, 0),
          1,
          cv2.LINE_AA,
        )

    canvas_width = max(self.canvas.winfo_width(), 2)
    canvas_height = max(self.canvas.winfo_height(), 2)
    height, width = image.shape[:2]
    scale = min(canvas_width / width, canvas_height / height)
    draw_width = max(1, int(round(width * scale)))
    draw_height = max(1, int(round(height * scale)))
    offset_x = (canvas_width - draw_width) // 2
    offset_y = (canvas_height - draw_height) // 2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(rgb).resize((draw_width, draw_height), Image.Resampling.LANCZOS)
    self.photo = ImageTk.PhotoImage(resized)
    self.canvas.delete('all')
    self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor=tk.NW)
    self.display_transform = (scale, offset_x, offset_y, width, height)
    self._render_zoom(image, prediction, annotated_uv)
    self._update_metrics()

  def _render_zoom(self, image, prediction, annotated_uv):
    center = annotated_uv
    if center is None and prediction is not None and prediction['valid']:
      center = np.array([prediction['u'], prediction['v']])
    self.zoom_canvas.delete('all')
    if center is None:
      self.zoom_canvas.create_text(145, 105, text='运行定位或点击图像后显示', fill='#d0d4d8')
      return
    half = 45
    x, y = np.round(center).astype(int)
    x0, x1 = max(0, x - half), min(image.shape[1], x + half)
    y0, y1 = max(0, y - half), min(image.shape[0], y + half)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
      return
    target_width = max(self.zoom_canvas.winfo_width(), 290)
    target_height = max(self.zoom_canvas.winfo_height(), 210)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    zoomed = Image.fromarray(rgb).resize((target_width, target_height), Image.Resampling.NEAREST)
    self.zoom_photo = ImageTk.PhotoImage(zoomed)
    self.zoom_canvas.create_image(0, 0, image=self.zoom_photo, anchor=tk.NW)

  def _canvas_clicked(self, event):
    if self.display_transform is None:
      return
    scale, offset_x, offset_y, width, height = self.display_transform
    u = (event.x - offset_x) / scale
    v = (event.y - offset_y) / scale
    if not (0 <= u < width and 0 <= v < height):
      return
    name = self._current_pair()['name']
    self.annotations[name] = np.array([u, v], dtype=np.float64)
    self.pixel_var.set(f'真实中心：u={u:.2f}, v={v:.2f}')
    self._render()

  def _clear_annotation(self):
    name = self._current_pair()['name']
    self.annotations.pop(name, None)
    self.pixel_var.set('像素：-')
    self._render()

  def _annotated_observations(self):
    observations = []
    for name, uv in self.annotations.items():
      record = self.records.get(name)
      if record is None or not record['result'].get('plate_valid'):
        continue
      result = record['result']
      observations.append({
        'R': result['plate_rotation_camera'],
        't': result['plate_origin_camera_m'],
        'K': self.rectifier.K,
        'uv': uv,
      })
    return observations

  def _update_metrics(self):
    observations = self._annotated_observations()
    errors = []
    offset_mm = self._offset_mm()
    if offset_mm is not None:
      for item in observations:
        predicted = locate_screw(
          item['R'], item['t'], offset_mm / 1000.0, item['K'], self.rectifier.calibration.image_size,
        )
        if predicted['valid']:
          errors.append(float(np.linalg.norm(np.array([predicted['u'], predicted['v']]) - item['uv'])))
    annotated_count = len(self.annotations)
    usable_count = len(observations)
    if errors:
      rmse = float(np.sqrt(np.mean(np.square(errors))))
      current = self._current_pair()['name']
      current_error = None
      if current in self.annotations and current in self.records:
        prediction = self._prediction(self.records[current]['result'])
        if prediction is not None and prediction['valid']:
          current_error = float(np.linalg.norm(
            np.array([prediction['u'], prediction['v']]) - self.annotations[current]
          ))
      current_text = f'，当前误差 {current_error:.2f} px' if current_error is not None else ''
      self.metrics_var.set(
        f'已点击 {annotated_count} 张，可用 {usable_count} 张；RMSE {rmse:.2f} px{current_text}'
      )
    else:
      self.metrics_var.set(f'已点击 {annotated_count} 张，可用定位结果 {usable_count} 张')

  def _auto_fit(self):
    observations = self._annotated_observations()
    if len(observations) < 3:
      messagebox.showinfo('标注不足', '请至少运行并点击 3 张图中的真实螺丝帽中心。')
      return
    try:
      initial = self._offset_mm()
      offset_m, result = calibrate_offset_from_observations(
        observations,
        initial=None if initial is None else initial / 1000.0,
      )
    except Exception as exc:
      messagebox.showerror('自动拟合失败', str(exc))
      return
    self._set_offset_mm(offset_m * 1000.0)
    inliers = int(np.count_nonzero(result.inlier_mask))
    self.status_var.set(f'自动拟合完成：使用 {len(observations)} 张，其中 {inliers} 张为内点；尚未保存')

  def _load_saved_offset(self, show_message=True):
    try:
      offset_m = load_screw_offset(self.offset_path)
    except (ValueError, OSError) as exc:
      if show_message:
        messagebox.showerror('读取失败', str(exc))
      return
    if offset_m is None:
      self._set_offset_mm([0.0, 0.0, 0.0])
      if show_message:
        self.status_var.set('偏移文件尚未配置，当前从 0 mm 开始调节')
      return
    self._set_offset_mm(offset_m * 1000.0)
    if show_message:
      self.status_var.set(f'已恢复：{self.offset_path}')

  def _save_offset(self):
    offset_mm = self._offset_mm()
    if offset_mm is None:
      messagebox.showerror('参数无效', 'dx/dy/dz 必须是有效数字。')
      return
    try:
      save_screw_offset_mm(
        self.offset_path,
        offset_mm,
        source='offline screw offset GUI manual/annotation tuning',
        metadata={'annotated_views': len(self._annotated_observations())},
      )
    except (ValueError, OSError) as exc:
      messagebox.showerror('保存失败', str(exc))
      return
    if self.locator is not None:
      self.locator.screw_offset = offset_mm / 1000.0
    self.status_var.set(f'已保存毫米参数：{self.offset_path}')
    messagebox.showinfo('保存完成', f'已保存到\n{self.offset_path}')


def main(argv=None):
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = parse_args(argv)
  configure_local_display()
  try:
    root = tk.Tk()
  except tk.TclError as exc:
    raise SystemExit(
      '无法连接图形桌面。请在本机桌面终端运行，或通过 SSH X11 转发运行；'
      f'当前 DISPLAY={os.environ.get("DISPLAY")!r}。原始错误：{exc}'
    ) from exc
  try:
    ScrewOffsetGui(root, args)
  except Exception as exc:
    root.withdraw()
    messagebox.showerror('启动失败', str(exc))
    root.destroy()
    raise
  root.mainloop()


if __name__ == '__main__':
  main()
