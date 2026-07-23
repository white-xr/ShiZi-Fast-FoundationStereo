import argparse
from collections import deque
import logging
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from scripts import run_config, run_demo
from triangle_locator.calibration import (
  StereoRectifier,
  load_stereo_calibration,
  stereo_calibration_from_sdk_profiles,
)
from triangle_locator.pipeline import TriangleLocator, append_result_jsonl, write_result_json


ORBBEC_PYTHON_DIR = Path('/home/depthai/venv/enpower/vision-model/pyorbbecsdk/install_286/lib')
ORBBEC_SDK_DIR = Path('/home/depthai/OrbbecSDK_v2/OrbbecSDK_v2.8.6_202604271452_6399409_linux_x86_64')
ORBBEC_STREAM_CPP = code_dir / 'orbbec_dual_color_stream.cpp'
ORBBEC_STREAM_BIN = repo_dir / 'build' / 'orbbec_probe' / 'orbbec_dual_color_stream'
ORBBEC_PACKET_HEADER = struct.Struct('<4s5I7Q')


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description='Run FoundationStereo from a live stereo camera.')
  parser.add_argument('--config', default='configs/stereo_infer.yaml', help='YAML config path')
  parser.add_argument('--width', type=int, default=None, help='Dual RGB frame width')
  parser.add_argument('--height', type=int, default=None, help='capture height')
  parser.add_argument('--fps', type=int, default=None, help='requested camera FPS')
  parser.add_argument('--interval_sec', type=float, default=None, help='sleep seconds after each inference')
  parser.add_argument('--max_frames', type=int, default=None, help='stop after N frames; 0 means run until Ctrl+C')
  parser.add_argument('--save_frames', action='store_true', help='save each frame to a numbered output dir instead of overwriting camera_latest')
  parser.add_argument('--preview', action='store_true', help='show live left/right preview; press q to stop')
  parser.add_argument('--out_dir', default=None, help='output directory; default is config out_dir/camera_latest')
  return parser.parse_args(argv)


def camera_cfg(config, key, default=None):
  camera = config.get('camera') or {}
  return camera.get(key, default)


def cli_or_config(cli_value, config, key, default=None):
  return cli_value if cli_value is not None else camera_cfg(config, key, default)


def load_orbbec_sdk():
  if ORBBEC_PYTHON_DIR.exists():
    sys.path.insert(0, str(ORBBEC_PYTHON_DIR))
    os.environ['LD_LIBRARY_PATH'] = f'{ORBBEC_PYTHON_DIR}:{os.environ.get("LD_LIBRARY_PATH", "")}'
  try:
    import pyorbbecsdk as ob
  except ImportError as exc:
    raise RuntimeError(
      '找不到 pyorbbecsdk。已经尝试加载 '
      f'{ORBBEC_PYTHON_DIR}，请先编译或安装 Orbbec Python SDK。'
    ) from exc
  return ob


def ob_format_by_name(ob, name):
  if not name:
    return None
  return getattr(ob.OBFormat, str(name).upper())


def frame_to_bgr(frame, ob):
  frame = frame.as_video_frame()
  width = frame.get_width()
  height = frame.get_height()
  fmt = frame.get_format()
  data = np.asanyarray(frame.get_data())

  if fmt == ob.OBFormat.BGR:
    return np.resize(data, (height, width, 3)).copy()
  if fmt == ob.OBFormat.RGB:
    image = np.resize(data, (height, width, 3))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if fmt == ob.OBFormat.YUYV:
    image = np.resize(data, (height, width, 2))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
  if fmt == ob.OBFormat.UYVY:
    image = np.resize(data, (height, width, 2))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
  if fmt == ob.OBFormat.MJPG:
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
      raise RuntimeError('MJPG 解码失败')
    return image
  if fmt == ob.OBFormat.NV12:
    image = np.resize(data, (height * 3 // 2, width))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
  if fmt == ob.OBFormat.NV21:
    image = np.resize(data, (height * 3 // 2, width))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)
  if fmt == ob.OBFormat.I420:
    image = np.resize(data, (height * 3 // 2, width))
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
  if fmt == ob.OBFormat.Y8:
    image = np.resize(data, (height, width)).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  if fmt == ob.OBFormat.Y16:
    image = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  raise RuntimeError(f'不支持的 Orbbec 帧格式：{fmt}')


def sensor_available(device, sensor_type):
  sensors = device.get_sensor_list()
  for index in range(sensors.get_count()):
    if sensors.get_sensor_by_index(index).get_type() == sensor_type:
      return True
  return False


def pick_profile(pipeline, sensor_type, width, height, fps, formats, ob):
  profiles = pipeline.get_stream_profile_list(sensor_type)
  for fmt_name in formats:
    try:
      profile = profiles.get_video_stream_profile(int(width), int(height), ob_format_by_name(ob, fmt_name), int(fps))
      logging.info(f'Orbbec profile: sensor={sensor_type}, width={width}, height={height}, fps={fps}, format={fmt_name}')
      return profile
    except Exception:
      continue
  raise RuntimeError(f'找不到 {sensor_type} 的 {width}x{height}@{fps} profile，候选格式：{formats}')


def device_identity(device):
  info = device.get_device_info()
  return {
    'name': str(info.get_name()),
    'serial_number': str(info.get_serial_number()),
    'firmware_version': str(info.get_firmware_version()),
  }


def validate_device_identity(device, config):
  identity = device_identity(device)
  expected_serial = str(camera_cfg(config, 'serial_number', '') or '').strip()
  if expected_serial and identity['serial_number'] != expected_serial:
    raise RuntimeError(
      f'Orbbec serial mismatch: expected={expected_serial}, actual={identity["serial_number"]}'
    )
  logging.info(
    'Orbbec device: name=%s serial=%s firmware=%s',
    identity['name'], identity['serial_number'], identity['firmware_version'],
  )
  return identity


def select_dual_color_profiles(pipeline, config, ob, formats=None):
  device = pipeline.get_device()
  left_sensor = ob.OBSensorType.LEFT_COLOR_SENSOR
  right_sensor = ob.OBSensorType.RIGHT_COLOR_SENSOR
  if not sensor_available(device, left_sensor) or not sensor_available(device, right_sensor):
    raise RuntimeError('Gemini 305 does not expose both LEFT_COLOR_SENSOR and RIGHT_COLOR_SENSOR')
  width = int(camera_cfg(config, 'width', 1280))
  height = int(camera_cfg(config, 'height', 800))
  fps = int(camera_cfg(config, 'fps', 30))
  candidates = formats or camera_cfg(config, 'formats', ['BGR', 'RGB', 'YUYV', 'MJPG'])
  left = pick_profile(pipeline, left_sensor, width, height, fps, candidates, ob)
  right = pick_profile(pipeline, right_sensor, width, height, fps, candidates, ob)
  return left, right


def frameset_dual_color_frames(frames, ob):
  left_getter = getattr(frames, 'get_left_color_frame', None)
  right_getter = getattr(frames, 'get_right_color_frame', None)
  if left_getter is not None and right_getter is not None:
    return left_getter(), right_getter()
  # SDK 2.8.6 Python bindings expose generic get_frame for these frame types.
  return (
    frames.get_frame(ob.OBFrameType.LEFT_COLOR_FRAME),
    frames.get_frame(ob.OBFrameType.RIGHT_COLOR_FRAME),
  )


def build_orbbec_streamer():
  include_dir = ORBBEC_SDK_DIR / 'include'
  lib_dir = ORBBEC_SDK_DIR / 'lib'
  if not ORBBEC_STREAM_CPP.exists():
    raise RuntimeError(f'找不到 Orbbec C++ 采集源文件：{ORBBEC_STREAM_CPP}')
  if not (lib_dir / 'libOrbbecSDK.so').exists():
    raise RuntimeError(f'找不到 Orbbec SDK 2.8.6 动态库：{lib_dir / "libOrbbecSDK.so"}')
  needs_build = (
    not ORBBEC_STREAM_BIN.exists()
    or ORBBEC_STREAM_CPP.stat().st_mtime > ORBBEC_STREAM_BIN.stat().st_mtime
  )
  if not needs_build:
    return ORBBEC_STREAM_BIN

  ORBBEC_STREAM_BIN.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    'g++',
    '-std=c++17',
    str(ORBBEC_STREAM_CPP),
    '-o',
    str(ORBBEC_STREAM_BIN),
    f'-I{include_dir}',
    f'-L{lib_dir}',
    '-lOrbbecSDK',
    f'-Wl,-rpath,{lib_dir}',
  ]
  logging.info('Building Orbbec C++ streamer: %s', ' '.join(cmd))
  subprocess.run(cmd, check=True)
  return ORBBEC_STREAM_BIN


def read_exact(stream, size):
  chunks = []
  remaining = size
  while remaining:
    chunk = stream.read(remaining)
    if not chunk:
      raise RuntimeError('Orbbec C++ 采集进程已退出或没有继续输出帧')
    chunks.append(chunk)
    remaining -= len(chunk)
  return b''.join(chunks)


class CppOrbbecStereoSource:
  def __init__(self, config):
    self.calibration = None
    if bool(camera_cfg(config, 'query_sdk_calibration', True)):
      try:
        ob = load_orbbec_sdk()
        probe = ob.Pipeline()
        device = probe.get_device()
        preset = str(camera_cfg(config, 'preset', 'Dual Color Streams'))
        device.load_preset(preset)
        identity = validate_device_identity(device, config)
        left_profile, right_profile = select_dual_color_profiles(probe, config, ob, formats=['BGR'])
        self.calibration = stereo_calibration_from_sdk_profiles(
          left_profile,
          right_profile,
          source=f'Orbbec SDK {identity["serial_number"]} Dual RGB profiles',
        )
        del left_profile, right_profile, device, probe
      except Exception as exc:
        if not bool(camera_cfg(config, 'allow_calibration_fallback', True)):
          raise
        logging.warning('SDK calibration query failed; configured fallback may be used: %s', exc)
    binary = build_orbbec_streamer()
    width = int(camera_cfg(config, 'width', 1280))
    height = int(camera_cfg(config, 'height', 800))
    fps = int(camera_cfg(config, 'fps', 30))
    preset = str(camera_cfg(config, 'preset', 'Dual Color Streams'))
    self.process = subprocess.Popen(
      [str(binary), str(width), str(height), str(fps), preset],
      stdout=subprocess.PIPE,
      stderr=None,
      bufsize=0,
    )
    self.width = width
    self.height = height
    logging.info('Camera source: orbbec_sdk_cpp, width=%s, height=%s, fps=%s, preset=%s', width, height, fps, preset)

  def read(self):
    started = time.perf_counter()
    header_bytes = read_exact(self.process.stdout, ORBBEC_PACKET_HEADER.size)
    (
      magic, width, height, channels, left_bytes, right_bytes, frame_id,
      left_frame_id, right_frame_id, left_timestamp_us, right_timestamp_us,
      left_system_timestamp_us, right_system_timestamp_us,
    ) = ORBBEC_PACKET_HEADER.unpack(header_bytes)
    if magic != b'OBL2':
      raise RuntimeError(f'Orbbec C++ 帧头异常：{magic!r}')
    if channels != 3:
      raise RuntimeError(f'Orbbec C++ 输出通道数异常：{channels}')

    expected_bytes = width * height * channels
    if left_bytes < expected_bytes or right_bytes < expected_bytes:
      raise RuntimeError(
        f'Orbbec C++ 输出尺寸异常：{width}x{height}x{channels}, left={left_bytes}, right={right_bytes}'
      )
    left = np.frombuffer(read_exact(self.process.stdout, left_bytes), dtype=np.uint8)[:expected_bytes]
    right = np.frombuffer(read_exact(self.process.stdout, right_bytes), dtype=np.uint8)[:expected_bytes]
    left = left.reshape((height, width, channels)).copy()
    right = right.reshape((height, width, channels)).copy()
    left_timestamp_ms = left_timestamp_us / 1000.0
    right_timestamp_ms = right_timestamp_us / 1000.0
    return {
      'frame_id': frame_id,
      'left_frame_id': left_frame_id,
      'right_frame_id': right_frame_id,
      'left_timestamp_ms': left_timestamp_ms,
      'right_timestamp_ms': right_timestamp_ms,
      'timestamp_ms': (left_timestamp_ms + right_timestamp_ms) / 2.0,
      'timestamp_delta_ms': abs(left_timestamp_ms - right_timestamp_ms),
      'left_system_timestamp_ms': left_system_timestamp_us / 1000.0,
      'right_system_timestamp_ms': right_system_timestamp_us / 1000.0,
      'left_bgr': left,
      'right_bgr': right,
      'capture_ms': (time.perf_counter() - started) * 1000.0,
    }

  def release(self):
    if self.process.poll() is None:
      self.process.terminate()
      try:
        self.process.wait(timeout=2)
      except subprocess.TimeoutExpired:
        self.process.kill()
        self.process.wait()


class OrbbecStereoSource:
  def __init__(self, config):
    self._runtime_config = config
    self.ob = load_orbbec_sdk()
    self.pipeline = self.ob.Pipeline()
    self.device = self.pipeline.get_device()
    self.identity = validate_device_identity(self.device, config)
    self.config = self.ob.Config()

    preset = camera_cfg(config, 'preset')
    if preset:
      logging.info(f'Orbbec load preset: {preset}')
      self.device.load_preset(str(preset))
      logging.info(f'Orbbec current preset: {self.device.get_current_preset_name()}')

    stream = camera_cfg(config, 'stream', 'auto')
    width = int(camera_cfg(config, 'width', 1280))
    height = int(camera_cfg(config, 'height', 800))
    fps = int(camera_cfg(config, 'fps', 30))
    formats = camera_cfg(config, 'formats', ['BGR', 'RGB', 'YUYV', 'MJPG'])
    ir_formats = camera_cfg(config, 'ir_formats', ['Y8'])

    if stream in {'auto', 'dual_color'} and self._enable_dual_color(width, height, fps, formats):
      self.stream = 'dual_color'
    elif stream in {'auto', 'dual_ir'} and self._enable_dual_ir(width, height, fps, ir_formats):
      self.stream = 'dual_ir'
    else:
      raise RuntimeError('Gemini 305 当前没有可用的左右彩色流；也没有启用成功的左右 IR 流。')

    self.config.set_frame_aggregate_output_mode(self.ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    self.pipeline.enable_frame_sync()
    self.pipeline.start(self.config)
    logging.info(f'Orbbec SDK source started: stream={self.stream}, width={width}, height={height}, fps={fps}')

  def _enable_dual_color(self, width, height, fps, formats):
    left = self.ob.OBSensorType.LEFT_COLOR_SENSOR
    right = self.ob.OBSensorType.RIGHT_COLOR_SENSOR
    if not sensor_available(self.device, left) or not sensor_available(self.device, right):
      return False
    self.left_profile = pick_profile(self.pipeline, left, width, height, fps, formats, self.ob)
    self.right_profile = pick_profile(self.pipeline, right, width, height, fps, formats, self.ob)
    try:
      self.calibration = stereo_calibration_from_sdk_profiles(
        self.left_profile,
        self.right_profile,
        source=f'Orbbec SDK {self.identity["serial_number"]} Dual RGB profiles',
      )
    except Exception as exc:
      if not bool(camera_cfg(self._runtime_config, 'allow_calibration_fallback', True)):
        raise
      self.calibration = None
      logging.warning('SDK calibration query failed; configured fallback may be used: %s', exc)
    self.config.enable_stream(self.left_profile)
    self.config.enable_stream(self.right_profile)
    return True

  def _enable_dual_ir(self, width, height, fps, formats):
    left = self.ob.OBSensorType.LEFT_IR_SENSOR
    right = self.ob.OBSensorType.RIGHT_IR_SENSOR
    if not sensor_available(self.device, left) or not sensor_available(self.device, right):
      return False
    self.config.enable_stream(pick_profile(self.pipeline, left, width, height, fps, formats, self.ob))
    self.config.enable_stream(pick_profile(self.pipeline, right, width, height, fps, formats, self.ob))
    return True

  def read(self):
    started = time.perf_counter()
    if self.stream != 'dual_color':
      left_type = self.ob.OBFrameType.LEFT_IR_FRAME
      right_type = self.ob.OBFrameType.RIGHT_IR_FRAME

    for _ in range(30):
      frames = self.pipeline.wait_for_frames(1000)
      if frames is None:
        continue
      if self.stream == 'dual_color':
        left, right = frameset_dual_color_frames(frames, self.ob)
      else:
        left = frames.get_frame(left_type)
        right = frames.get_frame(right_type)
      if left is not None and right is not None:
        left_timestamp_ms = left.get_timestamp_us() / 1000.0
        right_timestamp_ms = right.get_timestamp_us() / 1000.0
        return {
          'frame_id': int(frames.get_index()),
          'left_frame_id': int(left.get_index()),
          'right_frame_id': int(right.get_index()),
          'left_timestamp_ms': left_timestamp_ms,
          'right_timestamp_ms': right_timestamp_ms,
          'timestamp_ms': (left_timestamp_ms + right_timestamp_ms) / 2.0,
          'timestamp_delta_ms': abs(left_timestamp_ms - right_timestamp_ms),
          'left_system_timestamp_ms': left.get_system_timestamp_us() / 1000.0,
          'right_system_timestamp_ms': right.get_system_timestamp_us() / 1000.0,
          'left_bgr': frame_to_bgr(left, self.ob),
          'right_bgr': frame_to_bgr(right, self.ob),
          'capture_ms': (time.perf_counter() - started) * 1000.0,
        }
    raise RuntimeError(f'Orbbec {self.stream} 连续等待后仍没有同时拿到左右帧')

  def release(self):
    self.pipeline.stop()


class LatestFrameWorker:
  def __init__(self, source):
    self.source = source
    self.queue = queue.Queue(maxsize=1)
    self.stop_event = threading.Event()
    self.error = None
    self.captured = 0
    self.dropped = 0
    self.thread = threading.Thread(target=self._run, name='stereo-capture', daemon=True)

  def start(self):
    self.thread.start()

  def _run(self):
    while not self.stop_event.is_set():
      try:
        frame = self.source.read()
      except Exception as exc:
        self.error = exc
        self.stop_event.set()
        return
      self.captured += 1
      try:
        self.queue.put_nowait(frame)
      except queue.Full:
        try:
          self.queue.get_nowait()
          self.dropped += 1
        except queue.Empty:
          pass
        self.queue.put_nowait(frame)

  def get(self, timeout=2.0):
    try:
      return self.queue.get(timeout=timeout)
    except queue.Empty:
      if self.error is not None:
        raise RuntimeError(f'camera capture failed: {self.error}') from self.error
      raise RuntimeError('camera capture timed out')

  def close(self):
    self.stop_event.set()
    self.thread.join(timeout=2.0)


def make_output_dir(base_dir, frame_id, save_frames):
  base_dir = Path(base_dir)
  if save_frames:
    return base_dir / f'frame_{frame_id:06d}'
  return base_dir


def save_camera_debug_images(out_dir, left_bgr, right_bgr):
  raw_dir = Path(out_dir) / 'raw'
  raw_dir.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(raw_dir / 'left.png'), left_bgr)
  cv2.imwrite(str(raw_dir / 'right.png'), right_bgr)
  if right_bgr.shape[:2] != left_bgr.shape[:2]:
    right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))
  cv2.imwrite(str(raw_dir / 'preview.jpg'), cv2.hconcat([left_bgr, right_bgr]))


def frame_stats(image):
  return {
    'min': image.min(axis=(0, 1)).tolist(),
    'max': image.max(axis=(0, 1)).tolist(),
    'mean': image.mean(axis=(0, 1)).round(2).tolist(),
    'std': image.std(axis=(0, 1)).round(2).tolist(),
  }


def is_low_information_frame(left_bgr, right_bgr, min_std):
  if min_std <= 0:
    return False
  left_std = float(left_bgr.std())
  right_std = float(right_bgr.std())
  return left_std < min_std or right_std < min_std


def preview_pair(left_bgr, right_bgr):
  if right_bgr.shape[:2] != left_bgr.shape[:2]:
    right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))
  view = cv2.hconcat([left_bgr, right_bgr])
  cv2.imshow('left | right', view)
  return cv2.waitKey(1) & 0xFF


def rectifier_for_source(source, config):
  calibration = getattr(source, 'calibration', None)
  if calibration is None:
    if not bool(camera_cfg(config, 'allow_calibration_fallback', True)):
      raise RuntimeError('RECTIFICATION_INVALID: Orbbec SDK calibration is unavailable')
    calibration_file = config.get('stereo_calibration_file')
    if not calibration_file:
      raise RuntimeError('RECTIFICATION_INVALID: no SDK calibration and no fallback file')
    try:
      calibration = load_stereo_calibration(run_config.resolve_path(calibration_file))
    except (ValueError, FileNotFoundError) as exc:
      raise RuntimeError(f'RECTIFICATION_INVALID: {exc}') from exc
    logging.warning('Using stereo calibration fallback: %s', calibration.source)
  else:
    logging.info(
      'Using live SDK stereo calibration: source=%s baseline=%.9fm T=%s',
      calibration.source,
      calibration.baseline_m,
      calibration.T.reshape(3).tolist(),
    )
  try:
    return StereoRectifier(
      calibration,
      alpha=float((config.get('rectification') or {}).get('alpha', 0.0)),
    )
  except (ValueError, cv2.error) as exc:
    raise RuntimeError(f'RECTIFICATION_INVALID: {exc}') from exc


def main(argv=None):
  cli = parse_args(argv)
  config_path = run_config.resolve_path(cli.config)
  config = run_config.apply_runtime_safety(run_config.load_yaml(config_path))
  camera = dict(config.get('camera') or {})
  for key in ('width', 'height', 'fps'):
    value = getattr(cli, key)
    if value is not None:
      camera[key] = value
  config['camera'] = camera
  locator_config = config.get('locator') or {}
  segmentation = config.get('segmentation') or {}
  if not bool(locator_config.get('enabled', False)):
    raise RuntimeError('realtime config must set locator.enabled=true')
  if not bool(segmentation.get('enabled', False)):
    raise RuntimeError('realtime triangle localization requires segmentation.enabled=true')

  interval_sec = float(cli_or_config(cli.interval_sec, config, 'interval_sec', 0))
  max_frames = int(cli_or_config(cli.max_frames, config, 'max_frames', 0))
  source = camera_cfg(config, 'source', 'orbbec_sdk')
  preview = bool(cli.preview or camera_cfg(config, 'preview', False))
  save_raw_frames = bool(camera_cfg(config, 'save_raw_frames', False))
  debug = bool((config.get('debug') or {}).get('enabled', False))
  save_preview = bool((config.get('output') or {}).get('save_preview', False))
  min_frame_std = float(camera_cfg(config, 'min_frame_std', 2.0))

  if cli.out_dir:
    out_dir = run_config.resolve_path(cli.out_dir)
  else:
    out_dir = run_config.resolve_path(camera_cfg(config, 'out_dir', Path(config.get('out_dir', 'workspace/output')) / 'camera_latest'))

  sdk_source = None
  worker = None
  processed = 0
  plate_valid_count = 0
  screw_valid_count = 0
  timing_rows = deque(maxlen=1000)
  plate_origins = deque(maxlen=1000)
  screw_pixels = deque(maxlen=1000)
  plane_rmses = deque(maxlen=1000)
  try:
    if source in {'orbbec_sdk', 'orbbec_sdk_cpp'}:
      sdk_source = CppOrbbecStereoSource(config)
      logging.info(f'Camera source: orbbec_sdk_cpp, output={out_dir}')
    elif source in {'orbbec_py', 'pyorbbecsdk'}:
      sdk_source = OrbbecStereoSource(config)
      logging.info(f'Camera source: orbbec_py, output={out_dir}')
    else:
      raise RuntimeError('complete realtime locator requires source=orbbec_sdk_cpp or orbbec_py')

    worker = LatestFrameWorker(sdk_source)
    worker.start()
    rectifier = rectifier_for_source(sdk_source, config)
    run_demo.configure_runtime()
    model_args = run_demo.load_args(run_config.runtime_overrides(config))
    model_args.show = 0
    model_args.intrinsic_file = None
    model = run_demo.load_model(model_args)
    locator = TriangleLocator(model, model_args, rectifier, config, repo_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / str((config.get('output') or {}).get('jsonl_file', 'results.jsonl'))
    latest_json_path = out_dir / str((config.get('output') or {}).get('latest_json_file', 'latest_result.json'))
    jsonl_path.write_text('', encoding='utf-8')
    while True:
      frame = worker.get(timeout=3.0)
      frame_id = int(frame['frame_id'])
      left_bgr = frame['left_bgr']
      right_bgr = frame['right_bgr']
      sample_out_dir = make_output_dir(out_dir, frame_id, bool(debug and cli.save_frames))
      if debug:
        sample_out_dir.mkdir(parents=True, exist_ok=True)

      if save_raw_frames:
        save_camera_debug_images(sample_out_dir, left_bgr, right_bgr)

      if is_low_information_frame(left_bgr, right_bgr, min_frame_std):
        logging.warning(
          '相机帧像素变化太小，跳过本帧推理。left=%s right=%s preview=%s',
          frame_stats(left_bgr),
          frame_stats(right_bgr),
          Path(sample_out_dir) / 'raw' / 'preview.jpg',
        )
        if max_frames and processed >= max_frames:
          break
        if interval_sec > 0:
          time.sleep(interval_sec)
        continue

      if preview and os.environ.get('DISPLAY') and preview_pair(left_bgr, right_bgr) == ord('q'):
        break
      if preview and not os.environ.get('DISPLAY') and frame_id == 1:
        logging.info(f'No DISPLAY，已保存可视化预览：{Path(sample_out_dir) / "raw" / "preview.jpg"}')

      try:
        result = locator.process(
          frame,
          output_dir=sample_out_dir if debug else (out_dir if save_preview else None),
          debug=debug,
          save_preview=save_preview,
        )
      except RuntimeError as exc:
        run_demo.release_cuda_cache()
        message = str(exc).lower()
        if 'out of memory' in message or 'cuda' in message:
          logging.error('CUDA 推理失败，已清理缓存并停止相机循环：%s', exc)
          break
        raise
      append_result_jsonl(jsonl_path, result)
      write_result_json(latest_json_path, result)
      processed += 1
      timing_rows.append(result['timings_ms'])
      if result['plate_valid']:
        plate_valid_count += 1
        plate_origins.append(result['plate_origin_camera_m'])
        if result['plane_rmse_px'] is not None:
          plane_rmses.append(result['plane_rmse_px'])
      if result['screw_valid']:
        screw_valid_count += 1
        screw_pixels.append([result['screw_u_raw'], result['screw_v_raw']])
      logging.info(
        'frame=%s plate_valid=%s screw_valid=%s reason=%s total=%.1fms captured=%s processed=%s dropped=%s',
        frame_id,
        result['plate_valid'],
        result['screw_valid'],
        result['invalid_reason'],
        result['timings_ms']['total_ms'],
        worker.captured,
        processed,
        worker.dropped,
      )

      if max_frames and processed >= max_frames:
        break
      if interval_sec > 0:
        time.sleep(interval_sec)
  except KeyboardInterrupt:
    logging.info('Stopped by user')
  finally:
    if worker is not None:
      worker.close()
    if sdk_source is not None:
      sdk_source.release()
    if preview and os.environ.get('DISPLAY'):
      cv2.destroyAllWindows()
    if timing_rows:
      logging.info(
        'runtime summary: captured=%s processed=%s dropped=%s plate_valid_ratio=%.3f screw_valid_ratio=%.3f',
        worker.captured if worker else 0,
        processed,
        worker.dropped if worker else 0,
        plate_valid_count / processed if processed else 0.0,
        screw_valid_count / processed if processed else 0.0,
      )
      for key in timing_rows[0]:
        values = np.asarray([row[key] for row in timing_rows], dtype=np.float64)
        logging.info(
          '%s: avg=%.1fms p90=%.1fms p95=%.1fms',
          key, values.mean(), np.percentile(values, 90), np.percentile(values, 95),
        )
      if plate_origins:
        logging.info('plate_origin_std_m=%s', np.std(np.asarray(plate_origins), axis=0).tolist())
      if screw_pixels:
        logging.info('screw_uv_std_px=%s', np.std(np.asarray(screw_pixels), axis=0).tolist())
      if plane_rmses:
        logging.info('plane_rmse_px_mean=%.4f', float(np.mean(plane_rmses)))


if __name__ == '__main__':
  main()
