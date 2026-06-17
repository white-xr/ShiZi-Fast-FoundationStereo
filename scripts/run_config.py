import argparse
import csv
from datetime import datetime
import html
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

code_dir = Path(__file__).resolve().parent
repo_dir = code_dir.parent
sys.path.append(str(repo_dir))

from scripts import run_demo
from Utils import o3d


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEFAULT_TRANSFER_FILES = [
  'depth_meter.npy',
  'depth_mm.png',
  'depth_vis.png',
  'cloud.ply',
  'cloud_denoise.ply',
  'merged_cloud.ply',
]
POSE_COLUMNS = [
  'm00', 'm01', 'm02', 'm03',
  'm10', 'm11', 'm12', 'm13',
  'm20', 'm21', 'm22', 'm23',
  'm30', 'm31', 'm32', 'm33',
]


def parse_args():
  parser = argparse.ArgumentParser(description='Run FoundationStereo from a YAML config.')
  parser.add_argument('--config', default='configs/stereo_infer.yaml', help='YAML config path')
  parser.add_argument('--limit', type=int, default=None, help='override batch limit')
  parser.add_argument('--dry_run', action='store_true', help='print matched pairs without inference')
  parser.add_argument('--check_transfer', action='store_true', help='test Windows SSH/SCP settings without inference')
  return parser.parse_args()


def load_yaml(path):
  with open(path, 'r', encoding='utf-8') as f:
    return yaml.safe_load(f) or {}


def resolve_path(value):
  if value is None:
    return None
  value = os.path.expandvars(os.path.expanduser(str(value)))
  path = Path(value)
  if path.is_absolute():
    return path
  return (repo_dir / path).resolve()


def default_suffix(pattern):
  suffix = Path(pattern.replace('*', 'sample')).suffix
  return suffix if suffix else '.png'


def id_to_path(directory, item, suffix):
  item_path = Path(str(item))
  filename = item_path.name if item_path.suffix else f'{item}{suffix}'
  return resolve_path(directory) / filename


def sample_name(left_file, provided=None):
  return str(provided) if provided else Path(left_file).stem


def is_auto_value(value):
  return value is None or str(value).strip().lower() in {'', 'auto'}


def get_data_dir(config):
  value = config.get('data_dir') or config.get('dataset_dir')
  if value:
    return resolve_path(value)

  batch = config.get('batch') or {}
  value = batch.get('data_dir') or batch.get('dataset_dir')
  if value:
    return resolve_path(value)

  return None


def apply_auto_dataset_paths(config):
  config = dict(config)
  data_dir = get_data_dir(config)
  if data_dir is None:
    return config

  if not data_dir.exists():
    raise FileNotFoundError(f'找不到数据目录：{data_dir}')

  intrinsic_file = config.get('intrinsic_file')
  if is_auto_value(intrinsic_file):
    k_file = data_dir / 'K.txt'
    if not k_file.exists():
      raise FileNotFoundError(f'数据目录下找不到 K.txt：{k_file}')
    config['intrinsic_file'] = str(k_file)

  batch = dict(config.get('batch') or {})
  batch.setdefault('pattern', '*.png')
  if is_auto_value(batch.get('left_dir')):
    batch['left_dir'] = str(data_dir / 'left_rgb')
  if is_auto_value(batch.get('right_dir')):
    batch['right_dir'] = str(data_dir / 'right_rgb')
  for key in ('left_dir', 'right_dir'):
    path = resolve_path(batch[key])
    if not path.exists():
      raise FileNotFoundError(f'数据目录下找不到 {path.name}：{path}')
  config['batch'] = batch

  return config


def collect_pairs(config):
  pairs = []

  for item in config.get('pairs', []) or []:
    pairs.append({
      'left_file': resolve_path(item['left_file']),
      'right_file': resolve_path(item['right_file']),
      'name': sample_name(item['left_file'], item.get('name')),
    })

  single = config.get('single')
  if single:
    pairs.append({
      'left_file': resolve_path(single['left_file']),
      'right_file': resolve_path(single['right_file']),
      'name': sample_name(single['left_file'], single.get('name')),
    })

  batch = config.get('batch')
  if batch:
    left_dir = resolve_path(batch['left_dir'])
    right_dir = resolve_path(batch['right_dir'])
    pattern = batch.get('pattern', '*.png')
    suffix = default_suffix(pattern)
    ids = batch.get('ids')

    if ids:
      left_files = [id_to_path(left_dir, item, suffix) for item in ids]
    else:
      globber = left_dir.rglob if batch.get('recursive', False) else left_dir.glob
      left_files = sorted(p for p in globber(pattern) if p.suffix.lower() in IMAGE_EXTS)

    start = batch.get('start')
    end = batch.get('end')
    if start is not None or end is not None:
      start = str(start) if start is not None else None
      end = str(end) if end is not None else None
      left_files = [
        p for p in left_files
        if (start is None or p.stem >= start) and (end is None or p.stem <= end)
      ]

    limit = batch.get('limit')
    if limit:
      left_files = left_files[:int(limit)]

    for left_file in left_files:
      right_file = right_dir / left_file.name
      if not right_file.exists():
        logging.warning(f'Missing right image for {left_file.name}: {right_file}')
        continue
      pairs.append({
        'left_file': left_file,
        'right_file': right_file,
        'name': sample_name(left_file),
      })

  seen = {}
  unique_pairs = []
  for pair in pairs:
    name = pair['name']
    seen[name] = seen.get(name, 0) + 1
    if seen[name] > 1:
      pair = pair.copy()
      pair['name'] = f'{name}_{seen[name]}'
    unique_pairs.append(pair)
  return unique_pairs


def runtime_overrides(config):
  parser_keys = {action.dest for action in run_demo.create_parser()._actions}
  skip = {'help', 'left_file', 'right_file'}
  return {
    key: value
    for key, value in config.items()
    if key in parser_keys and key not in skip
  }


def clone_args(args, updates):
  data = OmegaConf.to_container(args, resolve=True)
  data.update(updates)
  return OmegaConf.create(data)


def make_run_name(config):
  run_name = config.get('run_name')
  if run_name:
    return str(run_name)
  return datetime.now().strftime(str(config.get('run_dir_format', '%m%d%H%M')))


def load_pose_file(path):
  if not path:
    raise ValueError('fusion.enabled=true 时必须设置 fusion.pose_file。')
  pose_path = resolve_path(path)
  if not pose_path.exists():
    raise FileNotFoundError(f'融合已开启，但找不到位姿文件：{pose_path}')

  poses = {}
  with open(pose_path, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    missing_columns = ['frame_id', *POSE_COLUMNS]
    missing_columns = [name for name in missing_columns if name not in (reader.fieldnames or [])]
    if missing_columns:
      raise ValueError(f'位姿文件缺少列：{", ".join(missing_columns)}')

    for row in reader:
      frame_id = str(row['frame_id']).strip()
      if not frame_id:
        continue
      values = [float(row[name]) for name in POSE_COLUMNS]
      poses[frame_id] = np.array(values, dtype=np.float64).reshape(4, 4)
  return poses


def validate_fusion_poses(pairs, poses, missing_pose_policy):
  missing = [pair['name'] for pair in pairs if pair['name'] not in poses]
  if missing and missing_pose_policy == 'error':
    preview = ', '.join(missing[:10])
    suffix = ' ...' if len(missing) > 10 else ''
    raise ValueError(f'位姿文件缺少 {len(missing)} 帧：{preview}{suffix}')
  return missing


def merge_point_clouds(base_out_dir, results, fusion, poses):
  if not fusion or not fusion.get('enabled', False):
    return None
  if o3d is None:
    raise RuntimeError('融合点云需要 open3d，但当前环境没有导入成功。')

  missing_pose_policy = fusion.get('missing_pose_policy', 'error')
  merged = o3d.geometry.PointCloud()
  used = 0
  skipped = []

  for result in results:
    name = result['name']
    pose = poses.get(name)
    if pose is None:
      if missing_pose_policy == 'skip':
        skipped.append(name)
        continue
      raise ValueError(f'位姿文件缺少帧：{name}')

    cloud_path = base_out_dir / result['rel'] / 'cloud.ply'
    if not cloud_path.exists():
      logging.warning(f'跳过融合，未找到点云：{cloud_path}')
      continue

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    cloud.transform(pose)
    merged += cloud
    used += 1

  if used == 0:
    raise RuntimeError('没有可用于融合的点云。')

  voxel_size = float(fusion.get('voxel_size', 0.0) or 0.0)
  if voxel_size > 0:
    logging.info(f'融合点云体素降采样：{voxel_size} m')
    merged = merged.voxel_down_sample(voxel_size=voxel_size)

  output_file = fusion.get('output_file', 'merged_cloud.ply')
  output_path = base_out_dir / output_file
  output_path.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_point_cloud(str(output_path), merged)
  logging.info(f'融合点云已保存：{output_path}，使用 {used} 个点云')
  if skipped:
    logging.info(f'融合跳过 {len(skipped)} 帧缺失位姿。')
  return output_path


def write_batch_index(out_dir, results):
  rows = []
  for result in results:
    name = html.escape(result['name'])
    rel = html.escape(result['rel'])
    rows.append(
      f'<section><a href="{rel}/index.html"><h2>{name}</h2>'
      f'<img src="{rel}/depth_vis.png" alt="{name} depth"></a></section>'
    )
  root_links = []
  for path in sorted(out_dir.glob('*.ply')):
    name = html.escape(path.name)
    root_links.append(f'<li><a href="{name}">{name}</a></li>')

  page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stereo batch results</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f7f7f5; color: #202124; }}
    h1 {{ font-size: 24px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    section {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    a {{ color: inherit; text-decoration: none; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    img {{ width: 100%; display: block; background: #111; }}
    ul {{ margin-top: 18px; }}
  </style>
</head>
<body>
  <h1>Stereo batch results</h1>
  <ul>{''.join(root_links)}</ul>
  <main>{''.join(rows)}</main>
</body>
</html>
"""
  (out_dir / 'index.html').write_text(page, encoding='utf-8')


def powershell_quote(value):
  return "'" + str(value).replace("'", "''") + "'"


def ssh_target(transfer):
  user = transfer.get('user')
  host = transfer.get('host')
  if not host:
    raise ValueError('transfer.host is empty')
  return f'{user}@{host}' if user else str(host)


def remote_sample_dir(transfer, sample_path):
  remote_root = str(transfer.get('remote_dir', 'D:/FAST-FS')).replace('\\', '/').rstrip('/')
  return f'{remote_root}/{sample_path}'


def transfer_timeout(transfer):
  return int(transfer.get('timeout_sec', 20))


def run_transfer_command(cmd, fail_on_error, timeout_sec=20):
  try:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout_sec)
    if result.stdout.strip():
      logging.info(result.stdout.strip())
    if result.stderr.strip():
      logging.info(result.stderr.strip())
    return True
  except FileNotFoundError as exc:
    message = f'传输失败：找不到命令 {cmd[0]}，请先安装 OpenSSH 客户端。'
  except subprocess.TimeoutExpired as exc:
    message = f'传输超时：{timeout_sec}s 内没有连上。请检查 IP、防火墙和 Windows OpenSSH Server。'
  except subprocess.CalledProcessError as exc:
    details = '\n'.join(part.strip() for part in [exc.stdout or '', exc.stderr or ''] if part.strip())
    message = f'传输命令失败，退出码 {exc.returncode}：{" ".join(cmd)}'
    if details:
      message += f'\nSSH/SCP 输出：\n{details}'

  if fail_on_error:
    raise RuntimeError(message)
  logging.warning(message)
  return False


def check_transfer(transfer):
  if not transfer or not transfer.get('enabled', False):
    logging.info('transfer.enabled 现在是 false，没有开启传输。')
    return

  target = ssh_target(transfer)
  port = str(transfer.get('port', 22))
  timeout_sec = transfer_timeout(transfer)
  remote_dir = remote_sample_dir(transfer, '_ssh_test')

  logging.info(f'测试 SSH 连接：{target}:{port}')
  test_cmd = ['ssh', '-p', port, target, 'echo ssh_ok']
  if not run_transfer_command(test_cmd, False, timeout_sec):
    return

  logging.info(f'测试创建 Windows 目录：{remote_dir}')
  ps = f'New-Item -ItemType Directory -Force -Path {powershell_quote(remote_dir)} | Out-Null'
  mkdir_cmd = ['ssh', '-p', port, target, f'powershell -NoProfile -Command "{ps}"']
  if run_transfer_command(mkdir_cmd, False, timeout_sec):
    logging.info('传输配置检查通过。')


def transfer_outputs(sample_out_dir, remote_relative_dir, transfer):
  if not transfer or not transfer.get('enabled', False):
    return

  target = ssh_target(transfer)
  port = str(transfer.get('port', 22))
  remote_dir = remote_sample_dir(transfer, remote_relative_dir)
  fail_on_error = bool(transfer.get('fail_on_error', False))
  timeout_sec = transfer_timeout(transfer)

  if transfer.get('create_remote_dir', True):
    ps = f'New-Item -ItemType Directory -Force -Path {powershell_quote(remote_dir)} | Out-Null'
    mkdir_cmd = ['ssh', '-p', port, target, f'powershell -NoProfile -Command "{ps}"']
    if not run_transfer_command(mkdir_cmd, fail_on_error, timeout_sec):
      return

  file_names = transfer.get('files') or DEFAULT_TRANSFER_FILES
  local_files = [sample_out_dir / name for name in file_names if (sample_out_dir / name).exists()]
  if not local_files:
    logging.warning(f'没有找到可传输文件：{sample_out_dir}')
    return

  scp_target = f'{target}:{remote_dir}/'
  scp_cmd = ['scp', '-P', port, *[str(path) for path in local_files], scp_target]
  if run_transfer_command(scp_cmd, fail_on_error, timeout_sec):
    logging.info(f'已传输 {len(local_files)} 个文件到 {target}:{remote_dir}/')


def main():
  args = parse_args()
  config_path = resolve_path(args.config)
  config = apply_auto_dataset_paths(load_yaml(config_path))

  run_demo.configure_runtime()
  if args.check_transfer:
    check_transfer(config.get('transfer', {}))
    return

  pairs = collect_pairs(config)
  if args.limit:
    pairs = pairs[:args.limit]
  if not pairs:
    raise SystemExit('No stereo pairs found. Check single/batch/pairs in the config.')

  logging.info(f'Config: {config_path}')
  if config.get('intrinsic_file'):
    logging.info(f'Intrinsic: {resolve_path(config["intrinsic_file"])}')
  logging.info(f'Pairs: {len(pairs)}')
  for idx, pair in enumerate(pairs, 1):
    logging.info(f'[{idx}/{len(pairs)}] {pair["name"]}: {pair["left_file"]} | {pair["right_file"]}')
  if args.dry_run:
    return

  fusion = config.get('fusion', {})
  poses = None
  if fusion and fusion.get('enabled', False):
    poses = load_pose_file(fusion.get('pose_file'))
    validate_fusion_poses(pairs, poses, fusion.get('missing_pose_policy', 'error'))

  output_root = resolve_path(config.get('out_dir', 'workspace/output'))
  run_name = make_run_name(config)
  base_out_dir = output_root / run_name
  base_out_dir.mkdir(parents=True, exist_ok=True)
  overwrite = bool(config.get('overwrite', True))
  transfer = config.get('transfer', {})
  logging.info(f'Output: {base_out_dir}')

  model_args = run_demo.load_args(runtime_overrides(config))
  model = run_demo.load_model(model_args)

  results = []
  for idx, pair in enumerate(pairs, 1):
    sample_out_dir = base_out_dir / pair['name']
    if sample_out_dir.exists() and not overwrite:
      logging.info(f'[{idx}/{len(pairs)}] skip existing: {sample_out_dir}')
      results.append({'name': pair['name'], 'rel': pair['name']})
      continue

    logging.info(f'[{idx}/{len(pairs)}] running: {pair["name"]}')
    sample_args = clone_args(model_args, {
      'left_file': str(pair['left_file']),
      'right_file': str(pair['right_file']),
      'out_dir': str(sample_out_dir),
    })
    run_demo.run_pair(model, sample_args, clean_out_dir=overwrite)
    transfer_outputs(sample_out_dir, f'{run_name}/{pair["name"]}', transfer)
    results.append({'name': pair['name'], 'rel': pair['name']})

  merged_path = merge_point_clouds(base_out_dir, results, fusion, poses)
  if merged_path is not None:
    transfer_outputs(base_out_dir, run_name, transfer)

  write_batch_index(base_out_dir, results)
  logging.info(f'Batch index: {base_out_dir / "index.html"}')


if __name__ == '__main__':
  main()
