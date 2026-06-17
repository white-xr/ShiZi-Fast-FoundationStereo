# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys, shutil, html
from pathlib import Path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio.v2 as imageio, logging, yaml
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default='/home/bowen/debug/stereo_output', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=0, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--scale', default=1, type=float)
  parser.add_argument('--hiera', default=0, type=int)
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
  parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
  parser.add_argument('--low_memory', type=int, default=0, help='use lower-memory correlation sampling')
  parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")
  parser.add_argument('--show', type=int, default=1, help='show cv2/open3d windows when DISPLAY is available')
  return parser


def load_args(overrides=None):
  overrides = overrides or {}
  defaults = vars(create_parser().parse_args([]))
  merged = defaults.copy()
  merged.update({k: v for k, v in overrides.items() if v is not None})

  with open(f'{os.path.dirname(merged["model_dir"])}/cfg.yaml', 'r') as ff:
    cfg: dict = yaml.safe_load(ff)
  cfg.update(merged)
  return OmegaConf.create(cfg)


def configure_runtime():
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)


def load_model(args):
  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  if isinstance(model, dict):
    raise TypeError(
      f'{args.model_dir} 是 checkpoint 字典，不是可直接推理的序列化模型。'
      '请把 model_dir 改成 *_serialize.pth，例如 weights/15-44-51/model_best_bp2_serialize.pth。'
    )
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp
  model.args.low_memory = bool(args.low_memory)
  try:
    has_normalize = 'normalize' in model.args
  except TypeError:
    has_normalize = hasattr(model.args, 'normalize')
  if not has_normalize:
    model.args.normalize = True

  model.cuda().eval()
  return model


def read_image(path):
  img = imageio.imread(path)
  if len(img.shape) == 2:
    img = np.tile(img[..., None], (1, 1, 3))
  return img[..., :3]


def save_depth_outputs(depth, out_dir, zfar):
  out_dir = Path(out_dir)
  np.save(out_dir / 'depth_meter.npy', depth)

  valid = np.isfinite(depth) & (depth > 0)
  if zfar and zfar > 0:
    valid &= depth <= zfar

  depth_mm = np.zeros(depth.shape, dtype=np.uint16)
  depth_mm[valid] = np.clip(depth[valid] * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
  imageio.imwrite(out_dir / 'depth_mm.png', depth_mm)

  vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
  if valid.any():
    valid_depth = depth[valid]
    min_val, max_val = np.percentile(valid_depth, [2, 98])
    if max_val <= min_val:
      min_val, max_val = valid_depth.min(), valid_depth.max()
    if max_val > min_val:
      scaled = ((depth - min_val) / (max_val - min_val)).clip(0, 1)
      vis = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
      vis[~valid] = 0
  imageio.imwrite(out_dir / 'depth_vis.png', vis)


def write_html_report(out_dir, title='Stereo result'):
  out_dir = Path(out_dir)
  rows = [
    ('Left', 'left.png'),
    ('Right', 'right.png'),
    ('Disparity', 'disp_vis.png'),
    ('Depth', 'depth_vis.png'),
  ]
  cards = []
  for label, filename in rows:
    if (out_dir / filename).exists():
      cards.append(
        f'<section><h2>{html.escape(label)}</h2>'
        f'<img src="{html.escape(filename)}" alt="{html.escape(label)}"></section>'
      )

  links = []
  for filename in ['disp.npy', 'depth_meter.npy', 'depth_mm.png', 'cloud.ply', 'cloud_denoise.ply']:
    if (out_dir / filename).exists():
      links.append(f'<li><a href="{html.escape(filename)}">{html.escape(filename)}</a></li>')

  page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f7f7f5; color: #202124; }}
    h1 {{ margin: 0 0 18px; font-size: 24px; }}
    h2 {{ margin: 0 0 8px; font-size: 15px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    section {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    img {{ width: 100%; height: auto; display: block; background: #111; }}
    ul {{ margin-top: 18px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <main>{''.join(cards)}</main>
  <ul>{''.join(links)}</ul>
</body>
</html>
"""
  (out_dir / 'index.html').write_text(page, encoding='utf-8')


def maybe_show_disparity(vis, args):
  if not args.show:
    return
  if not os.environ.get('DISPLAY'):
    logging.info("No DISPLAY, skipping cv2.imshow")
    return
  s = 1280 / vis.shape[1]
  resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
  cv2.imshow('disp', resized_vis[:, :, ::-1])
  cv2.waitKey(0)


def maybe_show_point_cloud(pcd, args):
  if not args.show:
    return
  if not os.environ.get('DISPLAY'):
    logging.info("No DISPLAY, skipping point cloud visualization")
    return
  if len(np.asarray(pcd.points)) == 0:
    logging.info("Point cloud is empty, skipping visualization")
    return

  logging.info("Visualizing point cloud. Press ESC to exit.")
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(pcd)
  vis.get_render_option().point_size = 1.0
  vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
  ctr = vis.get_view_control()
  ctr.set_front([0, 0, -1])
  idx = np.asarray(pcd.points)[:, 2].argmin()
  ctr.set_lookat(np.asarray(pcd.points)[idx])
  ctr.set_up([0, -1, 0])
  vis.run()
  vis.destroy_window()


def run_pair(model, args, clean_out_dir=True):
  out_dir = Path(args.out_dir)
  if clean_out_dir and out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  logging.info(f"args:\n{args}")

  scale = args.scale

  img0 = read_image(args.left_file)
  img1 = read_image(args.right_file)

  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f"img0: {img0.shape}")
  imageio.imwrite(out_dir / 'left.png', img0)
  imageio.imwrite(out_dir / 'right.png', img1)

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  logging.info(f"Start forward, 1st time run can be slow due to compilation")
  with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  logging.info("forward done")
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W).clip(0, None)
  np.save(out_dir / 'disp.npy', disp)

  cmap = None
  min_val = None
  max_val = None
  vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(out_dir / 'disp_vis.png', vis)
  maybe_show_disparity(vis, args)

  disp_for_depth = disp.copy()
  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp_for_depth[invalid] = np.inf

  depth = None
  K = None
  if args.intrinsic_file:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = np.zeros_like(disp_for_depth, dtype=np.float32)
    valid_disp = np.isfinite(disp_for_depth) & (disp_for_depth > 1e-6)
    depth[valid_disp] = K[0,0] * baseline / disp_for_depth[valid_disp]
    save_depth_outputs(depth, out_dir, args.zfar)

  if args.get_pc:
    if depth is None or K is None:
      logging.info("No intrinsic_file, skipping point cloud output")
      write_html_report(out_dir, title=Path(args.left_file).stem)
      return
    if o3d is None:
      logging.info("open3d is not installed, skipping point cloud output")
      write_html_report(out_dir, title=Path(args.left_file).stem)
      return
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(str(out_dir / 'cloud.ply'), pcd)
    logging.info(f"PCL saved to {out_dir}")

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      pcd = pcd.voxel_down_sample(voxel_size=0.001)
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(str(out_dir / 'cloud_denoise.ply'), inlier_cloud)
      pcd = inlier_cloud

    maybe_show_point_cloud(pcd, args)
  write_html_report(out_dir, title=Path(args.left_file).stem)


def main(argv=None):
  cli_args = create_parser().parse_args(argv)
  configure_runtime()
  args = load_args(vars(cli_args))
  model = load_model(args)
  run_pair(model, args, clean_out_dir=True)


if __name__=="__main__":
  main()
