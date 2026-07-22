import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args():
  parser = argparse.ArgumentParser(description='Calibrate a synchronized stereo pair using a chessboard or ChArUco board.')
  parser.add_argument('--left_glob', required=True)
  parser.add_argument('--right_dir', required=True)
  parser.add_argument('--output', default='configs/stereo_calib.yaml')
  parser.add_argument('--method', choices=['chessboard', 'charuco'], default='chessboard')
  parser.add_argument('--columns', type=int, default=9, help='chessboard inner corners or ChArUco squares')
  parser.add_argument('--rows', type=int, default=6)
  parser.add_argument('--square_m', type=float, required=True)
  parser.add_argument('--marker_m', type=float, default=None)
  parser.add_argument('--dictionary', default='DICT_4X4_50')
  return parser.parse_args()


def chessboard_points(image, columns, rows, object_template):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  found, corners = cv2.findChessboardCorners(gray, (columns, rows))
  if not found:
    return None
  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
  corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
  return object_template.copy(), corners.astype(np.float32)


def charuco_points(left, right, args):
  if args.marker_m is None:
    raise ValueError('--marker_m is required for ChArUco calibration')
  dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
  board = cv2.aruco.CharucoBoard((args.columns, args.rows), args.square_m, args.marker_m, dictionary)

  def detect(image):
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None:
      return None, None
    count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board)
    if count is None or count < 6:
      return None, None
    return charuco_corners, charuco_ids.reshape(-1)

  left_corners, left_ids = detect(left)
  right_corners, right_ids = detect(right)
  if left_ids is None or right_ids is None:
    return None
  common = sorted(set(left_ids.tolist()) & set(right_ids.tolist()))
  if len(common) < 6:
    return None
  board_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
  object_points = board_points[common]
  left_map = {int(value): left_corners[index] for index, value in enumerate(left_ids)}
  right_map = {int(value): right_corners[index] for index, value in enumerate(right_ids)}
  return object_points, np.asarray([left_map[value] for value in common]), np.asarray([right_map[value] for value in common])


def main():
  args = parse_args()
  left_files = [Path(value) for value in sorted(glob.glob(args.left_glob))]
  right_dir = Path(args.right_dir)
  object_points = []
  left_points = []
  right_points = []
  image_size = None
  template = np.zeros((args.columns * args.rows, 3), dtype=np.float32)
  template[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_m
  for left_file in left_files:
    right_file = right_dir / left_file.name
    left = cv2.imread(str(left_file))
    right = cv2.imread(str(right_file))
    if left is None or right is None or left.shape[:2] != right.shape[:2]:
      continue
    image_size = (left.shape[1], left.shape[0])
    if args.method == 'chessboard':
      left_detection = chessboard_points(left, args.columns, args.rows, template)
      right_detection = chessboard_points(right, args.columns, args.rows, template)
      if left_detection is None or right_detection is None:
        continue
      obj, left_corners = left_detection
      _, right_corners = right_detection
    else:
      detection = charuco_points(left, right, args)
      if detection is None:
        continue
      obj, left_corners, right_corners = detection
    object_points.append(obj)
    left_points.append(left_corners)
    right_points.append(right_corners)
  if len(object_points) < 8 or image_size is None:
    raise RuntimeError(f'only {len(object_points)} valid stereo calibration pairs; at least 8 are required')
  _, K1, D1, _, _ = cv2.calibrateCamera(object_points, left_points, image_size, None, None)
  _, K2, D2, _, _ = cv2.calibrateCamera(object_points, right_points, image_size, None, None)
  stereo_error, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
    object_points, left_points, right_points, K1, D1, K2, D2, image_size,
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
  )
  payload = {
    'configured': True,
    'source': f'{args.method} stereo calibration',
    'reprojection_error_px': float(stereo_error),
    'image_width': image_size[0],
    'image_height': image_size[1],
    'K1': K1.tolist(),
    'D1': D1.reshape(-1).tolist(),
    'K2': K2.tolist(),
    'D2': D2.reshape(-1).tolist(),
    'R': R.tolist(),
    'T': T.reshape(3, 1).tolist(),
    'baseline_m': float(np.linalg.norm(T)),
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
  print(f'Wrote {output}; pairs={len(object_points)} stereo_error={stereo_error:.6f}')


if __name__ == '__main__':
  main()
