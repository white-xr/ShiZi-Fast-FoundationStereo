import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def enhance_gradient(img, alpha=0.85):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
  grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
  grad = cv2.magnitude(grad_x, grad_y)
  grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
  grad_3ch = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
  return cv2.addWeighted(img, alpha, grad_3ch, 1.0 - alpha, 0)


def enhance_clahe(img):
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l = clahe.apply(l)
  return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_image(img, mode='both', alpha=0.85):
  if img is None:
    raise ValueError('img must not be None')
  if img.dtype != np.uint8:
    raise ValueError('img must be a uint8 BGR image')

  if mode == 'gradient':
    return enhance_gradient(img, alpha=alpha)
  if mode == 'clahe':
    return enhance_clahe(img)
  if mode == 'both':
    return enhance_gradient(enhance_clahe(img), alpha=alpha)
  raise ValueError(f'Unsupported mode: {mode}')


def parse_args():
  parser = argparse.ArgumentParser(description='Enhance image texture before stereo inference.')
  parser.add_argument('--input', required=True, help='input image path or directory')
  parser.add_argument('--output', required=True, help='output image path or directory')
  parser.add_argument('--mode', choices=['gradient', 'clahe', 'both'], default='both')
  parser.add_argument('--alpha', type=float, default=0.85, help='original image weight for gradient mode')
  parser.add_argument('--preview', action='store_true', help='show original/enhanced comparison with OpenCV')
  return parser.parse_args()


def iter_images(input_path):
  if input_path.is_file():
    return [input_path]
  return sorted(path for path in input_path.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def output_path_for(input_path, output_path, image_path):
  if input_path.is_file():
    return output_path
  return output_path / f'{image_path.stem}.png'


def process_image(image_path, output_path, mode, alpha, preview):
  img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
  if img is None:
    raise RuntimeError(f'Failed to read image: {image_path}')

  enhanced = enhance_image(img, mode=mode, alpha=alpha)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if not cv2.imwrite(str(output_path), enhanced):
    raise RuntimeError(f'Failed to write image: {output_path}')

  if preview:
    if not os.environ.get('DISPLAY'):
      print('No DISPLAY, skipping preview')
      return
    comparison = np.hstack((img, enhanced))
    cv2.imshow('original | enhanced', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
  args = parse_args()
  input_path = Path(args.input)
  output_path = Path(args.output)

  if not input_path.exists():
    raise FileNotFoundError(f'Input not found: {input_path}')
  if input_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

  images = iter_images(input_path)
  if not images:
    raise RuntimeError(f'No images found: {input_path}')

  for image_path in images:
    out_path = output_path_for(input_path, output_path, image_path)
    process_image(image_path, out_path, args.mode, args.alpha, args.preview)
    print(f'{image_path} -> {out_path}')


if __name__ == '__main__':
  main()
