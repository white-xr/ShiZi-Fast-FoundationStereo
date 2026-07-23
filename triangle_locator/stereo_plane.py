from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlaneFitResult:
  valid: bool
  invalid_reason: str | None
  coefficients: np.ndarray | None
  camera_plane: np.ndarray | None
  eroded_mask: np.ndarray
  valid_mask: np.ndarray
  inlier_mask: np.ndarray
  fitted_disparity: np.ndarray | None
  metrics: dict


def _stereo_value(stereo_geometry, key):
  if isinstance(stereo_geometry, dict):
    return stereo_geometry[key]
  return getattr(stereo_geometry, key)


def _raw_stereo_matrices(stereo_geometry):
  K1 = np.asarray(_stereo_value(stereo_geometry, 'K1'), dtype=np.float64).reshape(3, 3)
  K2 = np.asarray(_stereo_value(stereo_geometry, 'K2'), dtype=np.float64).reshape(3, 3)
  R = np.asarray(_stereo_value(stereo_geometry, 'R'), dtype=np.float64).reshape(3, 3)
  T = np.asarray(_stereo_value(stereo_geometry, 'T'), dtype=np.float64).reshape(3)
  P1 = K1 @ np.column_stack((np.eye(3), np.zeros(3, dtype=np.float64)))
  P2 = K2 @ np.column_stack((R, T))
  skew_t = np.array([
    [0.0, -T[2], T[1]],
    [T[2], 0.0, -T[0]],
    [-T[1], T[0], 0.0],
  ], dtype=np.float64)
  fundamental = np.linalg.inv(K2).T @ skew_t @ R @ np.linalg.inv(K1)
  return P1, P2, fundamental, R, T


def triangulate_pixels_with_disparity(pixels, disparity, stereo_geometry):
  """Triangulate horizontal FFS correspondences in unrectified camera geometry."""
  pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
  disparity = np.asarray(disparity, dtype=np.float64).reshape(-1)
  if len(pixels) != len(disparity):
    raise ValueError('pixels and disparity must contain the same number of samples')
  P1, P2, fundamental, R, T = _raw_stereo_matrices(stereo_geometry)
  right_u = pixels[:, 0] - disparity
  left_h = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64)))
  right_lines = (fundamental @ left_h.T).T
  right_v = pixels[:, 1].copy()
  stable = np.abs(right_lines[:, 1]) > 1e-12
  right_v[stable] = -(
    right_lines[stable, 0] * right_u[stable] + right_lines[stable, 2]
  ) / right_lines[stable, 1]
  right_pixels = np.column_stack((right_u, right_v))
  homogeneous = cv2.triangulatePoints(P1, P2, pixels.T, right_pixels.T)
  w = homogeneous[3]
  points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
  finite = np.isfinite(homogeneous).all(axis=0) & (np.abs(w) > 1e-12)
  points[finite] = (homogeneous[:3, finite] / w[finite]).T
  right_points = (R @ points.T).T + T
  cheirality = finite & (points[:, 2] > 0) & (right_points[:, 2] > 0)
  points[~cheirality] = np.nan
  return points


def _fit_camera_plane(points):
  points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
  points = points[np.isfinite(points).all(axis=1)]
  if len(points) < 3:
    raise ValueError('not enough finite 3D points for a camera plane')
  center = points.mean(axis=0)
  _, _, vh = np.linalg.svd(points - center, full_matrices=False)
  normal = vh[-1]
  if normal[2] < 0:
    normal = -normal
  normal /= np.linalg.norm(normal)
  return np.array([*normal, -float(normal @ center)], dtype=np.float64)


def erode_mask(mask, pixels=0, relative=0.0):
  mask = np.asarray(mask, dtype=bool)
  relative_pixels = int(round(min(mask.shape) * float(relative or 0.0)))
  radius = max(int(pixels or 0), relative_pixels)
  if radius <= 0:
    return mask.copy()
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
  return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _solve_weighted(design, values, weights=None):
  if weights is None:
    return np.linalg.lstsq(design, values, rcond=None)[0]
  root = np.sqrt(np.asarray(weights, dtype=np.float64))
  return np.linalg.lstsq(design * root[:, None], values * root, rcond=None)[0]


def _huber_refine(design, values, initial, delta, iterations):
  coefficients = np.asarray(initial, dtype=np.float64)
  for _ in range(int(iterations)):
    residuals = values - design @ coefficients
    scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals))) + 1e-9
    normalized = np.abs(residuals) / scale
    weights = np.ones_like(normalized)
    large = normalized > float(delta)
    weights[large] = float(delta) / normalized[large]
    updated = _solve_weighted(design, values, weights)
    if np.linalg.norm(updated - coefficients) < 1e-9:
      coefficients = updated
      break
    coefficients = updated
  return coefficients


def _spatial_coverage(mask, inliers, grid_rows, grid_cols, min_cell_points):
  height, width = mask.shape
  eligible = 0
  covered = 0
  for row in range(int(grid_rows)):
    y0 = row * height // int(grid_rows)
    y1 = (row + 1) * height // int(grid_rows)
    for col in range(int(grid_cols)):
      x0 = col * width // int(grid_cols)
      x1 = (col + 1) * width // int(grid_cols)
      if np.count_nonzero(mask[y0:y1, x0:x1]) < int(min_cell_points):
        continue
      eligible += 1
      if np.count_nonzero(inliers[y0:y1, x0:x1]) >= int(min_cell_points):
        covered += 1
  return float(covered / eligible) if eligible else 0.0


def disparity_plane_to_camera(coefficients, K, baseline_m):
  a, b, c = np.asarray(coefficients, dtype=np.float64)
  K = np.asarray(K, dtype=np.float64)
  normal = np.array([
    a * K[0, 0],
    b * K[1, 1],
    a * K[0, 2] + b * K[1, 2] + c,
  ], dtype=np.float64)
  rhs = float(K[0, 0] * baseline_m)
  norm = np.linalg.norm(normal)
  if norm <= 1e-12:
    raise ValueError('degenerate disparity plane')
  return np.array([*(normal / norm), -rhs / norm], dtype=np.float64)


def fit_disparity_plane(
  disparity,
  mask,
  K,
  baseline_m,
  config=None,
  validity_mask=None,
  stereo_geometry=None,
):
  config = config or {}
  disparity = np.asarray(disparity, dtype=np.float64)
  eroded = erode_mask(
    mask,
    pixels=int(config.get('erode_px', 3)),
    relative=float(config.get('erode_relative', 0.0)),
  )
  mask_pixels = int(np.count_nonzero(eroded))
  valid = eroded & np.isfinite(disparity)
  valid &= disparity > float(config.get('min_disp', 0.1))
  if validity_mask is not None:
    valid &= np.asarray(validity_mask, dtype=bool)
  max_disp = config.get('max_disp')
  if max_disp not in {None, '', 0}:
    valid &= disparity <= float(max_disp)

  depth = np.zeros_like(disparity)
  if stereo_geometry is None:
    depth[valid] = float(K[0, 0]) * float(baseline_m) / disparity[valid]
  else:
    candidate_y, candidate_x = np.where(valid)
    candidate_points = triangulate_pixels_with_disparity(
      np.column_stack((candidate_x, candidate_y)),
      disparity[valid],
      stereo_geometry,
    )
    candidate_depth = candidate_points[:, 2]
    finite_depth = np.isfinite(candidate_depth) & (candidate_depth > 0)
    depth[candidate_y[finite_depth], candidate_x[finite_depth]] = candidate_depth[finite_depth]
    valid[candidate_y[~finite_depth], candidate_x[~finite_depth]] = False
  min_depth = float(config.get('min_depth_m', 0.0) or 0.0)
  max_depth = float(config.get('max_depth_m', 0.0) or 0.0)
  if min_depth > 0:
    valid &= depth >= min_depth
  if max_depth > 0:
    valid &= depth <= max_depth

  valid_pixels = int(np.count_nonzero(valid))
  valid_ratio = float(valid_pixels / mask_pixels) if mask_pixels else 0.0
  empty_inliers = np.zeros(mask.shape, dtype=bool)
  base_metrics = {
    'mask_pixels': mask_pixels,
    'valid_pixels': valid_pixels,
    'valid_ratio': valid_ratio,
    'inlier_ratio': 0.0,
    'disparity_rmse_px': None,
    'depth_rmse_mm': None,
    'spatial_coverage': 0.0,
    'quality_score': 0.0,
  }
  min_points = int(config.get('min_points', 200))
  if mask_pixels == 0 or valid_pixels < min_points:
    return PlaneFitResult(False, 'INSUFFICIENT_DISPARITY', None, None, eroded, valid, empty_inliers, None, base_metrics)

  ys, xs = np.where(valid)
  values = disparity[valid]
  design = np.column_stack((xs, ys, np.ones(valid_pixels, dtype=np.float64)))
  threshold = float(config.get('ransac_threshold_px', 0.8))
  rng = np.random.default_rng(int(config.get('random_seed', 0)))
  best = None
  best_count = 0
  for _ in range(int(config.get('ransac_iterations', 300))):
    ids = rng.choice(valid_pixels, size=3, replace=False)
    sample = design[ids]
    if abs(np.linalg.det(sample)) < 1e-8:
      continue
    candidate = np.linalg.solve(sample, values[ids])
    candidate_inliers = np.abs(values - design @ candidate) <= threshold
    count = int(np.count_nonzero(candidate_inliers))
    if count > best_count:
      best = candidate
      best_count = count
  if best is None or best_count < min_points:
    return PlaneFitResult(False, 'PLANE_FIT_FAILED', None, None, eroded, valid, empty_inliers, None, base_metrics)

  initial_inliers = np.abs(values - design @ best) <= threshold
  refined = _solve_weighted(design[initial_inliers], values[initial_inliers])
  refined = _huber_refine(
    design[initial_inliers],
    values[initial_inliers],
    refined,
    delta=float(config.get('huber_delta', 1.5)),
    iterations=int(config.get('irls_iterations', 10)),
  )
  residuals = values - design @ refined
  inlier_values = np.abs(residuals) <= threshold
  inlier_count = int(np.count_nonzero(inlier_values))
  inlier_ratio = float(inlier_count / valid_pixels)
  inlier_mask = np.zeros(mask.shape, dtype=bool)
  inlier_mask[ys[inlier_values], xs[inlier_values]] = True

  disparity_rmse = float(np.sqrt(np.mean(np.square(residuals[inlier_values])))) if inlier_count else None
  fitted_at_inliers = design[inlier_values] @ refined
  measured_at_inliers = values[inlier_values]
  if stereo_geometry is None:
    depth_errors = float(K[0, 0]) * float(baseline_m) * (
      1.0 / measured_at_inliers - 1.0 / fitted_at_inliers
    )
  else:
    inlier_pixels = np.column_stack((xs[inlier_values], ys[inlier_values]))
    measured_xyz = triangulate_pixels_with_disparity(
      inlier_pixels, measured_at_inliers, stereo_geometry,
    )
    fitted_xyz = triangulate_pixels_with_disparity(
      inlier_pixels, fitted_at_inliers, stereo_geometry,
    )
    depth_errors = measured_xyz[:, 2] - fitted_xyz[:, 2]
    depth_errors = depth_errors[np.isfinite(depth_errors)]
  depth_rmse_mm = float(np.sqrt(np.mean(np.square(depth_errors))) * 1000.0) if len(depth_errors) else None
  coverage = _spatial_coverage(
    eroded,
    inlier_mask,
    int(config.get('grid_rows', 3)),
    int(config.get('grid_cols', 3)),
    int(config.get('min_cell_points', 10)),
  )
  rmse_limit = float(config.get('max_rmse_px', 1.0))
  quality = float(np.clip(
    min(1.0, valid_ratio / max(float(config.get('min_valid_ratio', 0.35)), 1e-6))
    * min(1.0, inlier_ratio / max(float(config.get('min_inlier_ratio', 0.7)), 1e-6))
    * min(1.0, coverage / max(float(config.get('min_spatial_coverage', 0.55)), 1e-6))
    * max(0.0, 1.0 - (disparity_rmse or rmse_limit) / max(rmse_limit * 2.0, 1e-6)),
    0.0,
    1.0,
  ))
  metrics = {
    'mask_pixels': mask_pixels,
    'valid_pixels': valid_pixels,
    'valid_ratio': valid_ratio,
    'inlier_ratio': inlier_ratio,
    'disparity_rmse_px': disparity_rmse,
    'depth_rmse_mm': depth_rmse_mm,
    'spatial_coverage': coverage,
    'quality_score': quality,
  }
  passes = (
    valid_ratio >= float(config.get('min_valid_ratio', 0.35))
    and inlier_ratio >= float(config.get('min_inlier_ratio', 0.7))
    and coverage >= float(config.get('min_spatial_coverage', 0.55))
    and disparity_rmse is not None
    and disparity_rmse <= rmse_limit
  )
  fitted = np.zeros(disparity.shape, dtype=np.float32)
  all_y, all_x = np.indices(disparity.shape)
  fitted[eroded] = (refined[0] * all_x[eroded] + refined[1] * all_y[eroded] + refined[2]).astype(np.float32)
  try:
    if stereo_geometry is None:
      camera_plane = disparity_plane_to_camera(refined, K, baseline_m)
    else:
      plane_pixels = np.column_stack((xs[inlier_values], ys[inlier_values]))
      plane_points = triangulate_pixels_with_disparity(
        plane_pixels,
        fitted_at_inliers,
        stereo_geometry,
      )
      camera_plane = _fit_camera_plane(plane_points)
  except ValueError:
    return PlaneFitResult(False, 'PLANE_FIT_FAILED', refined, None, eroded, valid, inlier_mask, fitted, metrics)
  return PlaneFitResult(passes, None if passes else 'PLANE_QUALITY_LOW', refined, camera_plane, eroded, valid, inlier_mask, fitted, metrics)


def intersect_pixels_with_disparity_plane(
  pixels,
  coefficients,
  K,
  baseline_m,
  stereo_geometry=None,
):
  pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
  coefficients = np.asarray(coefficients, dtype=np.float64)
  K = np.asarray(K, dtype=np.float64)
  disparity = coefficients[0] * pixels[:, 0] + coefficients[1] * pixels[:, 1] + coefficients[2]
  if np.any(~np.isfinite(disparity)) or np.any(disparity <= 0):
    raise ValueError('triangle vertex has invalid fitted disparity')
  if stereo_geometry is not None:
    points = triangulate_pixels_with_disparity(pixels, disparity, stereo_geometry)
    if np.any(~np.isfinite(points)):
      raise ValueError('triangle vertex cannot be triangulated in raw stereo geometry')
    return points
  z = K[0, 0] * float(baseline_m) / disparity
  x = (pixels[:, 0] - K[0, 2]) * z / K[0, 0]
  y = (pixels[:, 1] - K[1, 2]) * z / K[1, 1]
  return np.column_stack((x, y, z))
