import cv2
import numpy as np
import pytest

from triangle_locator.stereo_plane import erode_mask, fit_disparity_plane, intersect_pixels_with_disparity_plane


def synthetic_plane(seed=4):
  height, width = 180, 240
  yy, xx = np.indices((height, width))
  truth = np.array([0.012, -0.008, 34.0])
  disparity = truth[0] * xx + truth[1] * yy + truth[2]
  mask = np.zeros((height, width), dtype=np.uint8)
  cv2.fillConvexPoly(mask, np.array([[35, 25], [205, 35], [122, 160]], np.int32), 1)
  rng = np.random.default_rng(seed)
  disparity = disparity + rng.normal(0, 0.08, disparity.shape)
  points = np.column_stack(np.where(mask > 0))
  outlier_ids = rng.choice(len(points), size=len(points) // 6, replace=False)
  outliers = points[outlier_ids]
  disparity[outliers[:, 0], outliers[:, 1]] += rng.normal(0, 8.0, len(outliers))
  return disparity.astype(np.float32), mask.astype(bool), truth


def test_ransac_huber_recovers_plane_with_outliers():
  disparity, mask, truth = synthetic_plane()
  K = np.array([[610.0, 0.0, 120.0], [0.0, 610.0, 90.0], [0.0, 0.0, 1.0]])
  result = fit_disparity_plane(disparity, mask, K, 0.018, {
    'erode_px': 2,
    'min_points': 200,
    'ransac_iterations': 500,
    'ransac_threshold_px': 0.5,
    'min_valid_ratio': 0.7,
    'min_inlier_ratio': 0.7,
    'min_spatial_coverage': 0.7,
    'max_rmse_px': 0.25,
  })
  assert result.valid, result.metrics
  np.testing.assert_allclose(result.coefficients, truth, atol=0.02)
  assert result.metrics['inlier_ratio'] > 0.75
  assert result.metrics['disparity_rmse_px'] < 0.2


def test_mask_erosion_and_valid_ratio():
  mask = np.zeros((20, 20), dtype=bool)
  mask[2:18, 2:18] = True
  eroded = erode_mask(mask, pixels=2)
  assert eroded.sum() < mask.sum()
  disparity = np.full(mask.shape, 20.0, dtype=np.float32)
  disparity[eroded] = 0.0
  K = np.array([[500.0, 0, 10], [0, 500.0, 10], [0, 0, 1]])
  result = fit_disparity_plane(disparity, mask, K, 0.02, {'erode_px': 2, 'min_points': 10})
  assert not result.valid
  assert result.invalid_reason == 'INSUFFICIENT_DISPARITY'
  assert result.metrics['valid_ratio'] == 0.0


def test_ffs_validity_mask_excludes_invisible_disparity():
  disparity = np.full((40, 40), 20.0, dtype=np.float32)
  mask = np.ones(disparity.shape, dtype=bool)
  validity = np.zeros(disparity.shape, dtype=bool)
  K = np.array([[500.0, 0, 20], [0, 500.0, 20], [0, 0, 1]])
  result = fit_disparity_plane(
    disparity,
    mask,
    K,
    0.02,
    {'erode_px': 0, 'min_points': 10},
    validity_mask=validity,
  )
  assert not result.valid
  assert result.invalid_reason == 'INSUFFICIENT_DISPARITY'
  assert result.metrics['valid_pixels'] == 0


def test_pixel_rays_intersect_fitted_disparity_plane():
  K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
  coefficients = np.array([0.01, -0.02, 30.0])
  pixels = np.array([[100.0, 80.0], [500.0, 90.0], [310.0, 400.0]])
  points = intersect_pixels_with_disparity_plane(pixels, coefficients, K, 0.018)
  expected_disparity = coefficients[0] * pixels[:, 0] + coefficients[1] * pixels[:, 1] + coefficients[2]
  np.testing.assert_allclose(points[:, 2], K[0, 0] * 0.018 / expected_disparity)
  projected_u = K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2]
  projected_v = K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]
  np.testing.assert_allclose(np.column_stack((projected_u, projected_v)), pixels)
