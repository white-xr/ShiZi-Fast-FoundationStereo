import cv2
import numpy as np

from triangle_locator.plate_pose import estimate_plate_pose, fit_triangle_vertices
from triangle_locator.stereo_plane import PlaneFitResult


def triangle_mask():
  mask = np.zeros((240, 320), dtype=np.uint8)
  cv2.fillConvexPoly(mask, np.array([[55, 45], [270, 60], [155, 215]], np.int32), 1)
  return mask.astype(bool)


def test_triangle_vertices_are_stable_and_not_mask_centroid():
  result = fit_triangle_vertices(triangle_mask(), {'min_contour_iou': 0.8, 'max_fit_error_px': 2.0})
  assert result.valid
  assert result.vertices_uv.shape == (3, 2)
  assert result.contour_iou > 0.9


def test_tilted_plate_pose_axes_are_orthonormal():
  mask = triangle_mask()
  coefficients = np.array([0.018, -0.011, 31.0])
  empty = np.zeros(mask.shape, dtype=bool)
  plane = PlaneFitResult(True, None, coefficients, np.array([0, 0, 1, -0.4]), mask, mask, mask, None, {})
  K = np.array([[620.0, 0, 160.0], [0, 615.0, 120.0], [0, 0, 1.0]])
  pose = estimate_plate_pose(mask, plane, K, 0.018, {'min_contour_iou': 0.8, 'max_fit_error_px': 2.0})
  assert pose.valid, pose.metrics
  np.testing.assert_allclose(pose.R.T @ pose.R, np.eye(3), atol=1e-7)
  assert np.linalg.det(pose.R) > 0.999
  assert pose.t[2] > 0
  assert abs(pose.R[2, 0]) + abs(pose.R[2, 1]) > 1e-3
