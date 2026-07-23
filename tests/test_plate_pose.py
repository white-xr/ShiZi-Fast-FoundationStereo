import cv2
import numpy as np

from triangle_locator.plate_pose import (
  estimate_plate_pose,
  fit_triangle_vertices,
  semantic_triangle_vertices,
  support_triangle_vertices,
  triangle_geometry_metrics,
)
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


def test_semantic_vertex_mapping_is_barycentric_and_reversible():
  support = np.array([[50.0, 40.0], [270.0, 50.0], [150.0, 215.0]])
  mapping = [
    [0.75, 0.15, 0.10],
    [0.10, 0.75, 0.15],
    [0.10, 0.15, 0.75],
  ]
  semantic = semantic_triangle_vertices(support, {'semantic_vertex_weights': mapping})
  np.testing.assert_allclose(semantic, np.asarray(mapping) @ support)
  np.testing.assert_allclose(
    support_triangle_vertices(semantic, {'semantic_vertex_weights': mapping}),
    support,
  )


def test_semantic_vertex_mapping_rejects_non_barycentric_rows():
  with np.testing.assert_raises(ValueError):
    semantic_triangle_vertices(
      np.array([[50.0, 40.0], [270.0, 50.0], [150.0, 215.0]]),
      {'semantic_vertex_weights': [[1, 0, 0], [0, 1, 0], [0, 0, 0.9]]},
    )


def test_downward_tip_is_c_even_when_a_side_is_the_longest_edge():
  mask = np.zeros((240, 320), dtype=np.uint8)
  expected = np.array([[80, 45], [185, 50], [130, 185]], np.int32)
  cv2.fillConvexPoly(mask, expected, 1)
  result = fit_triangle_vertices(mask.astype(bool), {
    'min_contour_iou': 0.8,
    'max_fit_error_px': 2.0,
  })
  assert result.valid, result
  A, B, C = result.vertices_uv
  assert A[0] < B[0]
  assert C[1] > max(A[1], B[1])
  assert np.linalg.norm(C - A) > np.linalg.norm(B - A)
  np.testing.assert_allclose(result.vertices_uv, expected, atol=6.0)


def test_flat_residual_triangle_is_rejected_by_physical_direction_constraint():
  geometry = triangle_geometry_metrics(
    np.array([[40.0, 50.0], [180.0, 53.0], [110.0, 55.0]]),
    {'min_tip_height_ratio': 0.2, 'min_tip_below_px': 1.0},
  )
  assert not geometry['valid']
  assert geometry['tip_height_ratio'] < 0.2


def test_occluded_triangle_uses_all_visible_components():
  mask = triangle_mask()
  mask[:, 145:168] = False
  result = fit_triangle_vertices(mask, {
    'min_contour_iou': 0.8,
    'min_occluded_contour_iou': 0.35,
    'max_fit_error_px': 6.0,
  })
  assert result.valid, result
  assert result.component_count == 2
  assert result.method.startswith('multi_component_')
  assert result.mask_containment > 0.85
  assert result.visible_fraction > 0.35
  expected = np.array([[55, 45], [270, 60], [155, 215]], dtype=np.float64)
  np.testing.assert_allclose(result.vertices_uv, expected, atol=18.0)


def test_small_detached_mask_noise_does_not_change_triangle_fit():
  mask = triangle_mask()
  mask[5:8, 5:8] = True
  result = fit_triangle_vertices(mask, {
    'min_contour_iou': 0.8,
    'max_fit_error_px': 2.0,
    'min_component_area_px': 32,
  })
  assert result.valid
  assert result.component_count == 1
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
  assert pose.vertices_uv[0, 0] < pose.vertices_uv[1, 0]
  assert np.dot(pose.R[:, 0], pose.vertices_xyz[1] - pose.vertices_xyz[0]) > 0


def test_left_and_right_tilt_keep_triangle_axis_directions():
  mask = triangle_mask()
  empty = np.zeros(mask.shape, dtype=bool)
  K = np.array([[620.0, 0, 160.0], [0, 615.0, 120.0], [0, 0, 1.0]])
  poses = []
  for slope in (-0.025, 0.025):
    plane = PlaneFitResult(
      True,
      None,
      np.array([slope, -0.008, 31.0]),
      np.array([0, 0, 1, -0.4]),
      mask,
      mask,
      mask,
      None,
      {},
    )
    pose = estimate_plate_pose(mask, plane, K, 0.018, {
      'min_contour_iou': 0.8,
      'max_fit_error_px': 2.0,
    })
    assert pose.valid
    poses.append(pose)
  assert np.dot(poses[0].R[:, 0], poses[1].R[:, 0]) > 0
  assert np.dot(poses[0].R[:, 1], poses[1].R[:, 1]) > 0
  assert np.dot(poses[0].R[:, 2], poses[1].R[:, 2]) > 0
