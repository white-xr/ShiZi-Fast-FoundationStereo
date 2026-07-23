from dataclasses import dataclass

import cv2
import numpy as np

from .stereo_plane import intersect_pixels_with_disparity_plane


@dataclass
class TriangleFit:
  valid: bool
  invalid_reason: str | None
  vertices_uv: np.ndarray | None
  contour_iou: float
  fit_error_px: float | None
  method: str | None
  component_count: int = 0
  visible_fraction: float = 0.0
  mask_containment: float = 0.0
  mask_area_px: int = 0
  tip_height_ratio: float | None = None
  tip_side_ratio: float | None = None
  tip_below_px: float | None = None
  geometry_score: float = 0.0


@dataclass
class PlatePose:
  valid: bool
  invalid_reason: str | None
  R: np.ndarray | None
  t: np.ndarray | None
  vertices_uv: np.ndarray | None
  vertices_xyz: np.ndarray | None
  metrics: dict


def _semantic_vertex_mapping(config=None):
  config = config or {}
  mapping = np.asarray(config.get('semantic_vertex_weights', np.eye(3)), dtype=np.float64)
  if mapping.shape != (3, 3) or not np.isfinite(mapping).all():
    raise ValueError('pose.semantic_vertex_weights must be a finite 3x3 matrix')
  if not np.allclose(mapping.sum(axis=1), 1.0, atol=1e-6):
    raise ValueError('each semantic vertex weight row must sum to 1')
  return mapping


def semantic_triangle_vertices(support_vertices, config=None):
  """Map the YOLO support silhouette to the physical A/B/C landmarks."""
  vertices = np.asarray(support_vertices, dtype=np.float64).reshape(3, 2)
  return _semantic_vertex_mapping(config) @ vertices


def support_triangle_vertices(semantic_vertices, config=None):
  """Recover the support silhouette used for mask overlap diagnostics."""
  vertices = np.asarray(semantic_vertices, dtype=np.float64).reshape(3, 2)
  mapping = _semantic_vertex_mapping(config)
  try:
    return np.linalg.solve(mapping, vertices)
  except np.linalg.LinAlgError as exc:
    raise ValueError('pose.semantic_vertex_weights must be invertible') from exc


def _line_from_points(points):
  points = np.asarray(points, dtype=np.float64)
  center = points.mean(axis=0)
  _, _, vh = np.linalg.svd(points - center, full_matrices=False)
  direction = vh[0]
  normal = np.array([-direction[1], direction[0]], dtype=np.float64)
  normal /= np.linalg.norm(normal)
  return np.array([normal[0], normal[1], -normal @ center], dtype=np.float64)


def _intersect_lines(first, second):
  matrix = np.array([[first[0], first[1]], [second[0], second[1]]], dtype=np.float64)
  if abs(np.linalg.det(matrix)) < 1e-7:
    raise ValueError('parallel triangle edges')
  return np.linalg.solve(matrix, -np.array([first[2], second[2]], dtype=np.float64))


def _triangle_iou(mask, vertices):
  iou, _, _ = _triangle_overlap(mask, vertices)
  return iou


def _triangle_overlap(mask, vertices):
  triangle = np.zeros(mask.shape, dtype=np.uint8)
  cv2.fillConvexPoly(triangle, np.round(vertices).astype(np.int32), 1)
  mask_bool = mask.astype(bool)
  triangle_bool = triangle.astype(bool)
  intersection = np.count_nonzero(triangle_bool & mask_bool)
  union = np.count_nonzero(triangle_bool | mask_bool)
  triangle_area = np.count_nonzero(triangle_bool)
  mask_area = np.count_nonzero(mask_bool)
  return (
    float(intersection / union) if union else 0.0,
    float(intersection / triangle_area) if triangle_area else 0.0,
    float(intersection / mask_area) if mask_area else 0.0,
  )


def _meaningful_components(mask, config):
  binary = np.asarray(mask, dtype=np.uint8)
  count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
  if count <= 1:
    return np.zeros(binary.shape, dtype=np.uint8), 0

  components = sorted(
    ((label, int(stats[label, cv2.CC_STAT_AREA])) for label in range(1, count)),
    key=lambda item: item[1],
    reverse=True,
  )
  largest_area = components[0][1]
  min_component_area = max(
    int(config.get('min_component_area_px', 32)),
    int(round(largest_area * float(config.get('min_component_area_ratio', 0.08)))),
  )
  max_components = max(1, int(config.get('max_components', 4)))
  kept = [label for label, area in components if area >= min_component_area][:max_components]
  if not kept:
    return np.zeros(binary.shape, dtype=np.uint8), 0
  return np.isin(labels, kept).astype(np.uint8), len(kept)


def _sample_hull_edges(hull):
  vertices = hull.reshape(-1, 2).astype(np.float64)
  samples = []
  for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
    count = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
    samples.append(np.linspace(start, end, count, endpoint=False))
  return np.vstack(samples)


def _distance_to_triangle_edges(points, vertices):
  lines = [
    _line_from_points(np.vstack((vertices[index], vertices[(index + 1) % 3])))
    for index in range(3)
  ]
  distances = np.column_stack([
    np.abs(points @ line[:2] + line[2]) for line in lines
  ])
  return float(np.mean(np.min(distances, axis=1)))


def _edge_fit(hull_points, seed_vertices):
  lines = []
  distances = []
  seed_lines = []
  for index in range(3):
    start = seed_vertices[index]
    end = seed_vertices[(index + 1) % 3]
    seed_lines.append(_line_from_points(np.vstack((start, end))))
  line_distances = np.column_stack([
    np.abs(hull_points @ line[:2] + line[2]) for line in seed_lines
  ])
  assignments = np.argmin(line_distances, axis=1)
  for index in range(3):
    selected = hull_points[assignments == index]
    if len(selected) < 4:
      raise ValueError('not enough hull points on triangle edge')
    line = _line_from_points(selected)
    lines.append(line)
    distances.extend(np.abs(selected @ line[:2] + line[2]).tolist())
  vertices = np.vstack([
    _intersect_lines(lines[(index - 1) % 3], lines[index]) for index in range(3)
  ])
  return vertices, float(np.mean(distances))


def _seed_triangle(hull):
  perimeter = cv2.arcLength(hull, True)
  for epsilon in (0.01, 0.015, 0.02, 0.03, 0.05):
    approx = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
    if len(approx) == 3:
      return approx.astype(np.float64), 'hull_lines'
  area, triangle = cv2.minEnclosingTriangle(hull)
  if triangle is None or area <= 0:
    raise ValueError('cannot initialize triangle')
  return triangle.reshape(3, 2).astype(np.float64), 'min_enclosing_fallback'


def triangle_geometry_metrics(vertices, config=None):
  config = config or {}
  vertices = np.asarray(vertices, dtype=np.float64)
  if vertices.shape != (3, 2) or not np.isfinite(vertices).all():
    return {
      'valid': False,
      'vertices': vertices,
      'base_length_px': 0.0,
      'tip_height_ratio': None,
      'tip_side_ratio': None,
      'tip_below_px': None,
      'geometry_score': 0.0,
    }

  # The plate is physically mounted with its tip down in the left RGB image.
  # Choose C from that invariant first; perspective can make either side longer
  # than the top edge, so edge length is not a stable semantic label.
  c_index = max(range(3), key=lambda index: (vertices[index, 1], -vertices[index, 0]))
  base_indices = [index for index in range(3) if index != c_index]
  A, B = sorted((vertices[base_indices[0]], vertices[base_indices[1]]), key=lambda point: (point[0], point[1]))
  C = vertices[c_index]
  ordered = np.vstack((A, B, C))

  base = B - A
  base_length = float(np.linalg.norm(base))
  if base_length <= 1e-9:
    return {
      'valid': False,
      'vertices': ordered,
      'base_length_px': base_length,
      'tip_height_ratio': None,
      'tip_side_ratio': None,
      'tip_below_px': None,
      'geometry_score': 0.0,
    }
  base_unit = base / base_length
  down_normal = np.array([-base_unit[1], base_unit[0]], dtype=np.float64)
  midpoint = (A + B) / 2.0
  tip_delta = C - midpoint
  tip_height_ratio = float(np.dot(tip_delta, down_normal) / base_length)
  tip_side_ratio = float(np.dot(tip_delta, base_unit) / base_length)
  tip_below_px = float(C[1] - max(A[1], B[1]))

  min_height = float(config.get('min_tip_height_ratio', 0.20))
  max_height = float(config.get('max_tip_height_ratio', 1.60))
  max_side = float(config.get('max_tip_side_ratio', 0.60))
  min_below = float(config.get('min_tip_below_px', 1.0))
  min_base = float(config.get('min_base_length_px', 12.0))
  valid = (
    base_length >= min_base
    and tip_below_px >= min_below
    and min_height <= tip_height_ratio <= max_height
    and abs(tip_side_ratio) <= max_side
  )
  height_score = float(np.clip(
    min(tip_height_ratio / max(min_height, 1e-6), max_height / max(tip_height_ratio, 1e-6)),
    0.0,
    1.0,
  ))
  side_score = float(np.clip(1.0 - abs(tip_side_ratio) / max(max_side, 1e-6), 0.0, 1.0))
  below_score = float(np.clip(tip_below_px / max(min_below * 4.0, 1.0), 0.0, 1.0))
  return {
    'valid': bool(valid),
    'vertices': ordered,
    'base_length_px': base_length,
    'tip_height_ratio': tip_height_ratio,
    'tip_side_ratio': tip_side_ratio,
    'tip_below_px': tip_below_px,
    'geometry_score': height_score * side_score * below_score,
  }


def fit_triangle_vertices(mask, config=None):
  config = config or {}
  mask, component_count = _meaningful_components(mask, config)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not contours:
    return TriangleFit(False, 'TRIANGLE_FIT_FAILED', None, 0.0, None, None)
  min_area = float(config.get('min_contour_area_px', 200.0))
  if int(mask.sum()) < min_area:
    return TriangleFit(False, 'TRIANGLE_FIT_FAILED', None, 0.0, None, None)
  # A single YOLO instance may contain two visible regions when the screwdriver
  # occludes the plate.  Their joint hull reconstructs the hidden center while
  # the component filter above prevents small mask artifacts from steering it.
  hull = cv2.convexHull(np.vstack(contours))
  hull_points = hull.reshape(-1, 2).astype(np.float64)
  try:
    seed, method = _seed_triangle(hull)
    if method == 'hull_lines':
      vertices, fit_error = _edge_fit(hull_points, seed)
    else:
      try:
        vertices, fit_error = _edge_fit(_sample_hull_edges(hull), seed)
        method = 'occlusion_hull_lines'
      except (ValueError, np.linalg.LinAlgError):
        vertices = seed
        fit_error = _distance_to_triangle_edges(hull_points, seed)
    if component_count > 1:
      method = f'multi_component_{method}'
  except (ValueError, np.linalg.LinAlgError, cv2.error):
    return TriangleFit(False, 'TRIANGLE_FIT_FAILED', None, 0.0, None, None)
  geometry = triangle_geometry_metrics(vertices, config)
  vertices = geometry['vertices']
  iou, visible_fraction, mask_containment = _triangle_overlap(mask, vertices)
  min_iou = float(config.get('min_contour_iou', 0.45))
  if component_count > 1:
    min_iou = float(config.get('min_occluded_contour_iou', min_iou))
  contour_valid = (
    np.isfinite(vertices).all()
    and iou >= min_iou
    and visible_fraction >= float(config.get('min_visible_fraction', 0.25))
    and mask_containment >= float(config.get('min_mask_containment', 0.85))
    and fit_error <= float(config.get('max_fit_error_px', 6.0))
  )
  valid = contour_valid and geometry['valid']
  invalid_reason = None if valid else (
    'TRIANGLE_DIRECTION_INVALID' if contour_valid and not geometry['valid'] else 'TRIANGLE_FIT_FAILED'
  )
  return TriangleFit(
    valid,
    invalid_reason,
    vertices,
    iou,
    fit_error,
    method,
    component_count,
    visible_fraction,
    mask_containment,
    int(mask.sum()),
    geometry['tip_height_ratio'],
    geometry['tip_side_ratio'],
    geometry['tip_below_px'],
    geometry['geometry_score'],
  )


def estimate_pose_from_vertices(
  vertices_uv,
  plane_fit,
  K,
  baseline_m,
  config=None,
  previous_rotation=None,
  metrics=None,
  stereo_geometry=None,
):
  config = config or {}
  metrics = dict(metrics or {})
  geometry = triangle_geometry_metrics(vertices_uv, config)
  vertices_uv = geometry['vertices']
  metrics.update({
    'base_length_px': geometry['base_length_px'],
    'tip_height_ratio': geometry['tip_height_ratio'],
    'tip_side_ratio': geometry['tip_side_ratio'],
    'tip_below_px': geometry['tip_below_px'],
    'geometry_score': geometry['geometry_score'],
  })
  if not geometry['valid']:
    return PlatePose(False, 'TRIANGLE_DIRECTION_INVALID', None, None, vertices_uv, None, metrics)
  try:
    points = intersect_pixels_with_disparity_plane(
      vertices_uv,
      plane_fit.coefficients,
      K,
      baseline_m,
      stereo_geometry=stereo_geometry,
    )
  except ValueError:
    return PlatePose(False, 'TRIANGLE_FIT_FAILED', None, None, vertices_uv, None, metrics)

  def axes(vertices_xyz):
    A, B, C = vertices_xyz
    origin = (A + B + C) / 3.0
    x_axis = B - A
    x_axis /= np.linalg.norm(x_axis)
    y_axis = C - (A + B) / 2.0
    y_axis -= x_axis * np.dot(y_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 1e-9:
      raise ValueError('degenerate triangle axes')
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return origin, np.column_stack((x_axis, y_axis, z_axis))

  try:
    origin, rotation = axes(points)
  except (ValueError, FloatingPointError):
    return PlatePose(False, 'TRIANGLE_FIT_FAILED', None, None, vertices_uv, points, metrics)

  orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
  metrics['orthogonality_error'] = orthogonality_error
  if previous_rotation is not None:
    normal_dot = float(np.dot(np.asarray(previous_rotation)[:, 2], rotation[:, 2]))
    metrics['previous_normal_dot'] = normal_dot
    if normal_dot < float(config.get('min_previous_normal_dot', 0.0)):
      return PlatePose(False, 'POSE_DISCONTINUITY', None, None, vertices_uv, points, metrics)
  if orthogonality_error > float(config.get('max_orthogonality_error', 1e-5)):
    return PlatePose(False, 'TRIANGLE_FIT_FAILED', None, None, vertices_uv, points, metrics)
  return PlatePose(True, None, rotation, origin, vertices_uv, points, metrics)


def estimate_plate_pose(
  mask,
  plane_fit,
  K,
  baseline_m,
  config=None,
  previous_rotation=None,
  stereo_geometry=None,
):
  config = config or {}
  triangle = fit_triangle_vertices(mask, config)
  metrics = {
    'triangle_iou': triangle.contour_iou,
    'triangle_fit_error_px': triangle.fit_error_px,
    'triangle_method': triangle.method,
    'triangle_component_count': triangle.component_count,
    'triangle_visible_fraction': triangle.visible_fraction,
    'triangle_mask_containment': triangle.mask_containment,
    'triangle_mask_area_px': triangle.mask_area_px,
    'tip_height_ratio': triangle.tip_height_ratio,
    'tip_side_ratio': triangle.tip_side_ratio,
    'tip_below_px': triangle.tip_below_px,
    'geometry_score': triangle.geometry_score,
    'pose_source': 'DIRECT',
  }
  if not triangle.valid:
    return PlatePose(False, triangle.invalid_reason, None, None, triangle.vertices_uv, None, metrics)
  try:
    semantic_vertices = semantic_triangle_vertices(triangle.vertices_uv, config)
  except ValueError:
    return PlatePose(False, 'TRIANGLE_FIT_FAILED', None, None, triangle.vertices_uv, None, metrics)
  metrics['support_vertices_uv'] = triangle.vertices_uv
  metrics['semantic_vertex_mapping_enabled'] = not np.allclose(
    _semantic_vertex_mapping(config), np.eye(3), atol=1e-9,
  )
  return estimate_pose_from_vertices(
    semantic_vertices,
    plane_fit,
    K,
    baseline_m,
    config,
    previous_rotation,
    metrics,
    stereo_geometry,
  )
