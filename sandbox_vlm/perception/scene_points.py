"""Deterministic reliable samples of native shared-coordinate point maps."""
import numpy as np


def reliable_scene_points(world_points, confidence, *, confidence_quantile: float,
                          grid_stride: int, max_points: int, min_points: int):
    points = np.asarray(world_points)
    conf = np.asarray(confidence)
    if points.ndim != 3 or points.shape[-1] != 3 or conf.shape != points.shape[:2]:
        raise ValueError('expected shared-reference HxWx3 points and HxW confidence')
    if (not 0 <= confidence_quantile <= 1 or type(grid_stride) is not int or grid_stride < 1
            or type(max_points) is not int or type(min_points) is not int
            or not 1 <= min_points <= max_points):
        raise ValueError('invalid reliable point sampling settings')
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf > 0)
    if not valid.any():
        return np.empty((0, 3)), dict(status='no_finite_points', selected_points=0)
    threshold = float(np.quantile(conf[valid], confidence_quantile))
    grid = np.zeros_like(valid)
    grid[::grid_stride, ::grid_stride] = True
    rows = points[valid & grid & (conf >= threshold)]
    if len(rows) > max_points:
        rows = rows[np.linspace(0, len(rows) - 1, max_points).astype(np.int64)]
    count = len(rows)
    enough = count >= min_points
    return (np.array(rows, dtype=np.float64, copy=True) if enough else np.empty((0, 3))), dict(
        status='reliable' if enough else 'insufficient_points', selected_points=count,
        confidence_threshold=threshold)


def scale_fixed_voxel_size(points, *, diagonal_fraction: float):
    """Called once on the first reliable geometry in a reconstruction.

    Uses arbitrary model units, not metres. Reject a zero initial diagonal;
    no invented metric fallback. Caller keeps null assignment until usable.
    """
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(diagonal_fraction)
            or diagonal_fraction <= 0):
        raise ValueError('invalid initial geometry or relative voxel size')
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diagonal <= np.finfo(np.float64).eps:
        raise ValueError('initial geometry has no usable spatial scale')
    return diagonal * diagonal_fraction
