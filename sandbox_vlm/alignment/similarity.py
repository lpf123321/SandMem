"""Configured robust positive-scale alignment; no object identity inference."""
from dataclasses import dataclass

import numpy as np

from sandbox_vlm.memory.geometry import decompose_similarity, transform_points


@dataclass
class AlignmentResult:
    status: str
    T_ref_from_local: np.ndarray | None
    inliers: np.ndarray
    residuals: np.ndarray
    reason: str | None
    diagnostics: dict


def estimate_similarity(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Umeyama least-squares Sim(3), rejecting collinear/constant supports."""
    source, reference = np.asarray(source, dtype=float), np.asarray(reference, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != reference.shape
            or len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(reference).all()):
        raise ValueError("at least three finite paired 3D points required")
    x, y = source - source.mean(axis=0), reference - reference.mean(axis=0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("collinear or constant alignment support")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    scale = float((singular * signs).sum() / (x * x).sum(axis=1).mean())
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = reference.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    decompose_similarity(transform)
    return transform


def robust_similarity(source: np.ndarray, reference: np.ndarray, *, residual_threshold: float,
                      iterations: int, min_inliers: int, min_inlier_ratio: float, seed: int) -> AlignmentResult:
    """RANSAC with three-point hypotheses and deterministic inlier refits."""
    source, reference = np.asarray(source, dtype=float), np.asarray(reference, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != reference.shape
            or not np.isfinite(source).all() or not np.isfinite(reference).all()):
        raise ValueError("finite paired Nx3 points required")
    if (type(iterations) is not int or iterations < 1 or type(min_inliers) is not int or min_inliers < 3
            or type(seed) is not int or not np.isfinite(residual_threshold) or residual_threshold <= 0
            or not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1):
        raise ValueError("invalid explicit RANSAC configuration")
    count = len(source)
    diagnostics = {"paired_points": count, "iterations": iterations, "seed": seed,
                   "residual_threshold": residual_threshold, "min_inliers": min_inliers,
                   "min_inlier_ratio": min_inlier_ratio}

    def failed(reason, inliers=None, residuals=None):
        return AlignmentResult("pending_alignment", None,
            np.zeros(count, dtype=bool) if inliers is None else inliers,
            np.full(count, np.inf) if residuals is None else residuals, reason, diagnostics.copy())

    if count < min_inliers:
        return failed("insufficient_correspondences")
    rng, best_mask, best_score = np.random.default_rng(seed), None, (-1, -np.inf)
    for _ in range(iterations):
        indices = rng.choice(count, 3, replace=False)
        try:
            transform = estimate_similarity(source[indices], reference[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
        mask = residuals <= residual_threshold
        score = (int(mask.sum()), -float(np.median(residuals[mask])) if mask.any() else -np.inf)
        if score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or best_mask.sum() < 3:
        return failed("degenerate_or_no_consensus")
    try:
        transform = estimate_similarity(source[best_mask], reference[best_mask])
        residuals = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
        mask = residuals <= residual_threshold
        if mask.sum() >= 3:
            transform = estimate_similarity(source[mask], reference[mask])
            residuals = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
            mask = residuals <= residual_threshold
    except (ValueError, np.linalg.LinAlgError):
        return failed("degenerate_consensus")
    diagnostics.update(inlier_count=int(mask.sum()), inlier_ratio=float(mask.mean()),
        inlier_median_residual=float(np.median(residuals[mask])) if mask.any() else None)
    if mask.sum() < min_inliers or mask.mean() < min_inlier_ratio:
        return failed("insufficient_consensus", mask, residuals)
    return AlignmentResult("aligned", transform, mask, residuals, None, diagnostics)
