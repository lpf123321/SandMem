"""Explicit, configurable depth filtering; scores are not probabilities."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ReliableDepth:
    pixels: np.ndarray
    threshold: float | None
    finite_score_count: int
    retained_pixel_count: int


def select_reliable_depth(depth: np.ndarray, confidence: np.ndarray, *, keep_fraction: float) -> ReliableDepth:
    """Clip-wide score quantile, then finite positive depth (decision D009).

    Quantile uses all finite confidence scores in the clip, before masking by
    object. Ties at the threshold are retained, so the fraction can exceed the
    requested one. Empty support returns empty evidence, never an invented score.
    """
    depth, confidence = np.asarray(depth), np.asarray(confidence)
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise ValueError("depth and confidence must share a frames,height,width grid")
    if isinstance(keep_fraction, bool) or not np.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be finite and in (0,1]")
    finite = np.isfinite(confidence)
    scores = confidence[finite]
    if not scores.size:
        return ReliableDepth(np.zeros(depth.shape, dtype=bool), None, 0, 0)
    threshold = float(np.quantile(scores, 1 - keep_fraction, method="linear"))
    pixels = finite & (confidence >= threshold) & np.isfinite(depth) & (depth > 0)
    return ReliableDepth(pixels, threshold, int(scores.size), int(pixels.sum()))
