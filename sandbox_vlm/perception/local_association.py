"""Conservative local mask matching; never declares a new physical instance."""
from dataclasses import dataclass

import numpy as np


@dataclass
class LocalMatch:
    status: str
    matched_track_id: str | None
    candidate_track_ids: list[str]
    mask_ious: dict[str, float]


def match_local_mask(prompt_mask: np.ndarray, track_masks: dict[str, np.ndarray], *, threshold: float) -> LocalMatch:
    """Compare a new SAM box-prompt mask to existing tracks at that same frame.

    Exactly one candidate >= threshold permits local ID reuse. Multiple matches
    stay ambiguous; low overlap stays unmatched. Neither creates confirmed
    physical instances. Rectangle-to-mask IoU is deliberately not substituted.
    """
    if type(threshold) not in (float, int) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("local IoU threshold must be in (0,1]")
    prompt_mask = np.asarray(prompt_mask)
    if prompt_mask.ndim != 2 or prompt_mask.dtype != np.bool_ or not all(prompt_mask.shape):
        raise ValueError("prompt mask must be a nonempty boolean grid")
    scores = {}
    for track_id, mask in track_masks.items():
        mask = np.asarray(mask)
        if not isinstance(track_id, str) or not track_id or mask.shape != prompt_mask.shape or mask.dtype != np.bool_:
            raise ValueError("track masks require explicit IDs and identical boolean grids")
        union = np.count_nonzero(prompt_mask | mask)
        scores[track_id] = float(np.count_nonzero(prompt_mask & mask) / union) if union else 0.
    candidates = sorted(track_id for track_id, score in scores.items() if score >= threshold)
    if len(candidates) == 1:
        return LocalMatch("unique_local_match", candidates[0], candidates, scores)
    return LocalMatch("ambiguous" if candidates else "unmatched", None, candidates, scores)
