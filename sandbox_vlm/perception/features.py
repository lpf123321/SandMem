"""Pool frozen SAM image features without an additional appearance encoder."""
from __future__ import annotations

import numpy as np

from sandbox_vlm.memory.sandbox import AppearanceFeature


def pool_sam_region_feature(image_features, source_mask: np.ndarray) -> AppearanceFeature:
    """Native SAM2 whole-image resize: transport mask coverage onto its feature grid.

    Caller must provide image-encoder features BEFORE memory conditioning. SAM2
    resizes the entire source image to a square, so area resampling of the source
    mask to the full feature grid is appropriate (no unrecorded crop/padding).
    Empty or invalid evidence fails explicitly instead of manufacturing a vector.
    """
    import torch
    import torch.nn.functional as F

    mask = np.asarray(source_mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not all(mask.shape):
        raise ValueError("source mask must be a nonempty boolean image")
    if image_features.ndim != 3 or not all(image_features.shape):
        raise ValueError("expected encoder features with shape C,H,W")
    with torch.no_grad():
        grid = image_features.detach().float()
        if not torch.isfinite(grid).all():
            raise ValueError("nonfinite image encoder features")
        weights = F.interpolate(torch.as_tensor(mask.copy(), device=grid.device)[None, None].float(),
                                size=grid.shape[-2:], mode="area")[0, 0]
        support = weights.sum()
        if support <= 0:
            raise ValueError("empty mask has no appearance evidence")
        vector = (grid * weights).sum(dim=(1, 2)) / support
        return AppearanceFeature(vector.cpu().numpy())
