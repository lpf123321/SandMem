"""Explicit source-image -> native VGGT grid mapping, independent of model inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageGridTransform:
    source_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    crop_top: int
    cropped_hw: tuple[int, int]
    pad_top: int
    pad_left: int
    output_hw: tuple[int, int]

    def __post_init__(self):
        sizes = (*self.source_hw, *self.resized_hw, *self.cropped_hw, *self.output_hw)
        if any(type(v) is not int or v <= 0 for v in sizes):
            raise ValueError("image sizes must be positive integers")
        if any(type(v) is not int or v < 0 for v in (self.crop_top, self.pad_top, self.pad_left)):
            raise ValueError("crop/padding offsets must be nonnegative integers")
        if (self.crop_top + self.cropped_hw[0] > self.resized_hw[0]
                or self.cropped_hw[1] != self.resized_hw[1]
                or self.pad_top + self.cropped_hw[0] > self.output_hw[0]
                or self.pad_left + self.cropped_hw[1] > self.output_hw[1]):
            raise ValueError("inconsistent resize/crop/padding geometry")

    @property
    def A_grid_from_source(self) -> np.ndarray:
        """Pixel-centre convention, matching PIL resize followed by crop and padding."""
        sy, sx = np.asarray(self.resized_hw) / np.asarray(self.source_hw)
        return np.array([[sx, 0, (sx - 1) / 2 + self.pad_left],
                         [0, sy, (sy - 1) / 2 - self.crop_top + self.pad_top],
                         [0, 0, 1]], dtype=np.float64)

    def mask_to_grid(self, mask: np.ndarray) -> np.ndarray:
        """Nearest-neighbour categorical mask transport; padding is never observed."""
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.source_hw:
            raise ValueError("mask must be boolean and match source image size")
        rh, rw = self.resized_hw
        resized = np.asarray(Image.fromarray(mask).resize((rw, rh), Image.Resampling.NEAREST))
        ch, cw = self.cropped_hw
        result = np.zeros(self.output_hw, dtype=bool)
        result[self.pad_top:self.pad_top + ch, self.pad_left:self.pad_left + cw] = resized[self.crop_top:self.crop_top + ch]
        return result

    def intrinsic_in_source(self, intrinsic_grid: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.A_grid_from_source, np.asarray(intrinsic_grid, dtype=np.float64))


def vggt_grid_transforms(source_sizes: list[tuple[int, int]], mode: str = "crop") -> list[ImageGridTransform]:
    """Mirror installed VGGT's 518/14 native loader, including mixed-size batch padding.

    This is an adapter to that loader, not a new preprocessing policy. Unsupported
    extreme aspect ratios fail explicitly just as zero-size native resizing would.
    """
    if mode not in {"crop", "pad"}:
        raise ValueError("unsupported VGGT preprocessing mode")
    entries = []
    for h, w in source_sizes:
        if type(h) is not int or type(w) is not int or min(h, w) <= 0:
            raise ValueError("source dimensions must be positive integers")
        if mode == "pad" and h > w:
            rh, rw = 518, round(w * 518 / h / 14) * 14
        else:
            rw, rh = 518, round(h * 518 / w / 14) * 14
        top = (rh - 518) // 2 if mode == "crop" and rh > 518 else 0
        cropped = (min(rh, 518) if mode == "crop" else rh, rw)
        entries.append(((h, w), (rh, rw), top, cropped))
    if not entries:
        return []
    output = (518, 518) if mode == "pad" else (
        max(e[3][0] for e in entries), max(e[3][1] for e in entries))
    return [ImageGridTransform(src, resized, top, cropped,
                                (output[0] - cropped[0]) // 2, (output[1] - cropped[1]) // 2, output)
            for src, resized, top, cropped in entries]
