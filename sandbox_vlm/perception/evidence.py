"""Compact evidence emitted only after a valid discovery box and nonempty SAM seed."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PromptEvidence:
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    source_hw: tuple[int, int]
    sam_mask_nonempty: bool

    def __post_init__(self):
        box = np.asarray(self.bbox_xyxy, dtype=float)
        if (type(self.frame_id) is not int or self.frame_id < 0
                or len(self.source_hw) != 2
                or any(type(x) is not int or x <= 0 for x in self.source_hw)
                or box.shape != (4,) or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= self.source_hw[1])
                or not (0 <= box[1] < box[3] <= self.source_hw[0])
                or self.sam_mask_nonempty is not True):
            raise ValueError("actual evidence requires a valid source box and nonempty SAM prompt mask")

    def to_dict(self):
        return dict(frame_id=self.frame_id, bbox_xyxy=list(self.bbox_xyxy),
                    source_hw=list(self.source_hw), sam_mask_nonempty=True)

    @classmethod
    def from_dict(cls, value):
        return cls(value['frame_id'], tuple(value['bbox_xyxy']), tuple(value['source_hw']),
                   value['sam_mask_nonempty'])
