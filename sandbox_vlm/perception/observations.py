"""Compact local candidates: no global identity and no implicit geometric success."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sandbox_vlm.memory.geometry import BBox3D, CameraParams
from sandbox_vlm.memory.sandbox import AppearanceFeature


@dataclass
class LocalObservation:
    video_id: str
    local_observation_id: str
    coordinate_reference: str
    name: str
    short_description: str
    frame_ids: list[int]
    bbox_3d: BBox3D | None = None
    appearance: AppearanceFeature | None = None
    failure_reasons: list[str] = field(default_factory=list)
    prompt_evidence: list[dict] = field(default_factory=list)

    def __post_init__(self):
        from sandbox_vlm.perception.evidence import PromptEvidence
        for evidence in self.prompt_evidence:
            PromptEvidence.from_dict(evidence)
        if not all(isinstance(s, str) and s.strip() for s in
                   (self.video_id, self.local_observation_id, self.coordinate_reference, self.name)):
            raise ValueError("video, observation, reference and name must be explicit")
        if (not self.frame_ids or any(type(f) is not int or f < 0 for f in self.frame_ids)
                or self.frame_ids != sorted(set(self.frame_ids))):
            raise ValueError("observation must have unique ordered source frame IDs")
        if self.bbox_3d is not None and self.bbox_3d.frame != self.coordinate_reference:
            raise ValueError("local box reference mismatch")
        if (self.bbox_3d is None or self.appearance is None) and not self.failure_reasons:
            raise ValueError("missing geometry/features require an explicit failure reason")

    def to_dict(self):
        return {"video_id": self.video_id, "local_observation_id": self.local_observation_id,
                "coordinate_reference": self.coordinate_reference, "name": self.name,
                "short_description": self.short_description, "frame_ids": list(self.frame_ids),
                "bbox_3d": self.bbox_3d.to_dict() if self.bbox_3d is not None else None,
                "appearance": self.appearance.to_dict() if self.appearance is not None else None,
                "failure_reasons": list(self.failure_reasons),
                "prompt_evidence": list(self.prompt_evidence)}

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data,
                      "bbox_3d": BBox3D.from_dict(data["bbox_3d"]) if data.get("bbox_3d") is not None else None,
                      "appearance": AppearanceFeature.from_dict(data["appearance"]) if data.get("appearance") is not None else None})


@dataclass
class MaskedPoints:
    frame_id: int
    coordinate_reference: str
    pixel_xy: np.ndarray
    points_ref: np.ndarray
    selected_pixel_count: int


def backproject_masked_depth(depth: np.ndarray, mask: np.ndarray, camera: CameraParams,
                              *, reliable_pixels: np.ndarray) -> MaskedPoints:
    """Consume an explicitly supplied reliability decision; never choose thresholds.

    All grids must already be mapped to camera.image_size. Nonfinite/nonpositive
    depths are excluded even if a caller accidentally marks them reliable.
    Empty support produces empty points, never a fabricated box.
    """
    depth, mask, reliable = np.asarray(depth), np.asarray(mask), np.asarray(reliable_pixels)
    if any(a.shape != camera.image_size for a in (depth, mask, reliable)):
        raise ValueError("depth, mask, reliability and camera must use the same image grid")
    if mask.dtype != np.bool_ or reliable.dtype != np.bool_:
        raise ValueError("mask and reliability must be explicit boolean arrays")
    selected = mask & reliable & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(selected)
    pixel_xy = np.column_stack([xx, yy]).astype(np.float64)
    rays = np.column_stack([pixel_xy, np.ones(len(xx))]) @ np.linalg.inv(camera.intrinsic).T
    camera_points = rays * depth[yy, xx, None]
    pose = camera.T_ref_from_camera
    points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    return MaskedPoints(camera.frame_id, camera.reference, pixel_xy, points, len(xx))


def source_mask_to_local_points(source_mask: np.ndarray, grid_transform, depth: np.ndarray,
                                camera: CameraParams, *, reliable_pixels: np.ndarray) -> MaskedPoints:
    """Compose source-mask transport and backprojection with explicit reliability."""
    if grid_transform.output_hw != camera.image_size:
        raise ValueError("image transform does not match camera/depth output grid")
    grid_mask = grid_transform.mask_to_grid(source_mask)
    return backproject_masked_depth(depth, grid_mask, camera, reliable_pixels=reliable_pixels)


@dataclass
class BoxEstimate:
    bbox_3d: BBox3D | None
    support_frame_ids: list[int]
    point_count: int
    failure_reason: str | None = None


def estimate_local_aabb(batches: list[MaskedPoints], *, coordinate_reference: str,
                        min_points: int) -> BoxEstimate:
    """AABB of explicitly accepted points in ONE reference (decision D005).

    The caller supplies reliability and minimum support; neither is calibrated
    here. Extents describe observed surfaces, not inferred hidden object volume.
    Duplicate frame batches are rejected, so overlap cannot inflate support.
    """
    if not coordinate_reference.strip() or type(min_points) is not int or min_points < 1:
        raise ValueError("explicit reference and positive integer min_points required")
    seen, support = set(), []
    lower, upper, count = np.full(3, np.inf), np.full(3, -np.inf), 0
    for batch in batches:
        if batch.coordinate_reference != coordinate_reference:
            raise ValueError("cannot estimate a box across unaligned references")
        if type(batch.frame_id) is not int or batch.frame_id < 0 or batch.frame_id in seen:
            raise ValueError("point batches require unique nonnegative frame IDs")
        seen.add(batch.frame_id)
        points = np.asarray(batch.points_ref, dtype=np.float64)
        pixels = np.asarray(batch.pixel_xy)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or pixels.shape != (len(points), 2) or not np.isfinite(pixels).all()
                or batch.selected_pixel_count != len(points)):
            raise ValueError("malformed reliable point batch")
        if len(points):
            lower = np.minimum(lower, points.min(axis=0))
            upper = np.maximum(upper, points.max(axis=0))
            count += len(points)
            support.append(batch.frame_id)
    if count < min_points:
        return BoxEstimate(None, sorted(support), count, "insufficient_reliable_points")
    return BoxEstimate(BBox3D((lower + upper) / 2, upper - lower, np.eye(3), coordinate_reference),
                       sorted(support), count)
