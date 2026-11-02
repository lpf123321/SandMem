"""Compose a candidate SAM track with VGGT outputs; no global identity decision."""
from dataclasses import dataclass

import numpy as np

from sandbox_vlm.memory.geometry import CameraParams
from sandbox_vlm.memory.sandbox import AppearanceFeature
from sandbox_vlm.perception.confidence import select_reliable_depth
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.perception.observations import LocalObservation, estimate_local_aabb, source_mask_to_local_points


@dataclass
class LocalGeometryResult:
    observation: LocalObservation | None
    diagnostics: dict


def build_local_observation(*, video_id: str, observation_id: str, name: str, description: str,
                            frame_ids: list[int], source_sizes: list[tuple[int, int]],
                            masks: dict[int, np.ndarray], cameras: list[CameraParams],
                            depth: np.ndarray, confidence: np.ndarray,
                            representative_feature: AppearanceFeature | None,
                            keep_fraction: float, min_points: int) -> LocalGeometryResult:
    """Masks keyed by video frame ID; cameras/depth in identical clip order.

    Nonempty propagated masks are candidate observations only. This function
    creates no viewpoint links or confirmed identities. The caller selects the
    representative frozen feature explicitly; no learned fusion is introduced.
    """
    if not frame_ids or frame_ids != sorted(set(frame_ids)):
        raise ValueError("ordered unique frame IDs required")
    if len(source_sizes) != len(frame_ids) or len(cameras) != len(frame_ids):
        raise ValueError("inconsistent frame/camera/source-size counts")
    if set(masks) != set(frame_ids):
        raise ValueError("every processed frame must have an explicit candidate mask")
    reference = cameras[0].reference
    depth = np.asarray(depth)
    if depth.ndim != 3 or depth.shape[0] != len(frame_ids):
        raise ValueError("depth must follow the full clip frame order")
    transforms = vggt_grid_transforms(source_sizes, mode="crop")
    selected = select_reliable_depth(depth, confidence, keep_fraction=keep_fraction)
    batches, candidate_frames, point_counts = [], [], {}
    for i, (frame_id, camera, transform) in enumerate(zip(frame_ids, cameras, transforms)):
        if (camera.frame_id != frame_id or camera.reference != reference
                or camera.image_size != depth.shape[1:] or transform.output_hw != camera.image_size):
            raise ValueError("frame, reference or native VGGT grid mismatch")
        mask = np.asarray(masks[frame_id])
        batch = source_mask_to_local_points(mask, transform, depth[i], camera,
                                             reliable_pixels=selected.pixels[i])
        batches.append(batch)
        point_counts[str(frame_id)] = batch.selected_pixel_count
        if mask.any():
            candidate_frames.append(frame_id)
    estimate = estimate_local_aabb(batches, coordinate_reference=reference, min_points=min_points)
    diagnostics = {"depth_conf_keep_fraction": keep_fraction, "depth_conf_threshold": selected.threshold,
                   "finite_score_count": selected.finite_score_count,
                   "retained_depth_pixels": selected.retained_pixel_count,
                   "point_count_by_frame": point_counts, "point_count": estimate.point_count,
                   "geometry_support_frame_ids": estimate.support_frame_ids, "min_points": min_points,
                   "association_status": "candidate_track_only", "confirmed_global_links": 0}
    if not candidate_frames:
        return LocalGeometryResult(None, {**diagnostics, "failure_reason": "empty_candidate_track"})
    failures = []
    if estimate.failure_reason:
        failures.append(estimate.failure_reason)
    if representative_feature is None:
        failures.append("representative_feature_unavailable")
    observation = LocalObservation(video_id, observation_id, reference, name, description,
        candidate_frames, estimate.bbox_3d, representative_feature, failures)
    return LocalGeometryResult(observation, diagnostics)
