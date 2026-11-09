"""Experimental continuous geometry driver; not wired into frozen D116.

The schedule is the complete sampled video prefix, never a question retrieval.
Scene grouping is deliberately outside this object and cannot reset its cache.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable


def tensor_storage_bytes(value) -> int:
    """Count unique reachable tensor storage, including views sharing storage."""
    seen = set()
    total = 0

    def visit(obj):
        nonlocal total
        if isinstance(obj, dict):
            for item in obj.values():
                visit(item)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
        elif hasattr(obj, 'untyped_storage'):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()

    visit(value)
    return total


def trim_frame_cache(cache, *, keep_frames: int, slots_per_frame: int):
    """FIFO of whole frames in upstream [B,heads,slots,tokens,dim] K/V.

    clone() releases the old storage; a slice alone would still hold all history.
    Camera caches have num_iterations slots for each complete sampled frame.
    """
    if keep_frames < 1 or slots_per_frame < 1:
        raise ValueError('at least one historical frame must remain')
    trimmed = []
    counts = set()
    for entry in cache:
        if entry is None:
            trimmed.append(None)
            continue
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError('unsupported upstream KV cache format')
        key, value = entry
        if key.ndim != 5 or key.shape != value.shape or key.shape[2] % slots_per_frame:
            raise ValueError('cache does not contain complete frame groups')
        counts.add(key.shape[2] // slots_per_frame)
        slots = keep_frames * slots_per_frame
        trimmed.append((key[:, :, -slots:].clone(), value[:, :, -slots:].clone())
                       if key.shape[2] > slots else entry)
    if len(counts) > 1 or (counts and any(e is None for e in cache)):
        raise ValueError('cache layers disagree on frame history')
    return trimmed


class ContinuousStreamVGGT:
    """Persist both upstream caches across individual sampled-frame calls.

    Mirrors official StreamVGGT.inference's per-frame body at commit
    05682bcb05683c4c71522e9cb618fc8ca349063c. No tracking head is called.
    An initialized, frozen model and its preprocessing are supplied explicitly;
    this class never downloads a checkpoint or substitutes the old VGGT model.
    """

    def __init__(self, model, *, reference: str, frame_ids: tuple[int, ...], max_frames: int,
                 cache_limit_bytes: int, cache_frames: int = 16,
                 camera_iterations: int = 4, head_context: Callable = nullcontext):
        if not reference or type(max_frames) is not int or max_frames <= 0:
            raise ValueError('named reference and positive frame budget required')
        if type(cache_limit_bytes) is not int or cache_limit_bytes <= 0:
            raise ValueError('explicit positive geometry cache budget required')
        # Upstream selects a distinct token for the sequence's first frame from
        # nonempty history, not past_frame_idx. Keep >=1 old frame before append.
        if type(cache_frames) is not int or cache_frames < 2:
            raise ValueError('FIFO capacity must be at least two sampled frames')
        if type(camera_iterations) is not int or camera_iterations < 1:
            raise ValueError('positive camera refinement count required')
        if (not frame_ids or type(frame_ids) is not tuple
                or any(type(fid) is not int or fid < 0 for fid in frame_ids)
                or tuple(sorted(set(frame_ids))) != frame_ids):
            raise ValueError('complete immutable sampled-video frame schedule required')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('StreamVGGT must be frozen and in eval mode')
        self.model = model
        self.reference = reference
        self.frame_ids = frame_ids
        self.max_frames = max_frames
        self.cache_limit_bytes = cache_limit_bytes
        self.cache_frames = cache_frames
        self.camera_iterations = camera_iterations
        self.cached_frame_ids = ()
        self.head_context = head_context
        self.frame_index = 0
        self.last_frame_id = None
        self.grid_shape = None
        self.failed = False
        self.cache = [None] * model.aggregator.depth
        self.camera_cache = [None] * model.camera_head.trunk_depth

    def step(self, image, *, frame_id: int, sample_index: int,
             available_through: int):
        import torch

        if self.failed:
            raise RuntimeError('stream previously failed; no implicit restart or retry')
        if (type(frame_id) is not int or type(sample_index) is not int
                or sample_index != self.frame_index or frame_id > available_through
                or self.frame_index >= len(self.frame_ids)
                or frame_id != self.frame_ids[self.frame_index]
                or (self.last_frame_id is not None and frame_id <= self.last_frame_id)):
            raise ValueError('complete chronological sampled prefix required')
        if self.frame_index >= self.max_frames:
            raise RuntimeError('geometry frame budget exhausted')
        # Official per-frame input is [B=1, C=3, H, W].
        if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
            raise ValueError('expected one preprocessed RGB frame [1,3,H,W]')
        shape = tuple(image.shape[-2:])
        if self.grid_shape is not None and shape != self.grid_shape:
            raise ValueError('preprocessing grid must stay fixed throughout stream')
        before = tensor_storage_bytes((self.cache, self.camera_cache))
        try:
            with torch.inference_mode():
                self.cache = trim_frame_cache(self.cache, keep_frames=self.cache_frames - 1,
                                              slots_per_frame=1)
                self.camera_cache = trim_frame_cache(
                    self.camera_cache, keep_frames=self.cache_frames - 1,
                    slots_per_frame=self.camera_iterations)
                retained = self.cached_frame_ids[-(self.cache_frames - 1):]
                # Once sizes are known, reject a byte budget that cannot hold
                # the next full frame; never silently reduce configured K.
                trimmed_bytes = tensor_storage_bytes((self.cache, self.camera_cache))
                estimated_next = (trimmed_bytes // len(retained)) if retained else 0
                if trimmed_bytes + estimated_next > self.cache_limit_bytes:
                    raise RuntimeError('geometry cache budget cannot hold the next complete frame')
                images = image.unsqueeze(0)  # [1,1,3,H,W], same as upstream
                tokens, patch_start, self.cache = self.model.aggregator(
                    images, past_key_values=self.cache, use_cache=True,
                    past_frame_idx=self.frame_index)
                with self.head_context():
                    poses, self.camera_cache = self.model.camera_head(
                        tokens, num_iterations=self.camera_iterations,
                        past_key_values_camera=self.camera_cache, use_cache=True)
                    depth, depth_conf = self.model.depth_head(
                        tokens, images=images, patch_start_idx=patch_start)
                    points, point_conf = self.model.point_head(
                        tokens, images=images, patch_start_idx=patch_start)
                after = tensor_storage_bytes((self.cache, self.camera_cache))
                if after > self.cache_limit_bytes:
                    raise RuntimeError('geometry cache budget exceeded; stream stopped')
                expected_frames = len(retained) + 1
                for cache, slots in ((self.cache, 1), (self.camera_cache, self.camera_iterations)):
                    if any(e is None or e[0].ndim != 5
                           or e[0].shape != e[1].shape
                           or e[0].shape[2] != expected_frames * slots for e in cache):
                        raise ValueError('upstream cache growth differs from the audited frame layout')
                result = dict(
                    frame_id=frame_id, sample_index=sample_index, reference=self.reference,
                    pts3d_in_other_view=points[:, 0], conf=point_conf[:, 0],
                    depth=depth[:, 0], depth_conf=depth_conf[:, 0],
                    camera_pose=poses[-1][:, 0, :],
                    cost=dict(model_frame_steps=1, cache_before_bytes=before,
                              cache_after_bytes=after,
                              evicted_frame_ids=list(self.cached_frame_ids[:len(self.cached_frame_ids)-len(retained)]),
                              retained_frame_ids=list(retained + (frame_id,))))
        except Exception:
            # Upstream mutates lists in place. Retrying from partially updated
            # state would silently corrupt frame identity and camera history.
            self.failed = True
            raise
        self.frame_index += 1
        self.last_frame_id = frame_id
        self.grid_shape = shape
        self.cached_frame_ids = retained + (frame_id,)
        return result
