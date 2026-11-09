"""Scene lifecycle independent of geometry stream and global instance identity.

Uses the coauthor's current-voxel overlap fraction and frozen initial diagonal
limit. Configuration values are pilot settings, not validated hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sandbox_vlm.memory.geometry import CameraParams, BBox3D


class ScenePolicy(Protocol):
    def initial_limit(self, bounds: np.ndarray) -> object:
        """Freeze the expansion limit from the first reliable observation."""

    def sufficient_overlap(self, occupied: frozenset, incoming: frozenset) -> bool:
        """Compare only the active scene, using the agreed overlap formula."""

    def within_limit(self, proposed_bounds: np.ndarray, fixed_limit: object) -> bool:
        """Check the proposed union against the frozen initial limit."""


@dataclass(frozen=True)
class VoxelScenePolicy:
    voxel_size: float
    overlap_threshold: float
    expansion_factor: float

    def __post_init__(self):
        if (not np.isfinite([self.voxel_size, self.overlap_threshold, self.expansion_factor]).all()
                or self.voxel_size <= 0 or not 0 <= self.overlap_threshold <= 1
                or self.expansion_factor < 1):
            raise ValueError('invalid voxel scene policy')

    def initial_limit(self, bounds):
        return self.expansion_factor * max(float(np.linalg.norm(bounds[1] - bounds[0])), self.voxel_size)

    def sufficient_overlap(self, occupied, incoming):
        return bool(incoming) and len(occupied & incoming) / len(incoming) >= self.overlap_threshold

    def within_limit(self, proposed_bounds, fixed_limit):
        return float(np.linalg.norm(proposed_bounds[1] - proposed_bounds[0])) <= fixed_limit

    def voxelize(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('voxel input must be finite Nx3')
        indices = np.floor(points / self.voxel_size)
        if np.any(np.abs(indices) >= np.iinfo(np.int64).max):
            raise ValueError('voxel coordinates exceed supported integer range')
        return frozenset(map(tuple, indices.astype(np.int64).tolist()))


@dataclass(frozen=True)
class SceneObservation:
    observation_id: str
    instance_id: int
    frame_id: int
    bbox: BBox3D | None = None


@dataclass
class ContinuousScene:
    scene_id: int
    reference: str
    bounds: np.ndarray
    fixed_limit: object
    cameras: dict[int, CameraParams] = field(default_factory=dict)
    observations: list[SceneObservation] = field(default_factory=list)
    closed: bool = False


class ContinuousSceneMemory:
    """Accept chronological geometry, including frames without any instances.

    Missing reliable geometry retains the actual camera/observations with null
    scene assignment. These are existing instance links, not pending identities.
    Existing scenes/IDs/coordinates are never searched, merged or rewritten.
    """

    def __init__(self, *, reference: str, policy: ScenePolicy):
        if not reference:
            raise ValueError('named shared reference required')
        self.reference = reference
        self.policy = policy
        self.scenes: list[ContinuousScene] = []
        self._occupied: frozenset = frozenset()
        self.last_frame_id = None
        self.views = {}

    @property
    def active(self):
        return self.scenes[-1] if self.scenes and not self.scenes[-1].closed else None

    @property
    def occupied_voxels(self):
        return self._occupied

    def add(self, *, camera: CameraParams, reliable_points: np.ndarray,
            occupied_voxels: frozenset, observations: tuple[SceneObservation, ...],
            available_through: int):
        fid = camera.frame_id
        if camera.reference != self.reference:
            raise ValueError('scene transition cannot change coordinate reference')
        if fid > available_through or (self.last_frame_id is not None and fid <= self.last_frame_id):
            raise ValueError('camera frames must be chronological and causal')
        points = np.asarray(reliable_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('reliable points must be finite Nx3 in shared reference')
        if type(occupied_voxels) is not frozenset or bool(len(points)) != bool(occupied_voxels):
            raise ValueError('points and prefiltered voxel evidence must agree on availability')
        keys = set()
        for obs in observations:
            if (not obs.observation_id or obs.observation_id in keys or obs.frame_id != fid
                    or type(obs.instance_id) is not int or obs.instance_id < 0
                    or (obs.bbox is not None and obs.bbox.frame != self.reference)):
                raise ValueError('invalid actual observation identity or reference')
            keys.add(obs.observation_id)
        if not len(points):
            self.views[fid] = dict(camera=camera, observations=observations, scene_id=None)
            self.last_frame_id = fid
            return dict(status='no_reliable_geometry', scene_id=None)
        bounds = np.stack([points.min(axis=0), points.max(axis=0)])
        current = self.active
        reason = 'initial'
        if current is not None:
            union = np.stack([np.minimum(current.bounds[0], bounds[0]),
                              np.maximum(current.bounds[1], bounds[1])])
            if not self.policy.sufficient_overlap(self._occupied, occupied_voxels):
                reason = 'insufficient_overlap'
            elif not self.policy.within_limit(union.copy(), current.fixed_limit):
                reason = 'range_limit'
            else:
                reason = 'joined'
        if reason != 'joined':
            # Compute the new limit before changing the prior active scene.
            limit = self.policy.initial_limit(bounds.copy())
            if current is not None:
                current.closed = True
            current = ContinuousScene(len(self.scenes) + 1, self.reference, bounds.copy(), limit)
            self.scenes.append(current)
            self._occupied = occupied_voxels
        else:
            current.bounds = union
            self._occupied = self._occupied | occupied_voxels
        current.cameras[fid] = camera
        current.observations.extend(observations)
        self.views[fid] = dict(camera=camera, observations=observations, scene_id=current.scene_id)
        self.last_frame_id = fid
        return dict(status=reason, scene_id=current.scene_id)

    def finish(self):
        if self.active is not None:
            self.active.closed = True
        self._occupied = frozenset()

    def begin_reference(self, *, reference: str, policy: ScenePolicy):
        """Only when upstream explicitly starts a distinct reconstruction.

        Camera frame IDs stay video-global; never compare these separate grids.
        FIFO eviction alone must not call this method.
        """
        if not reference or reference in {s.reference for s in self.scenes} | {self.reference}:
            raise ValueError('independent reconstruction needs a fresh reference')
        self.finish()
        self.reference = reference
        self.policy = policy
