"""长期记忆: 统一沙盒 (Sandbox)。

设计约束 (对齐 Proposal / AGENTS.md):
  - 一个沙盒统一组织空间范围框、持久物理实例与观测视点, 不使用"三张表"。
  - 确定的跨视点实例对应由共享全局实例 ID 表达; 实例侧不重复保存完整出现帧列表。
  - 物体出现在哪些帧由视点的实际观测链接 (observed_instance_ids) 导出。
  - 原图通过帧 ID 从源视频按需读取, 不保存每物体原图副本或读取地址。
  - 仅有歧义的对应保留 AmbiguityNote; 若无真实观察依据, 不得写入 observed_instance_ids。
  - 合并/拆分/修订关联后, 全局 ID、视点链接与注记必须一致更新, 不留悬空引用。

概念结构:
  Sandbox
    spatial_extents[]  # bounds + coordinate_reference
      instances[]      # global_instance_id + name + short_description + bbox_3d + appearance
      viewpoints[]     # frame_id + camera_parameters + observed_instance_ids
    ambiguity_notes[]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import TYPE_CHECKING, Optional

import numpy as np

from sandbox_vlm.memory.geometry import BBox3D, CameraParams

if TYPE_CHECKING:
    from sandbox_vlm.perception.observations import LocalObservation


@dataclass
class AppearanceFeature:
    """代表性外观特征: 由冻结分割编码器在 mask 区域池化并 L2 归一化。"""

    vector: np.ndarray                     # (D,) float32, 已归一化
    dim: int = 0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("appearance storage dtype must be float16 or float32")
        vector = np.asarray(self.vector, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("appearance feature must be nonempty and finite")
        n = float(np.linalg.norm(vector))
        if not np.isfinite(n) or n <= 0:
            raise ValueError("appearance feature must have nonzero finite norm")
        self.vector = (vector / n).astype(self.dtype)
        if self.dim <= 0:
            self.dim = int(self.vector.shape[0])
        elif self.dim != self.vector.size:
            raise ValueError("appearance dimension differs from stored vector")

    @property
    def nbytes(self) -> int:
        return int(self.vector.nbytes)

    def to_dict(self) -> dict:
        return {"vector": self.vector.tolist(), "dim": self.dim, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict) -> "AppearanceFeature":
        return cls(vector=d["vector"], dim=d.get("dim", 0), dtype=d.get("dtype", "float32"))


@dataclass
class ObjectInstance:
    """持久物理实例。"""

    global_instance_id: int
    name: str
    short_description: str
    bbox_3d: BBox3D
    appearance: AppearanceFeature

    def to_dict(self) -> dict:
        return {
            "global_instance_id": self.global_instance_id,
            "name": self.name,
            "short_description": self.short_description,
            "bbox_3d": self.bbox_3d.to_dict(),
            "appearance": self.appearance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectInstance":
        return cls(
            global_instance_id=d["global_instance_id"],
            name=d["name"],
            short_description=d.get("short_description", ""),
            bbox_3d=BBox3D.from_dict(d["bbox_3d"]),
            appearance=AppearanceFeature.from_dict(d["appearance"]),
        )


@dataclass
class Viewpoint:
    """一个实际观测视点。observed_instance_ids 仅记录真实观察支持的关联。"""

    frame_id: int
    camera: CameraParams
    observed_instance_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "camera": self.camera.to_dict(),
            "observed_instance_ids": list(self.observed_instance_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Viewpoint":
        return cls(
            frame_id=d["frame_id"],
            camera=CameraParams.from_dict(d["camera"]),
            observed_instance_ids=list(d.get("observed_instance_ids", [])),
        )


@dataclass
class SpatialExtent:
    """空间范围框。不是物理房间; 多个范围框可能是尚未对齐的观测。"""

    extent_id: int
    coordinate_reference: str
    bounds: BBox3D
    instances: list[ObjectInstance] = field(default_factory=list)
    viewpoints: list[Viewpoint] = field(default_factory=list)

    def instance_ids(self) -> set[int]:
        return {i.global_instance_id for i in self.instances}

    def to_dict(self) -> dict:
        return {
            "extent_id": self.extent_id,
            "coordinate_reference": self.coordinate_reference,
            "bounds": self.bounds.to_dict(),
            "instances": [i.to_dict() for i in self.instances],
            "viewpoints": [v.to_dict() for v in self.viewpoints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialExtent":
        return cls(
            extent_id=d["extent_id"],
            coordinate_reference=d["coordinate_reference"],
            bounds=BBox3D.from_dict(d["bounds"]),
            instances=[ObjectInstance.from_dict(i) for i in d.get("instances", [])],
            viewpoints=[Viewpoint.from_dict(v) for v in d.get("viewpoints", [])],
        )


@dataclass
class AmbiguityNote:
    """待定关联注记: 仅保留必要候选实例 ID、相关帧与原因。"""

    local_observation_id: str
    candidate_instance_ids: list[int]
    frame_ids: list[int]
    reason: str

    def to_dict(self) -> dict:
        return {
            "local_observation_id": self.local_observation_id,
            "candidate_instance_ids": list(self.candidate_instance_ids),
            "frame_ids": list(self.frame_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AmbiguityNote":
        return cls(
            local_observation_id=d["local_observation_id"],
            candidate_instance_ids=list(d.get("candidate_instance_ids", [])),
            frame_ids=list(d.get("frame_ids", [])),
            reason=d.get("reason", ""),
        )


class Sandbox:
    """统一组织空间范围框、实例与视点。"""

    def __init__(self, spatial_extents: Optional[list[SpatialExtent]] = None,
                 ambiguity_notes: Optional[list[AmbiguityNote]] = None, *,
                 video_id: Optional[str] = None,
                 pending_observations: Optional[list[LocalObservation]] = None,
                 retain_unresolved: bool = True) -> None:
        if type(retain_unresolved) is not bool:
            raise ValueError('retain_unresolved must be boolean')
        if not retain_unresolved and (pending_observations or ambiguity_notes):
            raise ValueError('instances-only memory cannot contain unresolved observations')
        self.retain_unresolved = retain_unresolved
        self.video_id = video_id
        self.spatial_extents: list[SpatialExtent] = spatial_extents or []
        self.ambiguity_notes: list[AmbiguityNote] = ambiguity_notes or []
        self.pending_observations: list[LocalObservation] = pending_observations or []

    # ---- 查询 ----
    def all_instances(self) -> list[ObjectInstance]:
        return [i for e in self.spatial_extents for i in e.instances]

    def all_viewpoints(self) -> list[Viewpoint]:
        return [v for e in self.spatial_extents for v in e.viewpoints]

    def instance_ids(self) -> set[int]:
        return {i.global_instance_id for i in self.all_instances()}

    def get_instance(self, gid: int) -> Optional[ObjectInstance]:
        for i in self.all_instances():
            if i.global_instance_id == gid:
                return i
        return None

    def frames_for_instance(self, gid: int) -> list[int]:
        """由视点的实际观测链接导出该实例的出现帧 (不额外持久化)。"""
        out = []
        for v in self.all_viewpoints():
            if gid in v.observed_instance_ids:
                out.append(v.frame_id)
        return sorted(set(out))

    def instances_for_frame(self, frame_id: int) -> list[int]:
        return sorted({gid for v in self.all_viewpoints() if v.frame_id == frame_id
                       for gid in v.observed_instance_ids})

    def reference_of(self, gid: int) -> Optional[str]:
        for e in self.spatial_extents:
            if any(i.global_instance_id == gid for i in e.instances):
                return e.coordinate_reference
        return None

    # ---- 一致性 ----
    def validate(self) -> list[str]:
        """返回一致性问题列表 (空表示通过)。"""
        problems: list[str] = []
        if not self.retain_unresolved and (self.pending_observations or self.ambiguity_notes):
            problems.append('instances-only memory contains unresolved observations')
        ids: list[int] = []
        extent_ids = [e.extent_id for e in self.spatial_extents]
        if len(extent_ids) != len(set(extent_ids)):
            problems.append("duplicate extent_id")
        for e in self.spatial_extents:
            if e.bounds.frame != e.coordinate_reference:
                problems.append(f"extent {e.extent_id}: bounds reference mismatch")
            local_ids = e.instance_ids()
            frame_ids = [v.frame_id for v in e.viewpoints]
            if len(frame_ids) != len(set(frame_ids)):
                problems.append(f"extent {e.extent_id}: duplicate viewpoint frame_id")
            for i in e.instances:
                ids.append(i.global_instance_id)
                if i.bbox_3d.frame != e.coordinate_reference:
                    problems.append(f"instance {i.global_instance_id}: reference mismatch")
            for v in e.viewpoints:
                if v.camera.reference != e.coordinate_reference:
                    problems.append(f"viewpoint {v.frame_id}: camera reference mismatch")
                if v.camera.frame_id != v.frame_id:
                    problems.append(f"viewpoint {v.frame_id}: camera frame_id mismatch")
                if len(v.observed_instance_ids) != len(set(v.observed_instance_ids)):
                    problems.append(f"viewpoint {v.frame_id}: duplicate observed_instance_id")
                for gid in v.observed_instance_ids:
                    if gid not in local_ids:
                        problems.append(
                            f"extent {e.extent_id}: viewpoint {v.frame_id} "
                            f"links unknown instance {gid}")
        if len(ids) != len(set(ids)):
            dup = sorted(x for x, count in Counter(ids).items() if count > 1)
            problems.append(f"duplicate global_instance_id: {dup}")
        known = set(ids)
        pending_ids = [o.local_observation_id for o in self.pending_observations]
        if len(pending_ids) != len(set(pending_ids)):
            problems.append("duplicate pending local_observation_id")
        for observation in self.pending_observations:
            if not self.video_id or observation.video_id != self.video_id:
                problems.append(f"pending {observation.local_observation_id}: video mismatch")
            try:
                observation.__post_init__()
            except ValueError as exc:
                problems.append(f"pending {observation.local_observation_id}: {exc}")
        note_ids=[n.local_observation_id for n in self.ambiguity_notes]
        if len(note_ids)!=len(set(note_ids)):
            problems.append('duplicate ambiguity local_observation_id')
        pending_by_id={o.local_observation_id:o for o in self.pending_observations}
        for note in self.ambiguity_notes:
            observation=pending_by_id.get(note.local_observation_id)
            if observation is None:
                problems.append(f'ambiguity {note.local_observation_id}: unknown pending observation')
            elif note.frame_ids!=observation.frame_ids:
                problems.append(f'ambiguity {note.local_observation_id}: evidence frame mismatch')
            for gid in note.candidate_instance_ids:
                if gid not in known:
                    problems.append(
                        f"ambiguity {note.local_observation_id}: unknown candidate {gid}")
        return problems

    # ---- 容量计量 ----
    def capacity_report(self) -> dict:
        from sandbox_vlm.memory.storage import cpu_storage_report
        instances = self.all_instances()
        viewpoints = self.all_viewpoints()
        feat_bytes = sum(i.appearance.nbytes for i in instances)
        links = sum(len(v.observed_instance_ids) for v in viewpoints)
        return {
            "n_spatial_extents": len(self.spatial_extents),
            "n_instances": len(instances),
            "n_viewpoints": len(viewpoints),
            "n_links": links,
            "n_ambiguity_notes": len(self.ambiguity_notes),
            "n_pending_observations": len(self.pending_observations),
            "pending_feature_bytes": sum(o.appearance.nbytes for o in self.pending_observations if o.appearance is not None),
            "appearance_feature_bytes": feat_bytes,
            "appearance_feature_mb": round(feat_bytes / 1024 / 1024, 4),
            **cpu_storage_report(self),
            "note": "当前为 CPU 结构记忆计量；包含待定状态和 Python/NumPy 存储，不代表 GPU 显存预算已实现",
        }

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        core = {
            "video_id": self.video_id,
            "spatial_extents": [e.to_dict() for e in self.spatial_extents],
        }
        if self.retain_unresolved:
            core.update(ambiguity_notes=[n.to_dict() for n in self.ambiguity_notes],
                        pending_observations=[o.to_dict() for o in self.pending_observations])
        else:
            if self.pending_observations or self.ambiguity_notes:
                raise ValueError('instances-only memory contains unresolved observations')
            core['memory_policy'] = 'instances_only'
        return core

    @classmethod
    def from_dict(cls, d: dict) -> "Sandbox":
        from sandbox_vlm.perception.observations import LocalObservation
        # CoverageSandbox/VisualSandbox are intentionally different representations.
        # Previously their flat records were silently discarded by the defaults
        # below, producing an apparently valid *empty* 3D sandbox.
        if (not isinstance(d, dict) or 'spatial_extents' not in d
                or any(key in d for key in ('representation', 'question_key', 'instances', 'viewpoints'))):
            raise ValueError('explicit spatial sandbox required; visual hypotheses cannot be silently converted')
        policy = d.get('memory_policy', 'legacy')
        if policy not in ('legacy', 'instances_only'):
            raise ValueError('unknown spatial memory policy')
        if policy == 'instances_only' and any(k in d for k in ('pending_observations', 'ambiguity_notes')):
            raise ValueError('instances-only snapshot cannot carry unresolved observation fields')
        return cls(
            video_id=d.get("video_id"),
            spatial_extents=[SpatialExtent.from_dict(e) for e in d.get("spatial_extents", [])],
            ambiguity_notes=[AmbiguityNote.from_dict(n) for n in d.get("ambiguity_notes", [])],
            pending_observations=[LocalObservation.from_dict(o) for o in d.get("pending_observations", [])],
            retain_unresolved=policy != 'instances_only',
        )
