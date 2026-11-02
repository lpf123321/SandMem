"""几何原语: 3D 框与相机参数, 以及坐标变换约定。

坐标约定 (显式命名, 禁止混用):
  - `T_ref_from_camera` (4x4): 把相机坐标系中的点映射到参照系 (reference) 坐标。
    即 p_ref = T_ref_from_camera @ p_cam (齐次)。
  - 一个 SpatialExtent 有一个 `coordinate_reference` 名称; 其内的实例与视点都表达在该参照系。
  - 不同参照系之间不能直接比较位置/距离, 必须先做可靠对齐 (见 alignment 模块)。

尺度: VGGT 输出不保证真实米制尺度。绝对距离任务的尺度来源遵循评测协议, 不得使用测试标注。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBox3D:
    """有向 3D 包围盒 (OBB)，表达在某个参照系中。"""

    center: np.ndarray        # (3,)
    extents: np.ndarray       # (3,) 沿主轴的全长
    axes: np.ndarray          # (3,3) 主轴方向 (列向量, 正交归一)
    frame: str = "ref"        # 所属参照系名称

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.extents = np.asarray(self.extents, dtype=np.float64).reshape(3)
        self.axes = np.asarray(self.axes, dtype=np.float64).reshape(3, 3)
        if not all(np.isfinite(x).all() for x in (self.center, self.extents, self.axes)):
            raise ValueError("box geometry must be finite")
        if np.any(self.extents < 0):
            raise ValueError("box extents must be nonnegative")
        if not np.allclose(self.axes.T @ self.axes, np.eye(3), atol=1e-6):
            raise ValueError("box axes must be orthonormal")
        if not self.frame:
            raise ValueError("box reference must be named")

    def corners(self) -> np.ndarray:
        """返回 8 个角点 (8,3)。"""
        s = self.extents / 2.0
        out = []
        for sx in (-s[0], s[0]):
            for sy in (-s[1], s[1]):
                for sz in (-s[2], s[2]):
                    out.append(self.center + self.axes @ np.array([sx, sy, sz]))
        return np.array(out)

    @property
    def volume(self) -> float:
        return float(np.prod(np.abs(self.extents)))

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "extents": self.extents.tolist(),
            "axes": self.axes.tolist(),
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BBox3D":
        return cls(center=d["center"], extents=d["extents"],
                   axes=d["axes"], frame=d.get("frame", "ref"))


@dataclass
class CameraParams:
    """单个观测视点的相机参数。"""

    frame_id: int
    T_ref_from_camera: np.ndarray          # (4,4) 相机 -> 参照系
    intrinsic: np.ndarray                  # (3,3)
    image_size: tuple[int, int]            # (H, W)
    reference: str = "ref"

    def __post_init__(self) -> None:
        self.T_ref_from_camera = np.asarray(self.T_ref_from_camera, dtype=np.float64).reshape(4, 4)
        self.intrinsic = np.asarray(self.intrinsic, dtype=np.float64).reshape(3, 3)
        scale, _, _ = decompose_similarity(self.T_ref_from_camera)
        if not np.isclose(scale, 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("camera pose must be rigid; coordinate scale belongs to the reference")
        if (not np.isfinite(self.intrinsic).all()
                or self.intrinsic[0, 0] <= 0 or self.intrinsic[1, 1] <= 0
                or not np.allclose(self.intrinsic[2], [0, 0, 1])):
            raise ValueError("invalid pinhole intrinsic matrix")
        if len(self.image_size) != 2 or any(v <= 0 for v in self.image_size):
            raise ValueError("image_size must contain positive height and width")
        if not self.reference:
            raise ValueError("camera reference must be named")

    def project(self, pts_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project points; mask only means finite positive depth, never actual observation."""
        pts_ref = np.asarray(pts_ref, dtype=np.float64).reshape(-1, 3)
        T_cam_from_ref = np.linalg.inv(self.T_ref_from_camera)
        hom = np.concatenate([pts_ref, np.ones((len(pts_ref), 1))], axis=1)
        cam = (T_cam_from_ref @ hom.T).T[:, :3]
        valid = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1e-6)
        uv = np.full((len(cam), 2), np.nan)
        projected = (self.intrinsic @ cam[valid].T).T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
        return uv, valid

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "T_ref_from_camera": self.T_ref_from_camera.tolist(),
            "intrinsic": self.intrinsic.tolist(),
            "image_size": list(self.image_size),
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraParams":
        return cls(frame_id=d["frame_id"],
                   T_ref_from_camera=d["T_ref_from_camera"],
                   intrinsic=d["intrinsic"],
                   image_size=tuple(d["image_size"]),
                   reference=d.get("reference", "ref"))


def decompose_similarity(T_dst_from_src: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate positive-scale Sim(3); reject shear, reflection and projective transforms."""
    T = np.asarray(T_dst_from_src, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("transform must be affine homogeneous")
    scale = float(np.linalg.norm(T[:3, :3], axis=0).mean())
    if scale <= 1e-12:
        raise ValueError("transform scale must be positive")
    rotation = T[:3, :3] / scale
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("transform must have uniform positive scale and proper rotation")
    return scale, rotation, T[:3, 3].copy()


def transform_points(points: np.ndarray, T_dst_from_src: np.ndarray) -> np.ndarray:
    scale, rotation, translation = decompose_similarity(T_dst_from_src)
    return np.asarray(points, dtype=np.float64) @ (scale * rotation).T + translation


def transform_bbox(bbox: BBox3D, T_dst_from_src: np.ndarray, *, dst_reference: str) -> BBox3D:
    """Transform a box and explicitly name the destination coordinate reference."""
    scale, rotation, translation = decompose_similarity(T_dst_from_src)
    return BBox3D(
        center=scale * rotation @ bbox.center + translation,
        extents=scale * bbox.extents,
        axes=rotation @ bbox.axes,
        frame=dst_reference,
    )


def transform_camera(camera: CameraParams, T_dst_from_src: np.ndarray, *, dst_reference: str) -> CameraParams:
    """Transform camera centre with scale, orientation with rotation; preserve projection."""
    scale, rotation, translation = decompose_similarity(T_dst_from_src)
    pose = np.eye(4)
    pose[:3, :3] = rotation @ camera.T_ref_from_camera[:3, :3]
    pose[:3, 3] = scale * rotation @ camera.T_ref_from_camera[:3, 3] + translation
    return CameraParams(
        frame_id=camera.frame_id, T_ref_from_camera=pose,
        intrinsic=camera.intrinsic.copy(), image_size=camera.image_size,
        reference=dst_reference,
    )
