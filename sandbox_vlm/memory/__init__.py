"""长期沙盒记忆。"""
from sandbox_vlm.memory.geometry import BBox3D, CameraParams, transform_bbox, transform_camera, transform_points
from sandbox_vlm.memory.sandbox import (
    AmbiguityNote,
    AppearanceFeature,
    ObjectInstance,
    Sandbox,
    SpatialExtent,
    Viewpoint,
)

__all__ = [
    "BBox3D", "CameraParams", "transform_bbox", "transform_camera", "transform_points",
    "Sandbox", "SpatialExtent", "Viewpoint", "ObjectInstance",
    "AppearanceFeature", "AmbiguityNote",
]
