"""Real-model interface probes, NOT the completed perception or association method."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sam_probe(config: dict, image_dir: Path, seed: dict, seed_index: int, output: Path) -> dict:
    """One discovered object, bidirectional propagation over all frames; no confirmed IDs."""
    import torch
    import hashlib
    from sam2 import _C
    from sandbox_vlm.perception.features import pool_sam_region_feature
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = hf_hub_download(config["checkpoint"], config["checkpoint_file"],
                                 revision=config["revision"], local_files_only=True)
    predictor = build_sam2_video_predictor(config["config"], checkpoint, device="cuda")
    predictor.eval().requires_grad_(False)
    masks, features = {}, {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(image_dir))
        predictor.add_new_points_or_box(state, frame_idx=seed_index, obj_id=0,
                                         box=np.asarray(seed["bbox_xyxy"], dtype=np.float32))
        for reverse in (False, True):
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=seed_index, reverse=reverse,
            ):
                if object_ids != [0]:
                    raise RuntimeError("unexpected local probe object IDs")
                masks[frame_index] = (logits[0, 0] > 0).cpu().numpy()
        for frame_index in range(state["num_frames"]):
            # Native SAM2 resizes the whole image to a square (no crop/padding).
            # Therefore the original-resolution mask maps to the full feature grid.
            predictor._get_image_feature(state, frame_index, batch_size=1)
            grid = state["cached_features"][frame_index][1]["backbone_fpn"][-1][0]
            if masks[frame_index].any():
                features[frame_index] = pool_sam_region_feature(grid, masks[frame_index]).vector
        report = {
            "status": "ok", "purpose": "single_prompt_interface_probe_not_instance_association",
            "seed_index": seed_index, "seed_detection": seed, "n_frames": state["num_frames"],
            "candidate_mask_pixels": {str(k): int(v.sum()) for k, v in masks.items()},
            "feature_dims": {str(k): len(v) for k, v in features.items()},
            "feature_norms": {str(k): float(np.linalg.norm(v)) for k, v in features.items()},
            "feature_source": "frozen SAM2 image backbone_fpn[-1], before memory conditioning",
            "feature_pool": "area-resized mask weighted mean then L2 normalization",
            "native_extension": {"path": _C.__file__,
                                 "sha256": hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()},
            "warning": "Candidate masks are not visibility or identity ground truth; no global links created.",
        }
    np.savez_compressed(output / "sam_candidate_masks.npz", **{str(k): v for k, v in masks.items()})
    np.savez_compressed(output / "sam_region_features.npz", **{str(k): v for k, v in features.items()})
    return report


def vggt_probe(config: dict, image_paths: list[Path], frame_ids: list[int], output: Path) -> dict:
    """All-frame joint forward and camera convention roundtrip; no reliability threshold."""
    import torch
    from huggingface_hub import hf_hub_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from sandbox_vlm.memory.geometry import CameraParams

    checkpoint = hf_hub_download(config["checkpoint"], "model.pt", revision=config["revision"], local_files_only=True)
    model = VGGT()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    images = load_and_preprocess_images([str(p) for p in image_paths], mode="crop").cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, config["dtype"])):
        predictions = model(images)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    depth = predictions["depth"][0, ..., 0].float().cpu().numpy()
    confidence = predictions["depth_conf"][0].float().cpu().numpy()
    extrinsics = extrinsics[0].float().cpu().numpy()
    intrinsics = intrinsics[0].float().cpu().numpy()
    height, width = depth.shape[1:]
    cameras, pixel_errors = [], []
    for i, frame_id in enumerate(frame_ids):
        world_to_camera = np.eye(4)
        world_to_camera[:3] = extrinsics[i]
        camera = CameraParams(frame_id, np.linalg.inv(world_to_camera), intrinsics[i],
                              (height, width), reference="probe_local")
        cameras.append(camera.to_dict())
        yy, xx = np.mgrid[0:height:16, 0:width:16]
        z = depth[i, yy, xx].reshape(-1)
        pixels = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        valid = np.isfinite(z) & (z > 0)
        rays = np.column_stack([pixels[valid], np.ones(valid.sum())]) @ np.linalg.inv(intrinsics[i]).T
        camera_points = rays * z[valid, None]
        pose = camera.T_ref_from_camera
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        projected, positive = camera.project(world_points)
        if positive.any():
            pixel_errors.extend(np.linalg.norm(projected[positive] - pixels[valid][positive], axis=1).tolist())
    np.savez_compressed(output / "vggt_raw.npz", depth=depth, depth_conf=confidence,
                        T_camera_from_local=extrinsics, intrinsics=intrinsics)
    (output / "cameras.json").write_text(json.dumps(cameras, indent=2))
    finite_conf = confidence[np.isfinite(confidence)]
    return {
        "status": "ok", "n_frames": len(image_paths), "preprocessed_image_size_hw": [height, width],
        "preprocessing": "native VGGT load_and_preprocess_images(mode=crop); intrinsics in that grid",
        "finite_positive_depth_fraction": float((np.isfinite(depth) & (depth > 0)).mean()),
        "depth_conf_quantiles": np.quantile(finite_conf, [0, .25, .5, .75, 1]).tolist() if finite_conf.size else [],
        "backproject_project_max_error_px": max(pixel_errors) if pixel_errors else None,
        "scale": "arbitrary; no metric scale or GT input",
        "warning": "Roundtrip checks algebra/interface only, not geometric accuracy; no filtered object boxes produced.",
    }
