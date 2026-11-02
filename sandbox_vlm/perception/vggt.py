"""Reusable frozen VGGT clip inference, with explicit camera convention."""
from pathlib import Path
import time

import numpy as np

from sandbox_vlm.memory import CameraParams


class VggtGeometry:
    def __init__(self, config):
        import torch
        from huggingface_hub import hf_hub_download
        from vggt.models.vggt import VGGT
        self.config = dict(config)
        checkpoint = hf_hub_download(config['checkpoint'], 'model.pt',
            revision=config['revision'], local_files_only=True)
        self.model = VGGT()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.model = self.model.cuda().eval().requires_grad_(False)

    def infer(self, paths, frame_ids, *, reference):
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        if not paths or len(paths) != len(frame_ids) or len(paths) > self.config['max_views']:
            raise ValueError('invalid VGGT clip length')
        started = time.perf_counter()
        images = load_and_preprocess_images([str(Path(p)) for p in paths], mode='crop').cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=getattr(torch, self.config['dtype'])):
            predictions = self.model(images)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
        depth = predictions['depth'][0, ..., 0].float().cpu().numpy()
        confidence = predictions['depth_conf'][0].float().cpu().numpy()
        extrinsics, intrinsics = extrinsics[0].float().cpu().numpy(), intrinsics[0].float().cpu().numpy()
        cameras = []
        for fid, extrinsic, intrinsic in zip(frame_ids, extrinsics, intrinsics):
            pose = np.eye(4); pose[:3] = extrinsic
            cameras.append(CameraParams(fid, np.linalg.inv(pose), intrinsic, depth.shape[1:], reference))
        torch.cuda.synchronize()
        return dict(depth=depth, confidence=confidence, cameras=cameras,
                    cost=dict(model_calls=1, wall_seconds=time.perf_counter()-started,
                              input_frames=len(paths), scale='arbitrary'))
