"""Real frozen StreamVGGT + SAM + discovery adapter for ContinuousSandMem."""
from pathlib import Path
import numpy as np

from sandbox_vlm.memory.geometry import CameraParams, BBox3D
from sandbox_vlm.perception.continuous_stream import ContinuousStreamVGGT
from sandbox_vlm.perception.features import pool_sam_region_feature
from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
from sandbox_vlm.perception.evidence import PromptEvidence


class ContinuousPerceptionFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


class ContinuousPerception:
    def __init__(self, config, frame_ids, *, discovery, sam_predictor,
                 stream_model, native_loader, pose_decoder, attempt):
        import torch
        self.config, self.discovery, self.sam = config, discovery, sam_predictor
        self.native_loader, self.pose_decoder, self.attempt = native_loader, pose_decoder, attempt
        self.captured = []
        self.hook = self.sam.model.image_encoder.register_forward_hook(
            lambda module, args, output: self.captured.append(output['backbone_fpn'][-1][0].detach()))
        geo = config['geometry']
        self.stream = ContinuousStreamVGGT(stream_model, reference=config['reference'],
            frame_ids=tuple(frame_ids), max_frames=len(frame_ids), cache_frames=geo['cache_frames_including_current'],
            camera_iterations=geo['camera_refinement_iterations'], cache_limit_bytes=geo['cache_limit_bytes'],
            head_context=lambda: torch.autocast('cuda', enabled=False))

    def frame(self, path, fid, index, *, discover, available_through):
        import torch
        from PIL import Image
        self.attempt('geometry')
        image = self.native_loader([str(path)], mode='crop').cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            raw = self.stream.step(image, frame_id=fid, sample_index=index, available_through=available_through)
        ext, intr = self.pose_decoder(raw['camera_pose'][:, None].float(), image.shape[-2:])
        pose = np.eye(4)
        pose[:3] = ext[0, 0].float().cpu().numpy()
        camera = CameraParams(fid, np.linalg.inv(pose), intr[0, 0].float().cpu().numpy(),
                              tuple(image.shape[-2:]), self.config['reference'])
        points = raw['pts3d_in_other_view'][0].float().cpu().numpy()
        confidence = raw['conf'][0].float().cpu().numpy()
        detections, discovery_receipt, rejected = [], None, []
        if discover:
            self.attempt('discovery')
            discovery_receipt = self.discovery.discover(path, fid, question=None)
            if discovery_receipt['status'] != 'ok':
                raise ContinuousPerceptionFailure('discovery failed; QA is prohibited',
                    dict(frame_id=fid, geometry_cost=raw['cost'], discovery=discovery_receipt))
            detections = discovery_receipt['detections']
        with Image.open(path) as source:
            rgb = np.asarray(source.convert('RGB'))
        self.captured.clear()
        self.attempt('sam_encoder')
        rows = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.sam.set_image(rgb)
            if len(self.captured) != 1:
                raise RuntimeError('SAM raw frozen encoder feature capture changed')
            feature = self.captured.pop()
            descriptor = feature.float().mean(dim=(1, 2)).cpu().numpy()
            transform = vggt_grid_transforms([tuple(rgb.shape[:2])], mode='crop')[0]
            if transform.output_hw != points.shape[:2]:
                raise ValueError('SAM source mask mapping differs from StreamVGGT native grid')
            for detection in detections:
                self.attempt('sam_prompt')
                masks, _, _ = self.sam.predict(box=np.asarray(detection['bbox_xyxy'], dtype=np.float32),
                                               multimask_output=False)
                mask = masks[0].astype(bool)
                if not mask.any():
                    rejected.append(dict(name=detection['name'], reason='empty_SAM_mask'))
                    continue
                appearance = pool_sam_region_feature(feature, mask).vector
                grid_mask = transform.mask_to_grid(mask)
                valid = grid_mask & np.isfinite(points).all(-1) & np.isfinite(confidence) & (confidence > 0)
                box = None
                if valid.any():
                    threshold = float(np.quantile(confidence[valid], .5))
                    subset = points[valid & (confidence >= threshold)]
                    if len(subset) >= self.config['object_box_min_points']:
                        lower, upper = subset.min(0), subset.max(0)
                        box = BBox3D((lower + upper) / 2, upper - lower, np.eye(3), camera.reference).to_dict()
                rows.append(dict(name=detection['name'], description=detection['short_description'],
                    appearance=appearance,
                    evidence=PromptEvidence(fid, tuple(detection['bbox_xyxy']), tuple(rgb.shape[:2]), True).to_dict(),
                    bbox_3d=box))
            self.sam.reset_predictor()
        return dict(camera=camera, descriptor=descriptor, world_points=points, confidence=confidence,
                    detections=rows, available_through=available_through), dict(
                        frame_id=fid, geometry_cost=raw['cost'], discovery=discovery_receipt, rejected=rejected)

    def close(self):
        self.hook.remove()
        self.captured.clear()
        self.sam.reset_predictor()
