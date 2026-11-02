"""All discovered box prompts -> conservative local SAM candidate tracks."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import time

from sandbox_vlm.memory import AppearanceFeature
from sandbox_vlm.perception.features import pool_sam_region_feature
from sandbox_vlm.perception.local_association import match_local_mask
from sandbox_vlm.perception.clip_cache import clip_image_feature_cache


@dataclass
class CandidateTrack:
    local_track_id: str
    seed_detection: dict
    masks: dict[int, np.ndarray]
    representative_feature: AppearanceFeature | None
    prompt_records: list[dict] = field(default_factory=list)
    association_status: str = "pending_physical_identity"
    candidate_track_ids: list[str] = field(default_factory=list)


class SamCandidateTracker:
    def __init__(self, config: dict):
        from huggingface_hub import hf_hub_download
        from sam2 import _C
        from sam2.build_sam import build_sam2_video_predictor
        self.extension_path = Path(_C.__file__)
        checkpoint = hf_hub_download(config["checkpoint"], config["checkpoint_file"],
                                     revision=config["revision"], local_files_only=True)
        self.predictor = build_sam2_video_predictor(config["config"], checkpoint, device="cuda")
        self.predictor.eval().requires_grad_(False)
        self.clip_feature_cache = config.get('clip_feature_cache', False)
        if type(self.clip_feature_cache) is not bool:
            raise ValueError('clip_feature_cache must be boolean')

    def track_clip(self, image_dir: Path, frame_ids: list[int], detections: list[dict], *,
                   clip_id: int, iou_threshold: float, sparse_feature_sink=None,
                   sparse_frame_ids=None, prompt_region_sink=None) -> tuple[list[CandidateTrack], dict]:
        """One bounded clip state reused between independent box prompts.

        Source images are loaded once. SAM state is reset per new prompt; masks
        from completed candidate tracks stay on CPU for same-frame matching.
        A uniquely reused track keeps its first prompt's mask/feature, and records
        the additional prompt as evidence without silently modifying geometry.
        """
        import torch
        started = time.perf_counter()
        if not frame_ids or frame_ids != sorted(set(frame_ids)):
            raise ValueError("ordered unique source frame IDs required")
        if any(d["frame_id"] not in frame_ids for d in detections):
            raise ValueError("prompt outside the clip")
        ordered = sorted(enumerate(detections), key=lambda item: (frame_ids.index(item[1]["frame_id"]), item[0]))
        tracks, failures, calls, reused = [], [], 0, 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), clip_image_feature_cache(
                self.predictor, len(frame_ids), enabled=self.clip_feature_cache) as cache_stats:
            state = self.predictor.init_state(str(image_dir))
            if state["num_frames"] != len(frame_ids):
                raise ValueError("SAM image directory and frame mapping differ")
            sparse_ids = list(sparse_frame_ids or [])
            if len(sparse_ids)>4 or len(sparse_ids)!=len(set(sparse_ids)) or not set(sparse_ids)<=set(frame_ids):
                raise ValueError('sparse SAM frames exceed the bounded clip')
            if sparse_ids and sparse_feature_sink is None:
                raise ValueError('missing sparse feature consumer')
            sparse_encoder_forwards=0
            for fid in sparse_ids:
                index=frame_ids.index(fid)
                sparse_encoder_forwards+=int(index not in state['cached_features'])
                self.predictor._get_image_feature(state,index,batch_size=1)
                sparse_feature_sink(fid,state['cached_features'][index][1]['backbone_fpn'][-1][0])
            for original_index, detection in ordered:
                anchor = frame_ids.index(detection["frame_id"])
                self.predictor.reset_state(state)
                _, ids, logits = self.predictor.add_new_points_or_box(
                    state, frame_idx=anchor, obj_id=0,
                    box=np.asarray(detection["bbox_xyxy"], dtype=np.float32))
                calls += 1
                if ids != [0]:
                    raise RuntimeError("unexpected independent SAM prompt identity")
                prompt_mask = (logits[0, 0] > 0).cpu().numpy()
                if not prompt_mask.any():
                    failures.append({"detection_index": original_index, "detection": detection,
                                     "reason": "empty_seed_mask"})
                    continue
                match = match_local_mask(prompt_mask,
                    {t.local_track_id: t.masks[detection["frame_id"]] for t in tracks}, threshold=iou_threshold)
                prompt_record = {"detection_index": original_index, "detection": detection, "match": asdict(match),
                                 "sam_mask_nonempty": True, "sam_mask_pixels": int(prompt_mask.sum())}
                if prompt_region_sink is not None and detection['frame_id'] in sparse_ids:
                    self.predictor._get_image_feature(state,anchor,batch_size=1)
                    region_grid=state['cached_features'][anchor][1]['backbone_fpn'][-1][0]
                    region_feature=pool_sam_region_feature(region_grid,prompt_mask)
                    local_id=match.matched_track_id or f"clip_{clip_id}/track_{len(tracks):03d}"
                    prompt_region_sink(detection,prompt_mask,region_feature,local_id)
                if match.matched_track_id is not None:
                    track = next(t for t in tracks if t.local_track_id == match.matched_track_id)
                    track.prompt_records.append(prompt_record)
                    reused += 1
                    continue
                self.predictor._get_image_feature(state, anchor, batch_size=1)
                grid = state["cached_features"][anchor][1]["backbone_fpn"][-1][0]
                feature = pool_sam_region_feature(grid, prompt_mask)
                masks = {}
                for reverse in (False, True):
                    for index, object_ids, logits in self.predictor.propagate_in_video(
                            state, start_frame_idx=anchor, reverse=reverse):
                        if object_ids != [0]:
                            raise RuntimeError("unexpected candidate propagation identity")
                        masks[frame_ids[index]] = (logits[0, 0] > 0).cpu().numpy()
                if set(masks) != set(frame_ids):
                    raise RuntimeError("candidate track omitted processed source frames")
                tracks.append(CandidateTrack(f"clip_{clip_id}/track_{len(tracks):03d}", detection,
                    masks, feature, [prompt_record], candidate_track_ids=match.candidate_track_ids))
            self.predictor.reset_state(state)
        torch.cuda.synchronize()
        return tracks, {"box_prompt_calls": calls, "propagated_tracks": len(tracks),
                        "sparse_feature_frame_ids": sparse_ids, "wall_seconds": time.perf_counter()-started,
                        "sparse_stage_image_encoder_forward_calls": sparse_encoder_forwards,
                        "initialization_image_encoder_forwards": 1,
                        "reused_local_prompts": reused, "prompt_failures": failures,
                        "image_feature_cache": cache_stats,
                        "iou_threshold": iou_threshold, "confirmed_physical_instances": 0}

    def encode_sparse_frames(self, image_dir, frame_ids, requested_ids, consumer):
        """Discovery-independent frozen encoder diagnostic, no masks/identity claims."""
        import torch
        if len(requested_ids)>4 or len(requested_ids)!=len(set(requested_ids)) or not set(requested_ids)<=set(frame_ids):
            raise ValueError('invalid bounded feature frame request')
        started=time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            state=self.predictor.init_state(str(image_dir))
            if state['num_frames']!=len(frame_ids):
                raise ValueError('SAM/source frame mapping mismatch')
            encoder_forwards=1  # Official init_state warms frame zero once.
            for fid in requested_ids:
                index=frame_ids.index(fid)
                encoder_forwards+=int(index not in state['cached_features'])
                self.predictor._get_image_feature(state,index,batch_size=1)
                consumer(fid,state['cached_features'][index][1]['backbone_fpn'][-1][0])
            self.predictor.reset_state(state)
        torch.cuda.synchronize()
        return dict(purpose='frozen_encoder_only_no_masks',requested_frame_ids=requested_ids,
            image_encoder_requested_frames=len(requested_ids),sam_initialization_warmup_frame_id=frame_ids[0],
            image_encoder_forward_calls=encoder_forwards,
            input_clip_frames=len(frame_ids),wall_seconds=time.perf_counter()-started)
