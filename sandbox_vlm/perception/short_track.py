"""Causal, input-only selection and bounded SAM propagation for D050."""
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from sandbox_vlm.alignment.association import names_compatible
from sandbox_vlm.alignment.short_track import evaluate_tracks
from sandbox_vlm.memory.short_track import observations, observation_key, merge_correspondences


class BoundedShortTrack:
    def __init__(self, tracker, *, maximum_seeds=32, threshold=.7):
        if maximum_seeds != 32 or threshold != .7:
            raise ValueError('D050 fixed bounds, no threshold search')
        self.tracker, self.maximum_seeds, self.threshold = tracker, maximum_seeds, threshold
        self.attempted = set()
        self.seed_attempts = 0
        self.history = []
        self.accepted_edges = []

    def to_dict(self):
        return dict(maximum_seeds=self.maximum_seeds, threshold=self.threshold,
                    attempted_observation_keys=sorted(self.attempted), seed_attempts=self.seed_attempts,
                    history=self.history, accepted_edges=self.accepted_edges)

    def select_groups(self, obs, *, watermark):
        grouped = {}
        for o in obs:
            grouped.setdefault(o['evidence']['frame_id'], []).append(o)
        groups = []
        for sf, seeds in grouped.items():
            targets = [o for o in obs if sf < o['evidence']['frame_id'] <= min(sf+15, watermark)]
            if (any(observation_key(o) in self.attempted for o in seeds)
                    or not any(names_compatible(s['name'], t['name'], 'last_token') for s in seeds for t in targets)):
                continue
            if self.seed_attempts + sum(len(s) for _, s in groups) + len(seeds) > self.maximum_seeds:
                continue
            groups.append((sf, seeds))
        return groups

    def update(self, state, reader, *, watermark, audit_dir):
        import torch
        from sandbox_vlm.perception.clip_cache import clip_image_feature_cache
        obs = observations(state)
        if any(o['evidence']['frame_id'] > watermark for o in obs):
            raise ValueError('future observation in short tracker')
        groups = self.select_groups(obs, watermark=watermark)
        audit_dir = Path(audit_dir); audit_dir.mkdir(exist_ok=False)
        all_edges = []; started = time.perf_counter(); rows = []
        for sf, seeds in groups:
            self.seed_attempts += len(seeds)
            self.attempted.update(observation_key(o) for o in seeds)
            end = min(sf+15, watermark)
            local = [o for o in obs if sf <= o['evidence']['frame_id'] <= end]
            seed_ids = [o['observation_id'] for o in seeds]
            masks, tracks = {}, {}
            row = dict(source_frame=sf, end_frame=end, seed_attempts=len(seeds),
                       actual_prompt_attempts=0, propagated_object_frames=0, status='running')
            inference = None; begin = time.perf_counter()
            try:
                with tempfile.TemporaryDirectory(prefix='short-track-') as temporary:
                    source_started = time.perf_counter()
                    for data, ref, _ in reader.encoded_sequence(list(range(sf, end+1))):
                        (Path(temporary)/f'{ref.frame_id:06d}.jpg').write_bytes(data)
                    row['source_decode_write_seconds'] = time.perf_counter() - source_started
                    predictor = self.tracker.predictor
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), clip_image_feature_cache(
                            predictor, end-sf+1, enabled=True) as cache:
                        inference = predictor.init_state(temporary)
                        for o in local:
                            oid = o['observation_id']; fid = o['evidence']['frame_id']
                            predictor.reset_state(inference); row['actual_prompt_attempts'] += 1
                            _, ids, logits = predictor.add_new_points_or_box(inference, frame_idx=fid-sf, obj_id=0,
                                box=np.asarray(o['evidence']['bbox_xyxy'], dtype=np.float32))
                            if ids != [0]: raise ValueError('unexpected SAM object IDs')
                            masks[oid] = (logits[0,0] > 0).cpu().numpy().copy()
                            if list(masks[oid].shape) != o['evidence']['source_hw']:
                                raise ValueError('source mask grid mismatch')
                            if oid in seed_ids:
                                tracks[oid] = {}
                                if masks[oid].any():
                                    for index, ids, logits in predictor.propagate_in_video(
                                            inference, start_frame_idx=fid-sf, reverse=False):
                                        if ids != [0] or not fid <= index+sf <= end:
                                            raise ValueError('noncausal SAM propagation')
                                        tracks[oid][index+sf] = (logits[0,0] > 0).cpu().numpy().copy()
                                        row['propagated_object_frames'] += 1
                        predictor.reset_state(inference)
                    row['image_feature_cache'] = dict(cache)
                edges = evaluate_tracks(local, seed_ids, masks, tracks, threshold=self.threshold)
                # A propagated mask cannot override incompatible observed names.
                by_id = {o['observation_id']: o for o in local}
                for edge in edges:
                    if not names_compatible(by_id[edge['source']]['name'], by_id[edge['target']]['name'], 'last_token'):
                        edge.update(would_reuse=False, reason='incompatible_observed_names')
                all_edges.extend(edges)
                row.update(status='completed', correspondences=edges)
            except Exception as exc:
                row.update(status='failed', error=str(exc), traceback=traceback.format_exc())
                # An incomplete group contributes no accepted edge and is not retried.
            finally:
                arrays = {f'prompt__{oid}': a for oid, a in masks.items()}
                arrays.update({f'track__{oid}__F{fid}': a for oid, track in tracks.items() for fid, a in track.items()})
                np.savez_compressed(audit_dir/f'F{sf:06d}_masks.npz', **arrays)
                row.update(wall_seconds=time.perf_counter()-begin,
                           temporary_CPU_mask_bytes=sum(a.nbytes for a in arrays.values()))
                rows.append(row)
                del inference, masks, tracks, arrays
        by_id = {o['observation_id']: o for o in obs}
        for edge in all_edges:
            if edge['would_reuse']:
                self.accepted_edges.append(dict(edge,
                    source_key=observation_key(by_id[edge['source']]),
                    target_key=observation_key(by_id[edge['target']])))
        for edge in self.accepted_edges:
            if (edge['source_key'] != observation_key(by_id[edge['source']]) or
                    edge['target_key'] != observation_key(by_id[edge['target']])):
                raise ValueError('persistent short correspondence endpoint changed')
        result, union = merge_correspondences(state, obs, self.accepted_edges,
                                             watermark=watermark, threshold=self.threshold)
        summary = dict(watermark=watermark, groups=len(rows),
            seed_attempts=sum(r['seed_attempts'] for r in rows),
            prompt_attempts=sum(r['actual_prompt_attempts'] for r in rows),
            failures=sum(r['status'] != 'completed' for r in rows),
            propagated_object_frames=sum(r['propagated_object_frames'] for r in rows),
            removed_hypotheses=union['removed_hypotheses'], wall_seconds=time.perf_counter()-started)
        self.history.append(summary)
        return result, dict(summary=summary, clips=rows, union=union)
