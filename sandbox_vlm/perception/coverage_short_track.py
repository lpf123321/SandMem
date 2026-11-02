"""D102 repairs SAM scheduling; correspondence and geometric gates stay frozen.

Only scalar coverage metadata survives a window. There is no observation pool,
new identity verifier, VGGT tracking, or appearance-only global merge.
"""
from collections import defaultdict
from copy import deepcopy
import math

from sandbox_vlm.alignment.association import names_compatible
from sandbox_vlm.memory.short_track import observation_key, observations
from sandbox_vlm.perception.short_track import BoundedShortTrack


class CoverageShortTrack(BoundedShortTrack):
    def __init__(self, tracker):
        super().__init__(tracker)
        self.processed_through = {}
        self.failed_source_keys = set()
        self.new_seed_attempts = 0
        self.new_edges = []
        self.legacy_seed_attempts = 0
        self.last_watermark = -1
        self.skipped_seed_groups = []

    def import_legacy(self, state, legacy, clips, *, watermark):
        if watermark <= self.last_watermark:
            raise ValueError('non-increasing streaming prefix')
        by_frame = defaultdict(list)
        for o in observations(state):
            if o['evidence']['frame_id'] > watermark: raise ValueError('future source evidence')
            by_frame[o['evidence']['frame_id']].append(o)
        for clip in clips:
            sf, end = clip['source_frame'], clip['end_frame']
            if not sf <= end <= min(sf+15, watermark): raise ValueError('future or detached legacy window')
            for o in by_frame[sf]:
                key = observation_key(o)
                # A failed old attempt is preserved and never retried.
                if clip['status']!='completed': self.failed_source_keys.add(key)
                self.processed_through[key] = max(self.processed_through.get(key,-1),end)
        if legacy['seed_attempts'] < self.legacy_seed_attempts:
            raise ValueError('legacy cost cannot decrease')
        self.legacy_seed_attempts = legacy['seed_attempts']
        self.attempted |= set(legacy['attempted_observation_keys'])
        self.accepted_edges = deepcopy(legacy['accepted_edges']) + deepcopy(self.new_edges)
        self.seed_attempts = self.legacy_seed_attempts + self.new_seed_attempts
        self.last_watermark = watermark

    def select_groups(self, obs, *, watermark):
        if watermark != self.last_watermark or any(o['evidence']['frame_id']>watermark for o in obs):
            raise ValueError('unimported or future scheduling prefix')
        # One fixed coverage policy, set before outcomes: 32 seed attempts per
        # 64 actual observations, at most 128 total including inherited work.
        self.maximum_seeds = min(128,32*max(1,math.ceil(len(obs)/64)))
        grouped=defaultdict(list)
        for o in obs: grouped[o['evidence']['frame_id']].append(o)
        # Reserve the complete inherited 32-seed allocation: later prefixes
        # may consume it, so unused old slots are not a second free budget.
        groups=[];used=32+self.new_seed_attempts;self.skipped_seed_groups=[]
        for sf,seeds in sorted(grouped.items()):
            if any(observation_key(s) in self.failed_source_keys for s in seeds): continue
            targets=[o for o in obs if sf<o['evidence']['frame_id']<=min(sf+15,watermark)]
            eligible=any(t['evidence']['frame_id']>self.processed_through.get(observation_key(s),-1)
                and names_compatible(s['name'],t['name'],'last_token') for s in seeds for t in targets)
            if not eligible: continue
            if used+len(seeds)>self.maximum_seeds:
                self.skipped_seed_groups.append(dict(source_frame=sf,seeds=len(seeds),reason='fixed_coverage_budget'))
                continue
            groups.append((sf,seeds));used+=len(seeds)
        return groups

    def update(self,state,reader,*,watermark,audit_dir):
        obs=observations(state)
        old_edge_count=len(self.accepted_edges);old_attempts=self.seed_attempts
        result,audit=super().update(state,reader,watermark=watermark,audit_dir=audit_dir)
        self.new_seed_attempts += self.seed_attempts-old_attempts
        self.new_edges += deepcopy(self.accepted_edges[old_edge_count:])
        for clip in audit['clips']:
            sf=clip['source_frame'];covered=clip['end_frame']
            for o in obs:
                if o['evidence']['frame_id']==sf:
                    key=observation_key(o)
                    if clip['status']!='completed':self.failed_source_keys.add(key)
                    self.processed_through[key]=max(self.processed_through.get(key,-1),covered)
        audit['coverage_schedule']=dict(maximum_total_seeds=self.maximum_seeds,
            inherited_seed_attempts=self.legacy_seed_attempts,new_seed_attempts=self.new_seed_attempts,
            processed_through=dict(self.processed_through),skipped_groups=deepcopy(self.skipped_seed_groups))
        return result,audit

    def to_dict(self):
        return dict(policy='D102_actual_window_coverage',maximum_total_seeds=self.maximum_seeds,
            new_seed_attempts=self.new_seed_attempts,inherited_seed_attempts=self.legacy_seed_attempts,
            processed_through=dict(self.processed_through),new_edges=deepcopy(self.new_edges),
            failed_source_keys=sorted(self.failed_source_keys),
            last_watermark=self.last_watermark)
