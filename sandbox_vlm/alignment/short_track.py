"""Read-only bounded SAM correspondence tests; no persistent-ID updates."""
import numpy as np


def mask_iou(a, b):
    if a.dtype != np.bool_ or b.dtype != np.bool_ or a.ndim != 2 or a.shape != b.shape or not all(a.shape):
        raise ValueError('masks need identical boolean source grids')
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 0.


def evaluate_tracks(observations, seeds, prompt_masks, propagated, *, threshold=.7):
    """Require a nonempty forward path and unique mutual endpoint mask match.

    Only co-visible source observations compete with one another. Observations
    from different source times may describe one object and cannot be treated
    as distinct identities merely because their original IDs differ.
    """
    if not 0 < threshold <= 1:
        raise ValueError('invalid threshold')
    by_id = {o['observation_id']: o for o in observations}
    if len(by_id) != len(observations) or not set(seeds) <= set(by_id):
        raise ValueError('invalid actual observation identity')
    grouped = {}
    for oid, obs in by_id.items():
        fid = obs['evidence']['frame_id']
        if not obs['evidence']['sam_mask_nonempty'] or oid not in prompt_masks:
            raise ValueError('missing actual prompt evidence')
        grouped.setdefault(fid, []).append(oid)
    raw = {}
    for seed in seeds:
        sf = by_id[seed]['evidence']['frame_id']
        for tf, ids in grouped.items():
            if tf < sf:
                continue
            pred = propagated.get(seed, {}).get(tf)
            for target in ids:
                raw[(seed, target)] = mask_iou(pred, prompt_masks[target]) if pred is not None else 0.
    rows = []
    for (seed, target), score in raw.items():
        if seed == target:
            continue
        sf, tf = by_id[seed]['evidence']['frame_id'], by_id[target]['evidence']['frame_id']
        track = propagated.get(seed, {})
        continuous = all(f in track and track[f].any() for f in range(sf, tf+1))
        forward_candidates = [oid for oid in grouped[tf] if raw.get((seed, oid), 0.) >= threshold]
        peer_seeds = [s for s in seeds if by_id[s]['evidence']['frame_id'] == sf]
        reverse_candidates = [s for s in peer_seeds if raw.get((s, target), 0.) >= threshold]
        accepted = (sf < tf and continuous and prompt_masks[target].any() and
            forward_candidates == [target] and reverse_candidates == [seed])
        reason = ('same_frame_no_identity_upgrade' if sf == tf else
            'empty_or_missing_track' if not continuous else
            'below_mask_iou' if score < threshold else
            'ambiguous_mask_correspondence' if not accepted else 'unique_continuous_mask_correspondence')
        rows.append(dict(source=seed, target=target, source_frame=sf, target_frame=tf,
            mask_iou=score, continuous_nonempty=bool(continuous),
            endpoint_prompt_nonempty=bool(prompt_masks[target].any()),
            forward_candidates=forward_candidates, co_visible_reverse_candidates=reverse_candidates,
            would_reuse=bool(accepted), reason=reason))
    return rows


def reviewed_outcomes(rows, reviewed_pairs, prompt_masks, *, threshold=.7):
    lookup = {(r['source'], r['target']): r for r in rows}
    outcomes = []
    for pair in reviewed_pairs:
        key = pair['source'], pair['target']
        row = lookup.get(key)
        both_supported = bool(prompt_masks[key[0]].any() and prompt_masks[key[1]].any())
        if pair['relation'] == 'same_visible_object':
            passed = row is not None and row['would_reuse']
            score = row['mask_iou'] if row else None
        elif pair['relation'] == 'distinct_covisible_objects':
            score = mask_iou(prompt_masks[key[0]], prompt_masks[key[1]])
            # Same-frame veto alone is not a valid segmentation separation test.
            passed = both_supported and score < threshold and (row is None or not row['would_reuse'])
        elif pair['relation'] == 'distinct_across_frames':
            score = row['mask_iou'] if row else None
            passed = both_supported and row is not None and not row['would_reuse']
        else:
            raise ValueError('unreviewed relation cannot become diagnostic truth')
        outcomes.append(dict(**pair, passed=bool(passed), mask_iou=score,
            both_actual_masks_nonempty=both_supported, would_reuse=row['would_reuse'] if row else False))
    return outcomes
