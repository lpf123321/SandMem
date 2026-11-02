"""Separate lifetime discovery coverage from a fixed per-answer visual budget."""
import math
import numpy as np

from sandbox_vlm.perception.coverage import unpack_descriptor
from sandbox_vlm.qa.relevance import select_relevant


def allocate_discovery(ledger, *, query, total_frames, selected, previous_query,
                       minimum=64, per_minute=2., maximum=512, sparse_maximum=8,
                       sparse_fraction=.05, redundant_context_cosine=.995, positive_only=False):
    ids = sorted(ledger)
    if (not ids or ids[-1] > query or query >= total_frames or len(selected) != len(set(selected))
            or any(fid > previous_query or fid not in ledger for fid in selected)):
        raise ValueError('invalid causal discovery ledger')
    sparse = sum(ledger[f]['margin'] > 0 for f in ids) / len(ids) <= sparse_fraction
    cap = sparse_maximum if sparse else min(maximum, max(minimum, math.ceil(total_frames / 60 * per_minute)))
    quota = min(cap, max(1, math.ceil(cap * (query + 1) / total_frames)))
    available = [fid for fid in ids if previous_query < fid <= query]
    negative_excluded=0
    if positive_only and not sparse:
        negative_excluded=sum(ledger[f]['margin']<=0 for f in available)
        available=[f for f in available if ledger[f]['margin']>0]
    count = min(max(0, quota-len(selected)), len(available))
    if count == 0:
        return [], dict(sparse=sparse, quota=quota, lifetime_cap=cap, redundant_skipped=0,
                        negative_excluded=negative_excluded)
    if sparse:
        ranking = sorted(available, key=lambda fid: (-ledger[fid]['margin'], fid))
    else:
        # Preserve the coverage-first ranking of the requested quota, then use
        # relevance to fill slots rejected as visually redundant.
        primary = select_relevant(available, [ledger[f]['margin'] for f in available], query=query,
                                  count=count, separation=3)
        ranking = primary + sorted(set(available)-set(primary), key=lambda fid: (-ledger[fid]['margin'], fid))
    new, skipped = [], 0
    descriptors = {f: unpack_descriptor(ledger[f]['descriptor']).reshape(-1) for f in selected + available}
    for f in descriptors:
        descriptors[f] /= max(float(np.linalg.norm(descriptors[f])), 1e-12)
    for fid in ranking:
        if len(new) == count:
            break
        descriptor = descriptors[fid]
        if not sparse and any(float(descriptor @ descriptors[old]) >= redundant_context_cosine
                              for old in selected + new):
            skipped += 1
            continue
        new.append(fid)
    return sorted(new), dict(sparse=sparse, quota=quota, lifetime_cap=cap, redundant_skipped=skipped,
                             negative_excluded=negative_excluded)
