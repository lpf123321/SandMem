"""Causal, video-lifetime D041 perception allocation and bounded QA legends."""
import math

from sandbox_vlm.qa.relevance import select_relevant
from sandbox_vlm.qa.selection import terms


def allocate_frames(score_state, *, query, total_frames, selected, previous_query,
                    maximum=32, sparse_maximum=8, sparse_fraction=.05):
    ids = score_state.ids[:score_state.size].tolist()
    values = score_state.margins[:score_state.size].tolist()
    if (not ids or max(ids) > query or total_frames <= query or len(set(selected)) != len(selected)
            or len(selected) > maximum or any(f > previous_query for f in selected)):
        raise ValueError('invalid causal frame bank')
    sparse = sum(s > 0 for s in values) / len(ids) <= sparse_fraction
    cap = sparse_maximum if sparse else maximum
    quota = min(cap, max(1, math.ceil(maximum * (query + 1) / total_frames)))
    available = [(f, s) for f, s in zip(ids, values) if previous_query < f <= query]
    count = min(max(0, quota - len(selected)), len(available))
    if count == 0:
        return [], dict(sparse=sparse, quota=quota, lifetime_maximum=maximum)
    if sparse:
        chosen = sorted(f for f, _ in sorted(available, key=lambda p: (-p[1], p[0]))[:count])
    else:
        chosen = select_relevant([p[0] for p in available], [p[1] for p in available],
                                 query=query, count=count, separation=3)
    return chosen, dict(sparse=sparse, quota=quota, lifetime_maximum=maximum)


def visual_legend(state, question, *, maximum=64, max_notes=16):
    query = terms(question)
    scores = {i['visual_instance_id']: len(query & terms(i['name'] + ' ' + i['description']))
              for i in state.instances}
    candidates = [i for i in state.instances if scores[i['visual_instance_id']] > 0]
    fallback = not candidates
    if fallback:
        candidates = list(state.instances)
    candidates.sort(key=lambda i: (-scores[i['visual_instance_id']], i['visual_instance_id']))
    chosen = candidates[:maximum]
    links = {}
    for fid, views in state.viewpoints.items():
        for view in views:
            links.setdefault(view['visual_instance_id'], []).append(int(fid))
    legend = [dict(visual_id=i['visual_instance_id'], name=i['name'], description=i['description'][:64],
                   identity_status='visual_hypothesis', actual_frame_ids=sorted(set(links.get(i['visual_instance_id'], []))),
                   geometry_status='local_references_only' if i['local_geometry'] else 'not_available') for i in chosen]
    relevant_notes = [n for n in state.ambiguities if query & terms(n['name'] + ' ' + n['description'])]
    notes = [dict(name=n['name'], frame_id=n['evidence']['frame_id'],
                  candidate_visual_ids=n['candidate_visual_instance_ids'][:5], reason=n['reason'])
             for n in relevant_notes[:max_notes]]
    audit = dict(lexical_fallback=fallback, total_hypotheses=len(state.instances),
        omitted_relevant_hypothesis_ids=[i['visual_instance_id'] for i in candidates[maximum:]],
        omitted_relevant_note_count=max(0, len(relevant_notes) - max_notes),
        descriptions_shortened_to_characters=64, maximum_note_candidate_ids=5)
    return legend, notes, audit
