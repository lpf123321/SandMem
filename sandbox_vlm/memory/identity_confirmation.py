"""D098 actual-region identity chains and retrieval; no parent-track promotion."""
from itertools import combinations

from sandbox_vlm.memory.identity_evidence import region_key
from sandbox_vlm.qa.selection import terms


def identity_components(regions, relations, cannot_links):
    by_key = {r['region_key']: r for r in regions}
    for key, r in by_key.items():
        if key != region_key(r['region']):
            raise ValueError('actual region key changed')
    edges = set()
    neighbors = {}
    for relation in relations:
        if relation['status'] != 'supported_local_relation':
            continue
        a, b = relation['region_keys']
        if a == b or a not in by_key or b not in by_key:
            raise ValueError('identity edge lacks two actual endpoints')
        checks = relation['shared_frame_checks']
        if (not checks or any(not r['nonempty'] or not r['iou'] >= .70 for r in checks)
                or relation['forward_candidates'] != [b] or relation['backward_candidates'] != [a]
                or not {by_key[a]['region']['frame_id'], by_key[b]['region']['frame_id']}
                <= {r['frame_id'] for r in checks}):
            raise ValueError('identity edge lacks complete endpoint support')
        edges.add(tuple(sorted([a, b])))
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    cannot = {tuple(sorted(pair)) for pair in cannot_links}
    seen = set()
    accepted, rejected = [], []
    for key in sorted(neighbors):
        if key in seen:
            continue
        todo, group = [key], set()
        while todo:
            node = todo.pop()
            if node not in group:
                group.add(node)
                todo.extend(neighbors[node] - group)
        seen |= group
        members = sorted(group)
        conflicts = sorted(set(combinations(members, 2)) & cannot)
        if conflicts:
            rejected.append(dict(members=members, reason='actual_co_visible_conflict', conflicts=conflicts))
        else:
            accepted.append(members)
    return dict(accepted=accepted, rejected=rejected, direct_edges=sorted(edges),
        rule='connected direct-evidence chains sharing identical actual region keys; no name-only bridges',
        identity_scope='checked_actual_regions_only', ground_truth_identity_verified=False)


def retrieval(base, regions, components, baseline, question, *, candidate, maximum=31):
    """Change evidence using verified region links, keeping all other actuals unknown."""
    question_terms = terms(question)
    by_key = {r['region_key']: r for r in regions}
    membership = {key: i for i, group in enumerate(components) for key in group}
    relevant_groups = [i for i, group in enumerate(components)
        if any(question_terms & terms(by_key[k]['name']) for k in group)]
    mandatory = sorted({by_key[k]['region']['frame_id'] for i in relevant_groups for k in components[i]}) if candidate else []
    if len(mandatory) > maximum:
        raise ValueError('relevant identity endpoints exceed QA frame budget')
    if not candidate:
        return dict(frame_ids=list(baseline), mandatory_identity_frames=[], relevant_groups=[],
                    method='original D094/D096 temporal quantiles')
    coverage = {}
    for frame, links in base['viewpoints'].items():
        for link in links:
            r = base['records'][link['instance_id']]
            if not question_terms & terms(r['name'] + ' ' + r['description']):
                continue
            key = region_key(link['evidence'])
            target = ('identity', membership[key]) if key in membership else ('unknown_track', link['instance_id'])
            coverage.setdefault(int(frame), set()).add(target)
    available = sorted(set(map(int, base['viewpoints'])) | set(baseline))
    chosen = list(mandatory)
    covered = set().union(*(coverage.get(f, set()) for f in chosen)) if chosen else set()
    while len(chosen) < min(maximum, len(available)):
        candidates = [f for f in available if f not in chosen]
        def priority(fid):
            distance = min(abs(fid-old) for old in chosen) if chosen else 0
            return len(coverage.get(fid, set()) - covered), distance, fid in baseline, -fid
        fid = max(candidates, key=priority)
        chosen.append(fid)
        covered |= coverage.get(fid, set())
    return dict(frame_ids=sorted(chosen), mandatory_identity_frames=mandatory,
        relevant_groups=relevant_groups, method='actual identity endpoints, new relevant coverage, complementary times')
