"""Strict complete-answer and trace validation; no generation repair or retry."""
import json

ANSWER_SCHEMA = ('{"answer":"concise answer, or explicitly insufficient evidence",'
                 '"used_instance_ids":[],"used_frame_ids":[],"match_requests":[]}')


def parse_answer(raw, metadata, sandbox):
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('answer is not complete JSON') from exc
    if not isinstance(value, dict) or set(value) != {'answer', 'used_instance_ids', 'used_frame_ids', 'match_requests'}:
        raise ValueError('answer schema mismatch')
    if not isinstance(value['answer'], str) or not value['answer'].strip():
        raise ValueError('empty answer')
    for key, allowed in [('used_instance_ids', metadata['selection']['instance_ids']),
                         ('used_frame_ids', metadata['selection']['frame_ids'])]:
        ids = value[key]
        if (not isinstance(ids, list) or any(type(i) is not int for i in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= set(allowed)):
            raise ValueError('invalid or unavailable evidence citation: ' + key)
    if not isinstance(value['match_requests'], list) or any(not isinstance(r, dict) for r in value['match_requests']):
        raise ValueError('invalid revision requests')
    # Every cited confirmed identity must be grounded in an actually provided frame.
    for gid in value['used_instance_ids']:
        if not set(sandbox.frames_for_instance(gid)) & set(value['used_frame_ids']):
            raise ValueError('instance citation has no cited actual observation frame')
    return value
