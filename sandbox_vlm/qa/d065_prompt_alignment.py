"""D110: D065 instructions with unchanged D107 evidence and geometry pixels."""
import json
from sandbox_vlm.qa.prompt import messages_for
from sandbox_vlm.qa.render import SPACE_NOTICE


MAP_NOTICE = ('Supplemental estimated local 3D sandbox: I = record ID, F = actual frame ID; '
              'arbitrary scale.\n' + SPACE_NOTICE)


def aligned_messages(snapshot, images, diagram, legacy_notice):
    """Only adapt the legacy unavailable-geometry sentence to actual map input.

    D107's complete payload is data, not a new instruction template. Preserve
    its JSON field values exactly; retain the original D065 frame labels,
    question placement, density-based response policy and final reminder.
    """
    if legacy_notice.count('Geometry is unavailable for this run.') != 1:
        raise ValueError('unexpected frozen D065 memory notice')
    notice = legacy_notice.replace('Geometry is unavailable for this run.',
        'Estimated local geometry is supplied in the sandbox diagram.')
    messages = messages_for(snapshot['question'], images, snapshot['frame_ids'],
                            'brief_evidence' if snapshot['sparse'] else 'question_last')
    content = messages[0]['content']
    content[0]['text'] += notice + json.dumps(snapshot['payload'], ensure_ascii=False, separators=(',', ':'))
    content[1:1] = [dict(type='text', text=MAP_NOTICE), dict(type='image', image=diagram)]
    return messages, [diagram, *images]


def extract_d065_answer(raw, *, sparse):
    """Same strict last-line rule as D065, without fallback or truncation rescue."""
    answer = raw.strip()
    if sparse:
        last = answer.splitlines()[-1] if answer else ''
        if not last.startswith('FINAL_ANSWER:'):
            raise ValueError('missing exact final line; no repair')
        answer = last.removeprefix('FINAL_ANSWER:').strip()
    if not answer:
        raise ValueError('empty answer')
    return answer
