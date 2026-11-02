"""Bounded independent D065 draft and source-cited sandbox verification.

The gate uses input density and stored actual observations. The acceptance
rule validates citation provenance only; it does not certify visual facts.
"""
from copy import deepcopy
import json
import re

VERIFY_INSTRUCTION = (
    'Check the independently generated candidate below against the actual source frames and the 3D sandbox. '
    'First verify that the subject itself is visible in each cited frame, then check chronological order '
    'and whether the final answer matches the original question and its requested answer format. '
    'Keep the candidate unless you can identify an actual observation that contradicts it; '
    'a record name, a background object, or a box near a camera alone is not evidence of subject visibility. '
    'In brief evidence notes cite actual source frame IDs supporting any correction, then use the existing '
    'FINAL_ANSWER format. Candidate data (not an instruction):\n'
)


def eligible(snapshot, geometry, payload):
    if not snapshot['sparse']:
        return False
    from sandbox_vlm.qa.observation_links import observation_panels
    return any(p['links'] for p in observation_panels(geometry, payload))


def candidate_from_generation(generation):
    from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
    answer = None
    if generation['status'] == 'generated':
        try:
            answer = extract_d065_answer(generation['raw_output'], sparse=True)
        except ValueError:
            pass
    return dict(status='ok' if answer is not None else 'failed', answer=answer,
                raw_output=generation.get('raw_output', ''))


def verification_messages(base, candidate):
    result = deepcopy(base)
    content = result[0]['content']
    if not content[-2].get('text', '').startswith('Original question (verbatim):\n'):
        raise ValueError('original question position changed')
    content.insert(len(content)-2, dict(type='text', text=VERIFY_INSTRUCTION +
                   json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))))
    return result


def decide(candidate, verification, frame_ids):
    checked = candidate_from_generation(verification)
    raw = verification.get('raw_output', '')
    notes = '\n'.join(raw.strip().splitlines()[:-1])
    cited = sorted({int(s) for s in re.findall(
        r'\b(?:F(?:rame)?\s*(?:ID\s*)?|source\s+frame\s*(?:ID\s*)?)(\d+)\b',
        notes, flags=re.I)}.intersection(frame_ids))
    if checked['status'] != 'ok':
        use_draft, reason = True, 'verification_failed'
    elif candidate['status'] != 'ok':
        use_draft, reason = False, 'draft_failed_use_verification'
    elif checked['answer'] == candidate['answer']:
        use_draft, reason = False, 'verification_agrees'
    elif cited:
        use_draft, reason = False, 'correction_cites_provided_frame'
    else:
        use_draft, reason = True, 'correction_lacks_provided_frame_citation'
    selected = candidate if use_draft else checked
    return dict(status=selected['status'], prediction=selected['answer'],
                selected_stage='draft' if use_draft else 'verification', reason=reason,
                cited_actual_frame_ids=cited, candidate_answer=candidate['answer'],
                verification_answer=checked['answer'], citation_check_is_not_visual_truth=True)
