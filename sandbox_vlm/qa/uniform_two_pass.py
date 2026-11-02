"""Same two-pass policy for every input; numeric parsing is adapter formatting."""
from copy import deepcopy
import math
import re
from sandbox_vlm.qa.two_pass import verification_messages, VERIFY_INSTRUCTION
from sandbox_vlm.qa.bounded_notes import POLICY

FOOTER = 'Give brief evidence notes first, then the final line FINAL_ANSWER: <answer>.'


def promote_verification(base, snapshot, header):
    """Keep all evidence intact and use the identical D116 verification policy."""
    result=deepcopy(base);snap=deepcopy(snapshot);content=result[0]['content']
    content[0]['text']=header
    if content[-1].get('text')!=FOOTER:
        if not content[-1].get('text','').startswith('Original question (verbatim):\n'):
            raise ValueError('original question missing')
        content.append(dict(type='text',text=FOOTER))
    snap['sparse']=True;snap['response_style']='D065_strict'
    snap['limits']['max_output_tokens']=256
    snap['generation_kwargs']['max_new_tokens']=256
    snap['note_boundary']=dict(POLICY)
    return result,snap


def candidate_from_generation(generation,*,sparse=True,numeric=False):
    from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
    answer=None
    if generation['status']=='generated':
        try:
            answer=extract_d065_answer(generation['raw_output'],sparse=sparse)
            if numeric:
                answer=float(answer)
                if not math.isfinite(answer):raise ValueError('non-finite numeric answer')
        except ValueError:
            answer=None
    return dict(status='ok'if answer is not None else'failed',answer=answer,
                raw_output=generation.get('raw_output',''))


def decide(candidate,verification,frame_ids,*,numeric=False):
    checked=candidate_from_generation(verification,sparse=True,numeric=numeric)
    notes='\n'.join(verification.get('raw_output','').strip().splitlines()[:-1])
    cited=sorted({int(s)for s in re.findall(
        r'\b(?:F(?:rame)?\s*(?:ID\s*)?|source\s+frame\s*(?:ID\s*)?)(\d+)\b',notes,flags=re.I)}.intersection(frame_ids))
    if checked['status']!='ok':use_draft,reason=True,'verification_failed'
    elif candidate['status']!='ok':use_draft,reason=False,'draft_failed_use_verification'
    elif checked['answer']==candidate['answer']:use_draft,reason=False,'verification_agrees'
    elif cited:use_draft,reason=False,'correction_cites_provided_frame'
    else:use_draft,reason=True,'correction_lacks_provided_frame_citation'
    selected=candidate if use_draft else checked
    return dict(status=selected['status'],prediction=selected['answer'],
        selected_stage='draft'if use_draft else'verification',reason=reason,cited_actual_frame_ids=cited,
        candidate_answer=candidate['answer'],verification_answer=checked['answer'],citation_check_is_not_visual_truth=True)
