"""D072 message assembly; notes never update identities or compute an answer."""
import math


def group_messages(prompt, question, chunk, images):
    import json
    from sandbox_vlm.qa.coverage_evidence import MEMORY_NOTICE
    frames=chunk['frames']
    if not frames or frames!=sorted(set(frames)) or len(frames)!=len(images):
        raise ValueError('group images must preserve unique actual frame order')
    records=chunk['payload']['hypotheses']
    ids=[r[0]for r in records]
    if len(ids)!=len(set(ids)) or not set(chunk['owned_visual_ids'])<=set(ids):
        raise ValueError('lost or duplicate visual IDs')
    content=[dict(type='text',text=prompt+MEMORY_NOTICE+json.dumps(chunk['payload'],separators=(',',':')))]
    for frame,image in zip(frames,images):
        content.extend([dict(type='text',text=f'Actual source frame ID {frame}:'),dict(type='image',image=image)])
    content.append(dict(type='text',text='Original question (verbatim):\n'+question))
    return [dict(role='user',content=content)]


def final_messages(base, groups, notes):
    if len(groups)!=4 or len(notes)!=4 or any(not isinstance(n,str)or not n.strip()for n in notes):
        raise ValueError('all four nonempty evidence notes are required; no partial fallback')
    text=base
    for k,(group,note)in enumerate(zip(groups,notes)):
        text+=f"\nEvidence note group {k}, actual source frame IDs {group['frames']}:\n"+note
    return [dict(role='user',content=[dict(type='text',text=text)])]


def parse_final(raw):
    value=float(raw.strip())
    if not math.isfinite(value):raise ValueError('nonfinite answer')
    return value
