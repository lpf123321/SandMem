"""D065 answer policy with continuous-reference evidence and actual images."""
import json

from sandbox_vlm.qa.prompt import messages_for


def build_messages(question, frame_ids, images, diagram, payload, *, sparse):
    original = messages_for(question, images, frame_ids, 'brief_evidence' if sparse else 'question_last')
    content = [dict(original[0]['content'][0])]
    content[0]['text'] += (
        '\nThe extra image is an estimated 3D sandbox with arbitrary scale, not a video frame; '
        'instance IDs are appearance-based estimates and boxes bound observed surfaces at their recorded source frame. '
        'Scenes sharing one continuous reference retain shared coordinates; separate-reference panel layout has no physical meaning.')
    content.extend([dict(type='text', text='Estimated 3D sandbox and continuous camera trajectory:'),
                    dict(type='image', image=diagram),
                    dict(type='text', text='Relevant complete instance records and source links (JSON data):\n'
                         + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))),
                    dict(type='text', text='Actual video frames in chronological order:')])
    for fid, image in zip(frame_ids, images):
        content.extend([dict(type='text', text=f'Actual source frame ID {fid}:'), dict(type='image', image=image)])
    content.append(dict(type='text', text='Original question (verbatim):\n' + question))
    if sparse:
        content.append(dict(original[0]['content'][-1]))
    return [dict(role='user', content=content)], [diagram, *images]


def build_verification_messages(base, candidate):
    """Copy message containers but keep the exact actual image objects."""
    from sandbox_vlm.qa.two_pass import VERIFY_INSTRUCTION
    result = [dict(message, content=[dict(block) for block in message['content']]) for message in base]
    content = result[0]['content']
    if not content[-2].get('text', '').startswith('Original question (verbatim):\n'):
        raise ValueError('original question position changed')
    content.insert(len(content) - 2, dict(type='text', text=VERIFY_INSTRUCTION
        + json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))))
    return result
