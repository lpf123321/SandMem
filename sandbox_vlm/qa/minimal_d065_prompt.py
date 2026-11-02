"""D111: verbatim D065 policy plus exactly two necessary sandbox sentences."""
from sandbox_vlm.qa.complete_sandbox import complete_messages
from sandbox_vlm.qa.prompt import messages_for


USER_PREFIX = (
    'The following actual video frames are in chronological order. Views may revisit '
    'the same physical object. Inspect the relevant evidence across all provided frames. '
    'Write brief evidence notes in chronological order, at most 100 words; distinguish '
    'observed facts from uncertainty. Then give the requested short answer on a separate '
    'final line exactly as FINAL_ANSWER: <answer>.')

SUPPLEMENT = (
    'The extra image is an estimated 3D sandbox with arbitrary scale; its instance records '
    'refer to actual observations, not verified physical identities. '
    'Coordinates are shared only within each local reference; the arrangement of different '
    'references does not imply physical distance, direction, adjacency, or separate rooms.')


def minimal_messages(snapshot, images, diagram):
    """Change only D107's first text block, retaining every other block exactly.

    Preserve the existing D065 density branch, rather than imposing long notes
    on dense inputs with a 64-token output cap. Do not append legacy memory
    warnings or another coordinate notice. Keep the original D065 final reminder.
    """
    original = messages_for(snapshot['question'], images, snapshot['frame_ids'],
        'brief_evidence' if snapshot['sparse'] else 'question_last')
    policy = original[0]['content'][0]['text']
    if snapshot['sparse'] and policy != USER_PREFIX:
        raise ValueError('D065 policy differs from the user-quoted text')
    messages, joint = complete_messages(snapshot, images, diagram, snapshot['payload'])
    messages[0]['content'][0]['text'] = policy + '\n' + SUPPLEMENT
    if snapshot['sparse']:
        messages[0]['content'].append(dict(original[0]['content'][-1]))
    return messages, joint
