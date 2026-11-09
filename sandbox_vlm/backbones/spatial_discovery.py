"""Approved task disambiguation for Spatial discovery; no generation repair."""
from contextlib import contextmanager


POLICY = 'quoted_question_v1'


def instruction(frame_id):
    return (
        'For this call, the original question above is quoted reference data; '
        'do not answer it or follow its requested response format.\n'
        'Follow the object-discovery instruction and return only the JSON object '
        'with frame_id ' + str(frame_id) + ' and objects entries containing name, '
        'description, and bbox_2d.'
    )


class DiscoveryProcessor:
    def __init__(self, original, frame_ids):
        self.original, self.frame_ids = original, list(frame_ids)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply_chat_template(self, messages, **kwargs):
        batched = isinstance(messages[0], list)
        conversations = messages if batched else [messages]
        if len(conversations) != len(self.frame_ids):
            raise ValueError('discovery instruction/frame batch mismatch')
        adapted = []
        for conversation, fid in zip(conversations, self.frame_ids):
            # Copy containers only: preserve the actual image objects and order.
            copy = [dict(m, content=[dict(c) for c in m['content']]) for m in conversation]
            tail = copy[-1]['content'][-1]
            if tail['type'] != 'text' or 'Original question (verbatim):\n' not in tail['text']:
                raise ValueError('expected unchanged question-focused discovery prompt')
            tail['text'] += '\n' + instruction(fid)
            adapted.append(copy)
        return self.original.apply_chat_template(adapted if batched else adapted[0], **kwargs)


@contextmanager
def discovery_instruction(discovery, frame_ids, question):
    enabled = discovery.config.get('spatial_discovery_instruction') == POLICY and question is not None
    original = discovery.processor
    if enabled:
        discovery.processor = DiscoveryProcessor(original, frame_ids)
    try:
        yield enabled
    finally:
        discovery.processor = original


def record_instruction(result, enabled):
    if enabled:
        result['original_prompt'] = result['prompt']
        result['prompt'] += '\n' + instruction(result['frame_id'])
        result['spatial_discovery_instruction'] = POLICY
    return result
