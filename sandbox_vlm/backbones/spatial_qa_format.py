"""Spatial-only sparse response-format disambiguation; strict parsing retained.

No option/type/answer inspection: the existing sparse evidence-notes instruction
is the sole gate. This never changes images, token/output caps or answer content.
"""
from sandbox_vlm.backbones.spatial_transport import SpatialAnswerer
from sandbox_vlm.qa.minimal_d065_prompt import USER_PREFIX

POLICY = 'sparse_final_field_v1'
INSTRUCTION = (
    'For this call, follow the evidence-notes instruction above. '
    'The last nonempty line must be FINAL_ANSWER: <answer>, using the answer format '
    'requested in the original question; the earlier request for a direct short '
    'answer applies only to <answer>, not to the whole response.'
)


def adapt_messages(messages):
    if not messages or not messages[0]['content']:
        raise ValueError('empty QA messages')
    first = messages[0]['content'][0]
    enabled = first.get('type') == 'text' and first.get('text', '').startswith(USER_PREFIX)
    if not enabled:
        return messages, False
    copied = [dict(m, content=[dict(c) for c in m['content']]) for m in messages]
    copied[-1]['content'].append(dict(type='text', text=INSTRUCTION))
    return copied, True


class SparseQAProcessor:
    def __init__(self, processor):
        self.original = processor

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply_chat_template(self, messages, **kwargs):
        actual, _ = adapt_messages(messages)
        return self.original.apply_chat_template(actual, **kwargs)


class SpatialFormatAnswerer(SpatialAnswerer):
    def __init__(self, config, runtime):
        super().__init__(config, runtime)
        self.processor = SparseQAProcessor(self.processor)

    def answer(self, messages, images, limits, **kwargs):
        _, enabled = adapt_messages(messages)
        result = super().answer(messages, images, limits, **kwargs)
        result['spatial_response_format'] = dict(policy=POLICY if enabled else None,
            instruction=INSTRUCTION if enabled else None, original_messages_preserved=True,
            parser_changed=False, answer_content_constrained=False)
        return result
