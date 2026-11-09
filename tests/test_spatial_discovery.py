"""Prevent discovery-specific task instructions leaking to other model roles."""
from types import SimpleNamespace
import pytest
from sandbox_vlm.backbones.spatial_discovery import (
    POLICY, DiscoveryProcessor, discovery_instruction, instruction, record_instruction,
)
from sandbox_vlm.perception.discovery import discovery_prompt


class Recorder:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return messages


def test_independent_prompts_keep_question_and_actual_images():
    processor = Recorder()
    images = [object(), object()]
    ids = [477, 729]
    prompts = [discovery_prompt(fid, question='Original question?\nAnswer A–D.') for fid in ids]
    messages = [[dict(role='user', content=[dict(type='image', image=image),
        dict(type='text', text=prompt)])] for image, prompt in zip(images, prompts)]
    actual = DiscoveryProcessor(processor, ids).apply_chat_template(messages, tokenize=True)
    for i, fid in enumerate(ids):
        assert actual[i][0]['content'][0]['image'] is images[i]
        assert actual[i][0]['content'][1]['text'] == prompts[i] + '\n' + instruction(fid)
        assert messages[i][0]['content'][1]['text'] == prompts[i]


@pytest.mark.parametrize('question,enabled', [('q', True), (None, False)])
def test_scoped_processor_restored_after_failure(question, enabled):
    original = Recorder()
    discovery = SimpleNamespace(processor=original, config={'spatial_discovery_instruction': POLICY})
    with pytest.raises(RuntimeError):
        with discovery_instruction(discovery, [16], question) as active:
            assert active is enabled
            assert (discovery.processor is original) is (not enabled)
            raise RuntimeError('model failure')
    assert discovery.processor is original


def test_default_is_unchanged_and_actual_prompt_is_reported():
    original = Recorder()
    discovery = SimpleNamespace(processor=original, config={})
    with discovery_instruction(discovery, [16], 'q') as enabled:
        assert not enabled and discovery.processor is original
    result = dict(frame_id=16, prompt='original', raw_output='2', status='perception_failed')
    record_instruction(result, True)
    assert result['prompt'] == 'original\n' + instruction(16)
    assert result['original_prompt'] == 'original'
    assert result['raw_output'] == '2' and result['status'] == 'perception_failed'
