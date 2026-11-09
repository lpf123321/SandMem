"""Check sparse-format compatibility without choosing/repairing answers."""
from sandbox_vlm.backbones.spatial_qa_format import adapt_messages, INSTRUCTION
from sandbox_vlm.qa.prompt import messages_for
from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
import pytest


def test_sparse_addition_preserves_all_evidence_and_question():
    image = object()
    messages = messages_for('Original question? Answer directly.', [image], [477], 'brief_evidence')
    actual, enabled = adapt_messages(messages)
    assert enabled and actual[0]['content'][:-1] == messages[0]['content']
    assert actual[0]['content'][2]['image'] is image
    assert actual[0]['content'][-1] == dict(type='text', text=INSTRUCTION)
    assert messages[0]['content'][-1]['text'] != INSTRUCTION


def test_dense_path_and_strict_parser_remain_unchanged():
    messages = messages_for('How many?', [object()], [16], 'question_last')
    actual, enabled = adapt_messages(messages)
    assert not enabled and actual is messages
    with pytest.raises(ValueError, match='missing exact final line'):
        extract_d065_answer('B', sparse=True)
    assert extract_d065_answer('Evidence.\nFINAL_ANSWER: B', sparse=True) == 'B'


def test_format_instruction_does_not_depend_on_question_or_candidate():
    for question in ('How many?', 'Which order? A. one B. two'):
        actual, enabled = adapt_messages(messages_for(question, [object()], [477], 'brief_evidence'))
        assert enabled and actual[0]['content'][-1]['text'] == INSTRUCTION
