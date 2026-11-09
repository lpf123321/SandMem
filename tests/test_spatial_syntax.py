import json
import pytest
from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
from sandbox_vlm.backbones.spatial_syntax import discovery_schema, sparse_parser, visible_answer_pattern
from sandbox_vlm.qa.prompt import messages_for


def consume(parser, text):
    for character in text:
        if character not in parser.get_allowed_characters():
            raise ValueError('illegal character')
        parser = parser.add_character(character)
    return parser


def test_original_discovery_fields_and_frame_identity():
    schema = discovery_schema(477)
    valid = dict(frame_id=477, objects=[dict(name='chair', description='dark wooden chair', bbox_2d=[10, 20, 80, 90])])
    assert consume(JsonSchemaParser(schema), json.dumps(valid)).can_end()
    assert consume(JsonSchemaParser(schema), '{"frame_id":477,"objects":[]}').can_end()
    with pytest.raises(ValueError):
        consume(JsonSchemaParser(schema), '{"frame_id":729,')
    with pytest.raises(ValueError):
        consume(JsonSchemaParser(schema), 'D')


def test_description_length_is_not_a_new_syntax_restriction():
    schema = discovery_schema(16)
    value = dict(frame_id=16, objects=[dict(name='chair', description='one two three four five six seven eight nine', bbox_2d=[1,2,3,4])])
    assert consume(JsonSchemaParser(schema), json.dumps(value)).can_end()


def test_final_field_is_required_and_domain_is_visible_choices_only():
    config = CharacterLevelParserConfig()
    parser = lambda: sparse_parser('(?:A|B|C|D)', config)
    assert not consume(parser(), 'B').can_end()
    assert consume(parser(), 'F477 observed.\nFINAL_ANSWER: B').can_end()
    assert consume(parser(), 'FINAL_ANSWER:C').can_end()
    for invalid in ['B\nFINAL_ANSWER: 1', 'FINAL_ANSWER: A\nextra']:
        with pytest.raises(ValueError):
            consume(parser(), invalid)


def test_choice_domain_does_not_select_correct_answer():
    messages = messages_for('Which?\nA. one\nB. two', [object()], [477], 'brief_evidence')
    pattern, choices = visible_answer_pattern(messages)
    assert choices == ['A', 'B'] and pattern == '(?:A|B)'
    numeric = messages_for('How many?', [object()], [477], 'brief_evidence')
    assert visible_answer_pattern(numeric) == (r'[^\r\n]+', [])
