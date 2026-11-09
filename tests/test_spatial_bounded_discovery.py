import json
import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON


def parser():
    return BoundedJSON(JsonSchemaParser(discovery_schema(477),config=CharacterLevelParserConfig(max_json_array_length=512)))


def consume(parser,text):
    for c in text:
        if c not in parser.get_allowed_characters():raise ValueError('illegal character')
        parser=parser.add_character(c)
    return parser


def value(name='chair',description='dark chair',box=None):
    return dict(name=name,description=description,bbox_2d=box or [10,20,80,90])


def test_original_valid_fields_and_empty_objects_are_allowed():
    for objects in ([],[value()], [value(),value('table','wooden table',[100,20,180,90])]):
        assert consume(parser(),json.dumps(dict(frame_id=477,objects=objects))).can_end()


@pytest.mark.parametrize('field',['name','description'])
def test_eight_words_are_allowed_ninth_is_rejected(field):
    obj=value();obj[field]='one two three four five six seven eight'
    assert consume(parser(),json.dumps(dict(frame_id=477,objects=[obj]))).can_end()
    obj[field]+=' nine'
    with pytest.raises(ValueError):consume(parser(),json.dumps(dict(frame_id=477,objects=[obj])))


def test_exact_numeric_box_duplicate_is_rejected_without_identity_inference():
    objects=[value(),value('different name','different description',[10.0,20,80,90])]
    with pytest.raises(ValueError):consume(parser(),json.dumps(dict(frame_id=477,objects=objects)))


def test_string_escapes_do_not_bypass_word_limit():
    text='{"frame_id":477,"objects":[{"name":"one two three four five six seven eight\\u0020nine","description":"chair","bbox_2d":[10,20,80,90]}]}'
    with pytest.raises(ValueError):consume(parser(),text)
