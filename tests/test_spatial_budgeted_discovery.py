import json
from types import SimpleNamespace
import pytest
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetJSON,BudgetContext,number_prefix_feasible,unique_float_completion


def parser(cap=100000):
    ctx=BudgetContext(SimpleNamespace(encode=lambda text,**kwargs:list(text)),cap,477)
    return BudgetJSON(BoundedJSON(JsonSchemaParser(discovery_schema(477),
        config=CharacterLevelParserConfig(max_json_array_length=512))),ctx),ctx


def consume(parser,text):
    for c in text:
        if c not in parser.get_allowed_characters():raise ValueError('illegal character')
        parser=parser.add_character(c)
    return parser


def test_closure_occurs_only_after_actual_complete_record():
    value=dict(name='chair',description='dark chair',bbox_2d=[10,20,80,90])
    text=json.dumps(dict(frame_id=477,objects=[value]))
    root,ctx=parser();partial=consume(root,text[:-2])
    assert ',' in partial.get_allowed_characters()
    ctx.generated_tokens=ctx.cap-5
    assert partial.get_allowed_characters()==frozenset(']')
    closed=consume(partial,']}')
    assert closed.can_end() and not closed.get_allowed_characters()
    assert json.loads(closed.text)['objects']==[value]


def test_original_coordinate_bounds_and_order_are_enforced():
    for box in ([0,0,1000,1000],[10.5,20.5,10.6,21]):
        root,_=parser()
        assert consume(root,json.dumps(dict(frame_id=477,objects=[dict(name='chair',description='chair',bbox_2d=box)]))).can_end()
    for box in ([10,20,80,3399999],[1000,20,1000,90],[900,20,450,90],[10,90,80,20]):
        root,_=parser()
        with pytest.raises(ValueError):consume(root,json.dumps(dict(frame_id=477,objects=[dict(name='chair',description='chair',bbox_2d=box)])))


def test_prefix_viability_uses_original_bounds_without_coordinate_rounding():
    assert number_prefix_feasible('99',900)
    assert not number_prefix_feasible('4',900)
    assert not number_prefix_feasible('1000',-1,inclusive=False)
    assert number_prefix_feasible('1000',999)
    assert number_prefix_feasible('9.99',9.995)
    assert not number_prefix_feasible('0.0',0.5)


def test_numeric_uniqueness_does_not_allow_an_infinite_decimal_dead_prefix():
    existing={332.0111111111111}
    assert not unique_float_completion('332.011111111111111111',existing,0)
    assert unique_float_completion('332.01111111111',existing,0)
    assert unique_float_completion('332.0111111111112',existing,0)
    assert not unique_float_completion('1000',{1000},0)


def test_final_brackets_are_generated_before_cap_without_filling_values():
    root,ctx=parser()
    value=dict(name='chair',description='dark chair',bbox_2d=[10,20,80,90.001])
    text=json.dumps(dict(frame_id=477,objects=[value]),separators=(',',':'))
    partial=consume(root,text[:-4])
    ctx.generated_tokens=ctx.cap-6
    assert partial.get_allowed_characters()==frozenset(']')
    result=consume(partial,']}]}')
    assert result.can_end() and json.loads(result.text)==json.loads(text)
