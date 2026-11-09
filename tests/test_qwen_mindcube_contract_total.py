import json
from types import SimpleNamespace
import pytest
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext
from sandbox_vlm.backbones.qwen_mindcube_contract_fast import CachedClosedJSON
from sandbox_vlm.backbones.qwen_mindcube_contract_total import SafeBudgetJSON,feasible_number


def parser():
    raw=JsonSchemaParser(discovery_schema(0),config=CharacterLevelParserConfig(max_json_array_length=512))
    return SafeBudgetJSON(CachedClosedJSON(raw),BudgetContext(SimpleNamespace(encode=lambda s,**_:list(s)),100000,0))


def consume(p,text):
    for c in text:
        if c not in p.get_allowed_characters():raise ValueError('illegal numeric/syntax prefix')
        p=p.add_character(c)
    return p


def test_integer_extension_must_satisfy_both_bounds_and_float_uniqueness():
    assert not feasible_number('1',788,True,(1000,))
    assert not feasible_number('10',788,True,(1000,))
    assert not feasible_number('100',788,True,(1000,))
    assert feasible_number('9',788,True,(1000,))
    assert feasible_number('100',788,True,())
    assert feasible_number('789',788,True,(789,)) # decimals remain unrestricted
    assert not feasible_number('1000',788,True,(1000,))


def test_whitespace_cannot_strand_incomplete_coordinate():
    prefix='{"frame_id":0,"objects":[{"name":"a","description":"","bbox_2d":[0,788,195,'
    p=consume(parser(),prefix+'100')
    assert ' ' not in p.get_allowed_characters()
    assert '0' in p.get_allowed_characters()
    assert consume(p,'0]}]}').can_end()


def test_duplicate_box_failure_prefix_is_pruned_before_no_successor():
    first=dict(name='a',description='',bbox_2d=[0,788,195,1000])
    prefix=json.dumps(dict(frame_id=0,objects=[first]),separators=(',',':'))[:-2]+',{"name":"b","description":"","bbox_2d":[0,788,195,'
    p=consume(parser(),prefix)
    assert '1' not in p.get_allowed_characters()
    assert consume(p,'999]}]}').can_end()


def test_no_rounding_or_precision_limit_for_legal_distinct_decimals():
    assert feasible_number('332.0111111111112',0,True,(332.0111111111111,))
    assert not feasible_number('332.011111111111111111',332.0111111111111,True,())
    p=parser()
    value=dict(frame_id=0,objects=[dict(name='a',description='',bbox_2d=[0,0,1.00000000000001,2.00000000000001])])
    assert consume(p,json.dumps(value,separators=(',',':'))).can_end()
    assert consume(parser(),json.dumps(value)).can_end()


@pytest.mark.parametrize('lower',[-1,0,1,788,999])
def test_every_valid_integer_witness_has_all_numeric_prefixes_allowed(lower):
    for value in range(max(0,int(lower)+1),1001):
        for n in range(1,len(str(value))+1):
            assert feasible_number(str(value)[:n],lower,True,())
