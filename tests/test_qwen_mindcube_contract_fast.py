import json
from types import SimpleNamespace
import pytest
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON
from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedStringJSON
from sandbox_vlm.backbones.qwen_mindcube_contract_fast import CachedClosedJSON,advance_plain_closed


def make(fast=False,config=None):
    raw=JsonSchemaParser(discovery_schema(2),config=config or CharacterLevelParserConfig(max_json_array_length=512))
    ctx=BudgetContext(SimpleNamespace(encode=lambda s,**_:list(s)),100000,2)
    return FastBudgetJSON((CachedClosedJSON if fast else ClosedStringJSON)(raw),ctx)


def consume(p,text):
    for c in text:
        assert c in p.get_allowed_characters()
        p=p.add_character(c)
    return p


@pytest.mark.parametrize('text',[
    '{"frame_id":2,"objects":[]}',
    '{"frame_id":2,"objects":[{"name":"one two three four five six seven eight nine ten","description":"one two three four five six seven eight ","bbox_2d":[0,0,1,1]}]}',
    '{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight  ',
    '{"frame_id":2,"objects":[{"name":"chair","description":"blue\\n\\n',
    '{"frame_id":2,"objects":[{"name":"chair","description":"blue chair","bbox_2d":[1,2,1.1,2.1]}]}',
])
def test_character_language_is_identical(text):
    a,b=make(),make(True)
    for c in text:
        assert a.get_allowed_characters()==b.get_allowed_characters()
        if c not in a.get_allowed_characters():break
        a,b=a.add_character(c),b.add_character(c)
    assert a.get_allowed_characters()==b.get_allowed_characters()
    assert a.can_end()==b.can_end()


@pytest.mark.parametrize('role,prefix',[
    ('name','{"frame_id":2,"objects":[{"name":"'),
    ('description','{"frame_id":2,"objects":[{"name":"chair","description":"'),
    ('description','{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight'),
])
@pytest.mark.parametrize('head',['blue chair',' a','  ', 'x y z ', '中文 chair', 'dark-blue,', 'one two three four five six seven eight nine'])
def test_bulk_plain_transition_equals_character_transition(role,prefix,head):
    original=consume(make(True),prefix);slow=original
    try:slow=consume(slow,head)
    except (AssertionError,ValueError):slow=None
    fast=advance_plain_closed(original,head)
    assert (slow is None)==(fast is None)
    if slow is not None:
        for key in ['text','role','words','in_word','escape']:
            assert getattr(slow.base,key)==getattr(fast.base,key)
        assert slow.base.string_top().parsed_string==fast.base.string_top().parsed_string
        assert slow.get_allowed_characters()==fast.get_allowed_characters()


def test_actual_token_sets_match_reference_at_sensitive_states():
    import torch
    from transformers import AutoTokenizer
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar
    from sandbox_vlm.backbones.spatial_fast_syntax_v2 import PlainHeadIndex
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedIndex,ClosedEnforcer
    from sandbox_vlm.backbones.qwen_mindcube_contract_fast import FastClosedEnforcer
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',local_files_only=True)
    syntax=SyntaxTokens(tok);grammar=BudgetGrammar(syntax,512,tok)
    index=ClosedIndex(syntax.data,grammar.index);heads=PlainHeadIndex(syntax.data,index)
    prefixes=[
        '{"frame_id":2,"objects":[{"name":"',
        '{"frame_id":2,"objects":[{"name":"blue ',
        '{"frame_id":2,"objects":[{"name":"one two three four five six seven eight nine ',
        '{"frame_id":2,"objects":[{"name":"chair","description":"',
        '{"frame_id":2,"objects":[{"name":"chair","description":"blue',
        '{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight',
        '{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight ',
    ]
    for prefix in prefixes:
        old,new=consume(make(config=grammar.parser_config(512)),prefix),consume(make(True,config=grammar.parser_config(512)),prefix)
        a=ClosedEnforcer(syntax.data,old,index,old.context)
        b=FastClosedEnforcer(syntax.data,new,index,new.context,heads)
        assert set(a.get_allowed_tokens([1,2]).allowed_tokens)==set(b.get_allowed_tokens([1,2]).allowed_tokens),prefix
