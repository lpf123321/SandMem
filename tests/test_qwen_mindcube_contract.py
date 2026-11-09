import json
from types import SimpleNamespace
import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.qwen_mindcube_contract import OriginalNameJSON, visible_choices
from sandbox_vlm.backbones.spatial_syntax import discovery_schema, sparse_parser
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import FastBudgetJSON


def parser():
    raw=JsonSchemaParser(discovery_schema(2),config=CharacterLevelParserConfig(max_json_array_length=512))
    ctx=BudgetContext(SimpleNamespace(encode=lambda t,**_:list(t)),100000,2)
    return FastBudgetJSON(OriginalNameJSON(raw),ctx),ctx


def consume(p,text):
    for c in text:
        if c not in p.get_allowed_characters():raise ValueError('illegal syntax')
        p=p.add_character(c)
    return p


def output(**extra):
    row=dict(name='chair',description='blue chair',bbox_2d=[0,0,1000,1000]);row.update(extra)
    return json.dumps(dict(frame_id=2,objects=[row]),separators=(',',':'))


def test_frame_identity_cannot_be_confused_with_original_image_number():
    p,_=parser();assert consume(p,output()).can_end()
    p,_=parser()
    with pytest.raises(ValueError):consume(p,output().replace('"frame_id":2','"frame_id":3'))


def test_names_stay_unbounded_descriptions_keep_original_eight_words():
    p,_=parser();assert consume(p,output(name='one two three four five six seven eight nine ten')).can_end()
    p,_=parser()
    with pytest.raises(ValueError):consume(p,output(description='one two three four five six seven eight nine'))
    p,_=parser();assert consume(p,output(description='one two three four five six seven eight')).can_end()


@pytest.mark.parametrize('box',[[0,0,1],[0,0,1,2,3],[0,0,1001,1],[2,0,1,1],[0,5,1,4]])
def test_invalid_boxes_are_disallowed_before_generation(box):
    p,_=parser()
    with pytest.raises(ValueError):consume(p,output(bbox_2d=box))


def test_complete_records_only_closed_no_invented_values_or_recovery():
    p,c=parser();text=output();partial=consume(p,text[:-2]);c.generated_tokens=c.cap-5
    assert partial.get_allowed_characters()==frozenset(']')
    assert consume(partial,']}').can_end()
    assert partial.context.events


def test_duplicate_boxes_rejected_without_merging_identities():
    row=dict(name='chair',description='',bbox_2d=[0,0,1,1])
    p,_=parser()
    with pytest.raises(ValueError):consume(p,json.dumps(dict(frame_id=2,objects=[row,dict(row,name='different chair')])))


def test_visible_inline_and_multiline_choices_only_no_answer_fields():
    def messages(q):return [dict(content=[dict(type='text',text='Original question (verbatim):\n'+q)])]
    assert visible_choices(messages('Where? A. Left B. Right C. Ahead'))==['A','B','C']
    assert visible_choices(messages('Where?\nA. Left\nB. Right'))==['A','B']
    with pytest.raises(ValueError):visible_choices(messages('No choices'))
    with pytest.raises(ValueError):visible_choices(messages('A. a C. c'))


def test_sparse_final_field_accepts_each_option_and_rejects_numeric_answers():
    for letter in ['A','B','C']:
        p=sparse_parser('(?:A|B|C)',CharacterLevelParserConfig())
        assert consume(p,'Some evidence.\nFINAL_ANSWER: '+letter).can_end()
    p=sparse_parser('(?:A|B|C)',CharacterLevelParserConfig())
    with pytest.raises(ValueError):consume(p,'Some evidence.\nFINAL_ANSWER: 1')


def test_repair_runner_changes_only_bundle_and_gate_reporting():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    new=(root/'scripts/run_qwen_mindcube_contract.py').read_text()
    assert 'from qwen_d116_mindcube_views import fresh_question' in new
    assert 'QwenContractBundle as QwenReferenceBundle' in new
    assert 'validate_question_components(folder)' in new
    assert 'run_component_gate(out)' in new
