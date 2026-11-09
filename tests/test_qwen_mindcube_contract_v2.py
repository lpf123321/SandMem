import json
from types import SimpleNamespace
import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.qwen_mindcube_contract import visible_choices
from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedStringJSON as OriginalNameJSON
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


def test_previous_runner_preserved():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    new=(root/'scripts/run_qwen_mindcube_contract.py').read_text()
    assert 'from qwen_d116_mindcube_views import fresh_question' in new
    assert 'QwenContractBundle as QwenReferenceBundle' in new
    assert 'validate_question_components(folder)' in new
    assert 'run_component_gate(out)' in new


@pytest.mark.parametrize('separator',[' ',r'\n',r'\u0020',r'\t'])
def test_eighth_word_separator_requires_close_quote_not_padding(separator):
    p,_=parser()
    head='{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight'+separator
    partial=consume(p,head)
    assert partial.get_allowed_characters()==frozenset('"')
    assert consume(partial,'","bbox_2d":[0,0,1,1]}]}').can_end()


@pytest.mark.parametrize('padding',['  ',r'\n\n',r'\u0020\u0020',r'\t\t'])
def test_repeated_whitespace_is_rejected_before_cap(padding):
    p,_=parser()
    with pytest.raises(ValueError):consume(p,'{"frame_id":2,"objects":[{"name":"chair","description":"blue'+padding)


def test_long_names_remain_valid_but_blank_names_cannot_consume_budget():
    p,_=parser();assert consume(p,output(name='one two three four five six seven eight nine ten eleven twelve')).can_end()
    p,_=parser()
    with pytest.raises(ValueError):consume(p,'{"frame_id":2,"objects":[{"name":"  ')


def test_structural_padding_cannot_loop():
    p,_=parser()
    with pytest.raises(ValueError):consume(p,'{  ')


def test_actual_qwen_token_mask_cannot_emit_space_loop_after_eighth_word():
    import torch
    from types import MethodType
    from transformers import AutoTokenizer
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedIndex,closed_prefix
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',local_files_only=True)
    syntax=SyntaxTokens(tok);grammar=BudgetGrammar(syntax,512,tok)
    grammar.closed_index=ClosedIndex(syntax.data,grammar.index)
    grammar.prefix=MethodType(closed_prefix,grammar)
    fn=grammar.prefix(JsonSchemaParser(discovery_schema(2),config=grammar.parser_config(512)))
    prefix='{"frame_id":2,"objects":[{"name":"chair","description":"one two three four five six seven eight '
    ids=[1,2]
    for token in tok.encode(prefix,add_special_tokens=False):
        assert token in fn(0,torch.tensor(ids));ids.append(token)
    allowed=fn(0,torch.tensor(ids))
    for text in [' ','  ',' '*100,r'\n',r'\u0020']:
        assert tok.encode(text,add_special_tokens=False)[0] not in allowed
    for token in tok.encode('","bbox_2d":[0,0,1,1]}]}',add_special_tokens=False):
        assert token in fn(0,torch.tensor(ids));ids.append(token)
    assert tok.eos_token_id in fn(0,torch.tensor(ids))


def test_cold_v2_changes_only_view_wording_and_fail_fast():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/qwen_d116_mindcube_views.py').read_text()
    source=source.replace('from sandbox_vlm.data.numbered_views import full_view_ids,bind_messages','from sandbox_vlm.data.numbered_views_v2 import full_view_ids,bind_messages')
    source=source.replace("            counters['discovery_failures']+=1\n            continue", "            counters['discovery_failures']+=1\n            write(folder/'internal_failure.json',dict(stage='discovery',frame_id=fid,costs=counters,error=result.get('error'),raw_failure_preserved=True))\n            raise RuntimeError('internal discovery contract failure: '+str(result.get('error')))")
    source=source.replace("        if result['status'] not in ('generated','answer_failed') or not result.get('model_calls'):","        if result['status'] != 'generated' or not result.get('model_calls'):")
    assert ast.dump(ast.parse(source))==ast.dump(ast.parse((root/'scripts/qwen_d116_mindcube_views_v2.py').read_text()))


def test_verifier_uses_benchmark_view_order_without_changing_original_question():
    from sandbox_vlm.data.numbered_views_v2 import bind_messages
    from sandbox_vlm.qa.prompt import messages_for
    q='Which object? A. one B. two'
    messages=messages_for(q,[object(),object()],[0,1],'brief_evidence')
    messages[0]['content'].insert(-2,dict(type='text',text='Check the independently generated candidate below then check chronological order.'))
    actual,_=bind_messages(messages,[0,1],has_sandbox=False)
    assert any(b.get('text')=='Original question (verbatim):\n'+q for b in actual[0]['content'])
    assert any('then check the supplied view order' in b.get('text','') for b in actual[0]['content'])
