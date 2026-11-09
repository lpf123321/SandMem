import json
import random
from types import SimpleNamespace

import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext
from sandbox_vlm.backbones.qwen_mindcube_contract_fast import CachedClosedJSON
from sandbox_vlm.backbones.qwen_mindcube_reachable import ReachableJSON, SemanticJSON, completion_witness
from sandbox_vlm.perception.discovery import parse_discovery


def make(cap=512):
    raw = JsonSchemaParser(discovery_schema(1), config=CharacterLevelParserConfig(max_json_array_length=512))
    return ReachableJSON(CachedClosedJSON(raw), BudgetContext(SimpleNamespace(encode=lambda s, **_: list(s)), cap, 1))


def consume(p, text):
    for c in text:
        assert c in p.get_allowed_characters(), (p.text, c)
        p = p.add_character(c)
    return p


@pytest.mark.parametrize('prefix', [
    '', '{', '{"objects":[', '{"frame_id":1,"objects":[{"name":"',
    '{"frame_id":1,"objects":[{"name":"object","description":"blue,striped,striped,striped',
    '{"frame_id":1,"objects":[{"name":"object","description":"one two three four five six seven eight',
    '{"frame_id":1,"objects":[{"name":"object","description":"blue\\u',
    '{"frame_id":1,"objects":[{"name":"object","description":"","bbox_2d":[0,788,195,100',
])
def test_witness_is_valid_original_schema_not_a_record_repair(prefix):
    p = consume(make(), prefix)
    witness = p.witness()
    assert witness is not None and witness.isascii()
    # The witness is not attached to the recorded prefix.
    assert p.text == prefix
    assert consume(p, witness).can_end()
    parse_discovery(prefix+witness, 1, (640, 480))


@pytest.mark.parametrize('field', ['name', 'description'])
def test_no_whitespace_word_loop_can_exhaust_budget(field):
    head = '{"frame_id":1,"objects":[{"name":"'
    if field == 'description':
        head += 'toothbrush","description":"'
    p = consume(make(), head+'striped,striped')
    # One-character tokens are an adversarial tokenizer: prefer continuing the
    # same word forever. Closing is forced before other required fields strand.
    for step in range(512):
        p.context.generated_tokens = step
        if p.can_end():
            assert step < 512  # reserve EOS
            parse_discovery(p.text, 1, (640, 480))
            break
        allowed = p.get_allowed_characters()
        assert allowed
        c = 'x' if 'x' in allowed else next(c for c in p.witness() if c in allowed)
        p = p.add_character(c)
    else:
        pytest.fail('unclosed JSON at original cap')


def test_tight_string_only_closes_does_not_fill_the_missing_box():
    prefix = '{"frame_id":1,"objects":[{"name":"chair","description":"blue,striped'
    p = consume(make(), prefix)
    p.context.generated_tokens = 512-len(p.witness())-1
    assert p.get_allowed_characters() == frozenset('"')
    assert p.text == prefix


def test_unfinishable_unicode_escape_branch_blocked_before_dead_end():
    prefix = '{"frame_id":1,"objects":[{"name":"chair","description":"one two three four five six seven eight '
    p = consume(make(), prefix)
    assert p.get_allowed_characters() == frozenset('"')


def test_names_and_precision_have_no_new_fixed_limit():
    payload = dict(frame_id=1, objects=[dict(name=' '.join(['chair']*20), description='blue',
                                          bbox_2d=[0, 0, 1.00000000000001, 2.00000000000001])])
    assert consume(make(), json.dumps(payload, separators=(',', ':'))).can_end()


@pytest.mark.parametrize('seed', range(8))
def test_random_adversarial_singleton_token_paths_always_close(seed):
    rng = random.Random(seed)
    p = make(160)
    for step in range(160):
        p.context.generated_tokens = step
        if p.can_end():
            assert step < 160
            parse_discovery(p.text, 1, (10, 10))
            break
        allowed = p.get_allowed_characters()
        assert allowed
        # Include escapes, punctuation without spaces, long numbers and names.
        candidates = sorted(allowed.intersection(set('"{}[],:\\u0123456789.abc ')))
        p = p.add_character(rng.choice(candidates or sorted(allowed)))
    else:
        pytest.fail('budget invariant did not terminate')


def test_native_qwen_repetition_terminates_and_profile_cache_matches_full_trie():
    from transformers import AutoTokenizer
    from sandbox_vlm.backbones.spatial_syntax import SyntaxTokens
    from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetGrammar
    from sandbox_vlm.backbones.qwen_mindcube_contract_v2 import ClosedIndex
    from sandbox_vlm.backbones.spatial_fast_syntax_v2 import PlainHeadIndex
    from sandbox_vlm.backbones.qwen_mindcube_reachable import ReachableEnforcer
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-2B-Instruct',
        revision='89644892e4d85e24eaac8bacfd4f463576704203',local_files_only=True)
    syntax=SyntaxTokens(tok);grammar=BudgetGrammar(syntax,512,tok)
    index=ClosedIndex(syntax.data,grammar.index);heads=PlainHeadIndex(syntax.data,index)
    p=ReachableJSON(CachedClosedJSON(JsonSchemaParser(discovery_schema(1),config=grammar.parser_config(512))),BudgetContext(tok,512,1))
    prefix='{"frame_id":1,"objects":[{"name":"toothbrush","description":"colorful,striped'
    for c in prefix:p=p.add_character(c)
    e=ReachableEnforcer(syntax.data,p,index,p.context,heads)
    priority=tok.encode(',striped',add_special_tokens=False);seq=[1,2]
    eos=syntax.data.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    comparisons=0
    for step in range(512):
        p.context.generated_tokens=step
        allowed=e.get_allowed_tokens(seq).allowed_tokens
        current=e.prefix_states[tuple(seq)]
        if step in (2,32,440,485,500):
            reference=ReachableEnforcer(syntax.data,current.parser,index,p.context,heads)
            state=reference.OutputTensorState(parser=current.parser)
            reference._compute_allowed_tokens(tuple(seq),state)
            assert set(allowed)==set(state.allowed_tokens.allowed_tokens)
            comparisons+=1
        if eos.intersection(allowed):
            assert current.parser.can_end() and step<512
            parse_discovery(current.parser.text,1,(640,480))
            assert comparisons==5
            break
        candidate=priority[step%len(priority)]
        if candidate not in allowed:
            single=tok.encode(current.parser.witness()[0],add_special_tokens=False)
            assert len(single)==1 and single[0] in allowed
            candidate=single[0]
        seq.append(candidate)
    else:
        pytest.fail('actual Qwen token traversal exceeded original cap')
