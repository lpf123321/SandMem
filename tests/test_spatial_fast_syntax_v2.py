import json
from types import SimpleNamespace

import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON as OriginalBoundedJSON
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext, BudgetJSON
from sandbox_vlm.backbones.spatial_fast_syntax_v2 import BoundedJSON, FastBudgetJSON, PlainHeadIndex


def make(fast=False):
    config = CharacterLevelParserConfig(max_json_array_length=512,
        alphabet=''.join(chr(i) for i in range(32, 127)) + '\u00a0\u4e2d\u6587')
    raw = JsonSchemaParser(discovery_schema(477), config=config)
    context = BudgetContext(SimpleNamespace(encode=lambda t, **_: list(t)), 10000, 477)
    return FastBudgetJSON(BoundedJSON(raw), context) if fast else BudgetJSON(OriginalBoundedJSON(raw), context)


@pytest.mark.parametrize('text', [
    '{"frame_id":477,"objects":[]}',
    '{"objects":[{"description":"white chair","bbox_2d":[0,0,1000,1000],"name":"chair"}],"frame_id":477}',
    '{"frame_id":477,"objects":[{"name":"   chair","description":"a \\"quoted\\" chair","bbox_2d":[1,2,3,4]}]}',
    '{"frame_id":477,"objects":[{"name":"one two three four five six seven eight","description":"chair","bbox_2d":[1,2,3,4]}]}',
    '{"frame_id":477,"objects":[{"name":"one two three four five six seven eight ninth","description":"chair","bbox_2d":[1,2,3,4]}]}',
    '{"frame_id":477,"objects":[{"name":"chair","description":"中文\\u0020chair","bbox_2d":[1,2,1.5,3.5]}]}',
    '{"frame_id":477,"objects":[{"name":"chair","description":"white","bbox_2d":[10,20,9,30]}]}',
    '{"frame_id":477,"objects":[{"name":"a","description":"","bbox_2d":[1,2,3,4]},'
        '{"name":"b","description":"","bbox_2d":[1,2,3,4]}]}',
])
def test_character_language_and_cached_stack_lookups_match_reference(text):
    old, new = make(), make(True)
    for i, c in enumerate(text):
        assert old.get_allowed_characters() == new.get_allowed_characters(), (i, text[:i])
        assert old.can_end() == new.can_end()
        if c not in old.get_allowed_characters():
            break  # Invalid examples must fail at the same character.
        old, new = old.add_character(c), new.add_character(c)
    assert old.get_allowed_characters() == new.get_allowed_characters()
    assert old.can_end() == new.can_end()


def test_raw_controls_are_not_literal_heads_but_escaped_controls_remain():
    def node():
        return SimpleNamespace(children={}, tokens=[])
    root = node()
    for token, text in enumerate(['abc\n', 'abc"', 'abc\\', 'x\t', 'line\\n']):
        cur = root
        for c in text:
            cur = cur.children.setdefault(c, node())
        cur.tokens.append(token)
    stack = [root]
    marked = set()
    while stack:
        n = stack.pop()
        marked.add(id(n))
        stack.extend(n.children.values())
    index = PlainHeadIndex(SimpleNamespace(tokenizer_tree=SimpleNamespace(root=root)),
        SimpleNamespace(special_nodes=marked))
    assert {(h, c) for h, c, _ in index.heads} == {('abc', '"'), ('abc', '\\'), ('line', '\\')}
