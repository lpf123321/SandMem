"""Parser-level equivalence, including whitespace and near-cap closure."""
from types import SimpleNamespace

import pytest
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
from sandbox_vlm.backbones.spatial_syntax import discovery_schema
from sandbox_vlm.backbones.spatial_bounded_discovery import BoundedJSON
from sandbox_vlm.backbones.spatial_budgeted_discovery import BudgetContext, BudgetJSON
from sandbox_vlm.backbones.spatial_fast_syntax import FastBudgetJSON, advance_plain


def parser(cls=BudgetJSON, cap=512):
    ctx = BudgetContext(SimpleNamespace(encode=lambda t, **_: list(t)), cap, 477)
    raw = JsonSchemaParser(discovery_schema(477), config=CharacterLevelParserConfig(
        max_json_array_length=512, alphabet=''.join(chr(i) for i in range(32, 127)) + '\u00a0'))
    return cls(BoundedJSON(raw), ctx)


def consume(p, text):
    for c in text:
        if c not in p.get_allowed_characters():
            return None
        p = p.add_character(c)
    return p


@pytest.mark.parametrize('initial,head', [
    ('', '  dark chair'), ('seven red blue green white yellow black ', 'chair'),
    ('seven red blue green white yellow black chair ', 'ninth'),
    ('seven red blue green white yellow black chair ', '\u00a0'),
    ('chair', '   '), ('chair', '\u00a0lamp'), ('', '\u00a0'),
])
def test_bulk_plain_head_matches_character_transitions(initial, head):
    prefix = '{"frame_id":477,"objects":[{"name":"' + initial
    reference = consume(parser(), prefix)
    candidate = consume(parser(FastBudgetJSON), prefix)
    expected = consume(reference, head)
    actual = advance_plain(candidate, head)
    assert (expected is None) == (actual is None)
    if expected is not None:
        assert expected.text == actual.text
        assert expected.get_allowed_characters() == actual.get_allowed_characters()
        assert expected.base.words == actual.base.words
        assert expected.base.in_word == actual.base.in_word
        for suffix in ('",', ' extra', '\\n', '\\u0020'):
            a, b = consume(expected, suffix), consume(actual, suffix)
            assert (a is None) == (b is None)
            if a is not None:
                assert a.get_allowed_characters() == b.get_allowed_characters()


def test_cached_closure_tracks_mutable_token_budget_and_allows_original_failure():
    prefix = '{"frame_id":477,"objects":[{"name":"chair","description":"dark","bbox_2d":[10,20,80,90]}'
    reference = consume(parser(), prefix)
    candidate = consume(parser(FastBudgetJSON), prefix)
    for n in (0, 490, 507, 508, 509, 510, 511, 0):
        reference.context.generated_tokens = candidate.context.generated_tokens = n
        assert reference.get_allowed_characters() == candidate.get_allowed_characters()
        assert reference.closure() == candidate.closure()
    assert reference.context.events == candidate.context.events
