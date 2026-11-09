from types import SimpleNamespace
from copy import deepcopy
import pytest

from sandbox_vlm.backbones.spatial_bundle import create_spatial_backbone
from sandbox_vlm.backbones.spatial_model import MODEL_ID, REVISION


@pytest.mark.parametrize('policy',['original_discovery_json_v1','budget_closed_discovery_v3'])
def test_explicit_syntax_factory_shares_weights_and_all_roles(monkeypatch,policy):
    from sandbox_vlm.backbones import spatial_transport, spatial_syntax
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [1 if text=='Yes' else 2])
    processor = SimpleNamespace(tokenizer=tokenizer, image_processor=SimpleNamespace(merge_size=2))
    runtime = SimpleNamespace(transport_processor=processor, transport_model=object(), device='cpu',
        processor=processor)
    mllm = dict(model_id=MODEL_ID, revision=REVISION, dtype='bfloat16', max_new_tokens=512, temperature=0)
    config = dict(perception=dict(mllm=dict(mllm)), qa=dict(mllm=dict(mllm)), spatial_transport=dict(
        discovery_syntax=policy, sparse_qa_syntax='original_sparse_final_field_v1'))
    original = deepcopy(config)
    old = SimpleNamespace(runtime=runtime, discovery=SimpleNamespace(config=config['perception']['mllm']),
        answerer=SimpleNamespace(config=config['qa']['mllm']))
    monkeypatch.setattr(spatial_transport, 'SpatialBackbone', lambda *args, **kw: old)
    grammar = object()
    monkeypatch.setattr(spatial_syntax, 'SyntaxTokens', lambda tokenizer: grammar)
    from sandbox_vlm.backbones import spatial_budgeted_discovery
    monkeypatch.setattr(spatial_budgeted_discovery,'BudgetGrammar',lambda original,cap,tok:SimpleNamespace(original=original,cap=cap,contexts=[]))
    bundle = create_spatial_backbone(config, '/unused')
    assert bundle.answerer.model is bundle.discovery.model is bundle.relevance.model is bundle.coverage.model
    assert bundle.answerer.model is runtime.transport_model
    assert bundle.answerer.grammar is grammar
    if policy=='budget_closed_discovery_v3':
        assert bundle.discovery.grammar.original is grammar and bundle.discovery.grammar.cap==512
        bundle.discovery.grammar.contexts=['transient generated text']
        def fail(*args,**kwargs):raise RuntimeError('forward failure')
        monkeypatch.setattr(spatial_budgeted_discovery.SpatialBudgetedDiscovery,'discover_batch',fail)
        with pytest.raises(RuntimeError,match='forward failure'):
            bundle.discovery.discover_batch([],[])
        assert bundle.discovery.grammar.contexts==[]
    else:
        assert bundle.discovery.grammar is grammar
    assert config == original


def test_no_syntax_flags_returns_original_transport(monkeypatch):
    from sandbox_vlm.backbones import spatial_transport
    bundle = object()
    monkeypatch.setattr(spatial_transport, 'SpatialBackbone', lambda *args, **kw: bundle)
    assert create_spatial_backbone({}, '/unused') is bundle


def test_unknown_decoding_policy_rejected_before_model_load():
    with pytest.raises(ValueError, match='unapproved Spatial syntax policy'):
        create_spatial_backbone(dict(spatial_transport=dict(discovery_syntax='unknown')), '/unused')


def test_legacy_constructor_cannot_silently_ignore_explicit_syntax_config():
    from sandbox_vlm.backbones.spatial_transport import SpatialBackbone
    with pytest.raises(ValueError,match='do not silently bypass'):
        SpatialBackbone(dict(spatial_transport=dict(sparse_qa_syntax='original_sparse_final_field_v1')),'/unused')
