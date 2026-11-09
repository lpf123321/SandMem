"""Explicit Spatial syntax-adapter configuration; historical defaults untouched."""


def create_spatial_backbone(config, official_root, *, device='cuda:0'):
    from copy import deepcopy
    from .spatial_transport import SpatialBackbone, SpatialCoverage
    settings = config.get('spatial_transport', {})
    expected = dict(discovery_syntax=(None,'original_discovery_json_v1','budget_closed_discovery_v3'),
                    sparse_qa_syntax=(None,'original_sparse_final_field_v1'))
    for key, values in expected.items():
        if settings.get(key) not in values:
            raise ValueError('unapproved Spatial syntax policy: ' + key)
    native_config=deepcopy(config)
    for key in expected:native_config.get('spatial_transport',{}).pop(key,None)
    bundle = SpatialBackbone(native_config, official_root, device=device)
    if not any(settings.get(key) for key in expected):
        return bundle
    from .spatial_syntax import SyntaxTokens, SpatialSyntaxDiscovery, SpatialSyntaxAnswerer
    from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
    bundle.syntax_tokens = SyntaxTokens(bundle.runtime.processor.tokenizer)
    if settings.get('discovery_syntax'):
        if settings['discovery_syntax']=='budget_closed_discovery_v3':
            from .spatial_budgeted_discovery import BudgetGrammar, SpatialBudgetedDiscovery
            class EphemeralDiscovery(SpatialBudgetedDiscovery):
                def discover_batch(self,*args,**kwargs):
                    try:return super().discover_batch(*args,**kwargs)
                    finally:self.grammar.contexts=[]
                def discover(self,*args,**kwargs):
                    try:return super().discover(*args,**kwargs)
                    finally:self.grammar.contexts=[]
            grammar=BudgetGrammar(bundle.syntax_tokens,config['perception']['mllm']['max_new_tokens'],
                bundle.runtime.processor.tokenizer)
            bundle.discovery=EphemeralDiscovery(bundle.discovery.config,bundle.runtime,grammar)
        else:
            bundle.discovery = SpatialSyntaxDiscovery(bundle.discovery.config, bundle.runtime, bundle.syntax_tokens)
    if settings.get('sparse_qa_syntax'):
        bundle.answerer = SpatialSyntaxAnswerer(bundle.answerer.config, bundle.runtime, bundle.syntax_tokens)
    bundle.relevance = FrozenRelevanceScorer(bundle.answerer)
    bundle.coverage = SpatialCoverage(bundle.answerer)
    return bundle
