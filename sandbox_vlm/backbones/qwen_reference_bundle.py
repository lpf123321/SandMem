"""Shared original frozen Qwen3-VL-2B for all D116 MLLM roles; no Spatial syntax."""
from contextlib import contextmanager
from types import SimpleNamespace
from sandbox_vlm.qa.model import QwenAnswerer


class AuditedProcessor:
    def __init__(self, raw):
        self.raw = raw
        self.audit_sink = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def image_processor(self):
        return self.raw.image_processor

    @image_processor.setter
    def image_processor(self, value):
        self.raw.image_processor = value

    def apply_chat_template(self, *args, **kwargs):
        batch = self.raw.apply_chat_template(*args, **kwargs)
        if kwargs.get('tokenize', True) and self.audit_sink is not None:
            self.audit_sink([self.raw.tokenizer.decode(row, skip_special_tokens=False) for row in batch['input_ids']], batch)
        return batch


class AuditedQwenAnswerer(QwenAnswerer):
    def answer(self, messages, images, limits, **kwargs):
        trace=[]
        def capture(module,args,kw):
            grid=kw.get('grid_thw')
            trace.append(dict(component='actual_qwen_visual_encoder',image_grid_thw=grid.detach().cpu().tolist()))
        hook=self.model.model.visual.register_forward_pre_hook(capture,with_kwargs=True)
        try:
            result=super().answer(messages,images,limits,**kwargs)
        finally:
            hook.remove()
        result['actual_forward_trace']=trace
        if result['model_calls']:
            assert len(trace)==1 and trace[0]['image_grid_thw']==result['image_grid_thw'], 'actual Qwen image encoder mismatch'
        return result


class QwenReferenceBundle:
    def __init__(self, config, *, device='cuda:0'):
        from sandbox_vlm.perception.discovery import QwenDiscovery
        from sandbox_vlm.qa.model import QwenAnswerer
        from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
        from sandbox_vlm.perception.coverage import SharedVisionCoverage
        self.discovery = QwenDiscovery(config['perception']['mllm'], device=device)
        processor = AuditedProcessor(self.discovery.processor)
        self.discovery.processor = processor
        self.answerer = AuditedQwenAnswerer.from_discovery(config['qa']['mllm'], self.discovery)
        self.runtime = SimpleNamespace(model=self.discovery.model, processor=processor,
            transport_processor=processor, device=device, trace_media=self.trace_media)
        self.relevance = FrozenRelevanceScorer(self.answerer)
        self.coverage = SharedVisionCoverage(self.answerer)

    @contextmanager
    def trace_media(self):
        # The native SharedVisionCoverage checks exact Qwen pixels and grid on reuse.
        yield []
