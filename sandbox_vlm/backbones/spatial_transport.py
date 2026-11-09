"""Spatial-MLLM transport for the unchanged SandMem model-facing algorithms.

The inherited discovery, QA, relevance and coverage implementations keep their
prompts, parsing, budgets and selection arithmetic. Only model/processor APIs
are translated here. Loading this module does not load weights.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import sys

from sandbox_vlm.qa.model import QwenAnswerer
from sandbox_vlm.perception.discovery import QwenDiscovery, discovery_generation_kwargs
from sandbox_vlm.perception.coverage import SharedVisionCoverage


class SpatialBatch(dict):
    """Move nested spatial tensors as well as ordinary processor tensors."""
    def to(self, device):
        import torch
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [move(v) for v in value]
            return value
        return SpatialBatch({k: move(v) for k, v in self.items()})


class SpatialProcessor:
    """Expose the old call shape using the checkpoint's own chat template.

    Each batch member stays an independent prompt. Each image, including the
    sandbox, gets its own repeated temporal patch, never a fictitious video.
    """
    def __init__(self, processor):
        self.raw = processor
        self.audit_sink = None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def image_processor(self):
        return self.raw.image_processor

    @image_processor.setter
    def image_processor(self, value):
        self.raw.image_processor = value

    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        if not tokenize:
            return self.raw.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=add_generation_prompt, **kwargs)
        from qwen_vl_utils import process_vision_info
        from sandbox_vlm.backbones.spatial_inputs import spatial_tensors
        # The method has already selected and decoded causal source frames.
        # Reject video inputs rather than implicitly sampling a whole movie.
        batches = messages if isinstance(messages[0], list) else [messages]
        for conversation in batches:
            if any(c['type'] == 'video' for m in conversation for c in m['content']):
                raise ValueError('SandMem transport expects independent selected images')
        texts, images = [], []
        for conversation in batches:
            texts.append(self.raw.apply_chat_template(conversation, tokenize=False,
                add_generation_prompt=add_generation_prompt))
            current, videos = process_vision_info(conversation)
            if videos:
                raise ValueError('unexpected video input')
            images.extend(current or [])
        allowed = {'padding', 'return_tensors', 'return_dict', 'images_kwargs'}
        if set(kwargs) - allowed:
            raise ValueError('unsupported processor options: ' + str(set(kwargs) - allowed))
        if kwargs.get('return_tensors', 'pt') != 'pt':
            raise ValueError('Spatial transport requires torch tensors')
        if kwargs.get('images_kwargs', {}).get('device') not in (None, 'cpu'):
            raise ValueError('pinned Spatial processor uses CPU preprocessing')
        batch = self.raw(text=texts, images=images or None, padding=kwargs.get('padding', False),
                         return_tensors='pt', padding_side='left')
        grid = batch['image_grid_thw'].tolist() if images else []
        ip = self.image_processor
        image_tchw, _ = spatial_tensors(images, None, grid, [],
            patch_size=ip.patch_size, temporal_patch_size=ip.temporal_patch_size)
        batch.update(image_tchw=image_tchw, video_tchw=None)
        if self.audit_sink is not None:
            self.audit_sink(texts, batch)
        return SpatialBatch(batch)


class ConfigView:
    def __init__(self, config):
        self.raw = config

    def __getattr__(self, name):
        # Qwen2.5's text config is the top-level config; Qwen3 nests it.
        if name == 'text_config':
            return self.raw
        return getattr(self.raw, name)


class SpatialModel:
    """Translate Qwen3 feature reuse without bypassing Spatial's spatial stream."""
    def __init__(self, model, merge_size):
        self.raw = model
        self.config = ConfigView(model.config)
        self.merge_size = merge_size
        self.model = self  # scoped SharedVisionCoverage.reuse target
        self._visual_forward = model.visual.forward

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def get_image_features(self, pixel_values, image_grid_thw):
        embeddings = self._visual_forward(pixel_values.type(self.raw.visual.dtype),
                                           grid_thw=image_grid_thw)
        sizes = [t*h*w // self.merge_size**2 for t, h, w in image_grid_thw.tolist()]
        # Spatial/Qwen2.5 has no Qwen3 DeepStack. Reuse the 2D tower only.
        return embeddings.split(sizes), []

    @contextmanager
    def _reuse_visual(self, source_pixels=None, source_grid=None):
        import torch
        original = self.raw.visual.forward
        def forward(pixel_values, grid_thw):
            if source_pixels is not None:
                # Official Spatial casts pixels to BF16 before calling visual.
                # Check that cast, then validate cache identity against the
                # original float processor tensor, exactly as Qwen3 did.
                if (not torch.equal(grid_thw, source_grid) or
                        not torch.equal(pixel_values, source_pixels.to(pixel_values.dtype))):
                    raise ValueError('Spatial visual input differs from processor pixels/grid')
                features, _ = self.get_image_features(source_pixels, source_grid)
            else:
                features, _ = self.get_image_features(pixel_values, grid_thw)
            return torch.cat(features)
        self.raw.visual.forward = forward
        try:
            yield
        finally:
            self.raw.visual.forward = original

    def __call__(self, **kwargs):
        # Official Spatial forward returns all logits; it has no logits_to_keep.
        # Taking the same last-token Yes/No pair preserves the scorer arithmetic.
        keep = kwargs.pop('logits_to_keep', None)
        if keep not in (None, 1):
            raise ValueError('unsupported logits request')
        with self._reuse_visual(kwargs.get('pixel_values'), kwargs.get('image_grid_thw')):
            return self.raw(**kwargs)

    def generate(self, **kwargs):
        with self._reuse_visual(kwargs.get('pixel_values'), kwargs.get('image_grid_thw')):
            return self.raw.generate(**kwargs)


def _attach(instance, config, runtime):
    from sandbox_vlm.backbones.spatial_model import MODEL_ID, REVISION
    if (config['model_id'], config['revision'], config['dtype']) != (MODEL_ID, REVISION, 'bfloat16'):
        raise ValueError('the approved frozen Spatial-135K BF16 configuration is required')
    instance.config, instance.device = dict(config), runtime.device
    instance.processor = runtime.transport_processor
    instance.model = runtime.transport_model
    instance.generation_attempts = 0
    instance.runtime = runtime


class SpatialAnswerer(QwenAnswerer):
    @classmethod
    def from_discovery(cls, config, discovery):
        if any(config[k] != discovery.config[k] for k in ('model_id', 'revision', 'dtype')):
            raise ValueError('perception and QA backbone mismatch')
        return cls(config, discovery.runtime)

    def __init__(self, config, runtime):
        if config.get('temperature', 0) != 0:
            raise ValueError('retain the original greedy generation')
        _attach(self, config, runtime)

    def answer(self, messages, images, limits, **kwargs):
        original = limits
        limits = spatial_limits(limits, images)
        with self.runtime.trace_media() as trace:
            result = super().answer(messages, images, limits, **kwargs)
        result['actual_forward_trace'] = trace
        result['processor_alignment'] = dict(original_limits=vars(original),
            effective_limits=vars(limits), raw_image_limit_unchanged=True,
            approved_processed_side_1288=True)
        return result


class SpatialDiscovery(QwenDiscovery):
    def __init__(self, config, runtime):
        _attach(self, config, runtime)
        self.generation_kwargs = discovery_generation_kwargs(config)

    def discover(self, *args, **kwargs):
        from .spatial_discovery import discovery_instruction, record_instruction
        fid = args[1] if len(args) > 1 else kwargs['frame_id']
        with discovery_instruction(self, [fid], kwargs.get('question')) as enabled:
            with self.runtime.trace_media() as trace:
                result = super().discover(*args, **kwargs)
        record_instruction(result, enabled)
        result['actual_forward_trace'] = trace
        return result

    def discover_batch(self, *args, **kwargs):
        from .spatial_discovery import discovery_instruction, record_instruction
        ids = args[1] if len(args) > 1 else kwargs['frame_ids']
        with discovery_instruction(self, ids, kwargs.get('question')) as enabled:
            with self.runtime.trace_media() as trace:
                results = super().discover_batch(*args, **kwargs)
        for result in results:
            record_instruction(result, enabled)
            result['actual_forward_trace'] = trace
        return results


class SpatialCoverage(SharedVisionCoverage):
    @contextmanager
    def reuse_pixels(self, batch, indices, images):
        # Official utility performs Qwen2.5's own resize before the processor.
        # Keep the exact-checked tower reuse, but recompute CPU transforms so a
        # Qwen3 transform cache cannot be mistaken for Spatial input pixels.
        yield


class SpatialBackbone:
    """One shared frozen checkpoint for coverage, relevance, discovery and QA."""
    def __init__(self, config, official_root, *, device='cuda:0'):
        if any(config.get('spatial_transport',{}).get(k) for k in ('discovery_syntax','sparse_qa_syntax')):
            raise ValueError('syntax policies require spatial_bundle.create_spatial_backbone; do not silently bypass output adapters')
        from sandbox_vlm.backbones.spatial_model import Spatial135K, MODEL_ID, REVISION
        from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
        for section in ('perception', 'qa'):
            setting = config[section]['mllm']
            if (setting['model_id'], setting['revision'], setting['dtype']) != (MODEL_ID, REVISION, 'bfloat16'):
                raise ValueError('both perception and QA must select the approved frozen Spatial checkpoint')
        if config['qa']['mllm'].get('temperature', 0) != 0:
            raise ValueError('retain the original greedy generation')
        # Official Spatial source uses absolute `vggt.*` imports internally.
        # Restore SandMem's external VGGT namespace and path after loading so
        # swapping the MLLM cannot silently swap the geometry model as well.
        with isolated_vggt_imports():
            self.runtime = Spatial135K(official_root, device=device)
        self.runtime.transport_processor = SpatialProcessor(self.runtime.processor)
        self.runtime.transport_model = SpatialModel(self.runtime.model,
            self.runtime.processor.image_processor.merge_size)
        self.answerer = SpatialAnswerer(config['qa']['mllm'], self.runtime)
        self.discovery = SpatialDiscovery(config['perception']['mllm'], self.runtime)
        policy = config.get('spatial_transport', {}).get('discovery_instruction')
        if policy is not None:
            from .spatial_discovery import POLICY
            if policy != POLICY:
                raise ValueError('unknown Spatial discovery instruction policy')
            self.discovery.config['spatial_discovery_instruction'] = policy
        self.relevance = FrozenRelevanceScorer(self.answerer)
        self.coverage = SpatialCoverage(self.answerer)


def spatial_config(reference):
    """Copy an existing method config, changing only the two MLLM identities."""
    from sandbox_vlm.backbones.spatial_model import MODEL_ID, REVISION
    result = deepcopy(reference)
    for section in ('perception', 'qa'):
        result[section]['mllm'].update(model_id=MODEL_ID, revision=REVISION)
    return result


def spatial_limits(limits, images):
    """User-approved native alignment: original pixels <=1280, processed <=1288.

    No other input/output limit changes. Oversized source pixels still fail the
    inherited budget check; this is not permission to enlarge source images.
    """
    if limits.max_image_side == 1280 and all(max(i.size) <= 1280 for i in images):
        return replace(limits, max_image_side=1288)
    return limits


@contextmanager
def isolated_vggt_imports():
    """Isolate the pinned official source's import-time VGGT dependencies.

    Those modules perform their VGGT imports at module scope; instantiated
    model classes retain the imported objects after the namespace is restored.
    Model loading is sequential, never in a concurrent import thread.
    """
    previous_path = list(sys.path)
    saved = {k:v for k,v in sys.modules.items() if k == 'vggt' or k.startswith('vggt.')}
    for key in saved:
        del sys.modules[key]
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == 'vggt' or key.startswith('vggt.'):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = previous_path
