"""Transport contracts: preserve algorithms while translating backbone APIs."""
from types import SimpleNamespace

import pytest
import torch

from sandbox_vlm.backbones.spatial_transport import (
    SpatialBatch, SpatialModel, SpatialAnswerer, SpatialDiscovery, spatial_config,
)
from sandbox_vlm.qa.model import QwenAnswerer
from sandbox_vlm.perception.discovery import QwenDiscovery
from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
from sandbox_vlm.perception.coverage import SharedVisionCoverage


class Visual(torch.nn.Module):
    dtype = torch.bfloat16
    def forward(self, pixels, grid_thw):
        return pixels.reshape(-1, 64)


class RawModel:
    def __init__(self):
        self.visual = Visual()
        self.config = SimpleNamespace(image_token_id=7, max_position_embeddings=128000)
        self.calls = []

    def __call__(self, **kwargs):
        assert 'logits_to_keep' not in kwargs
        features = self.visual(kwargs['pixel_values'].to(self.visual.dtype), grid_thw=kwargs['image_grid_thw'])
        # Spatial stream must remain present even when the 2D tower is reused.
        assert len(kwargs['image_tchw']) == len(kwargs['image_grid_thw'])
        self.calls.append(kwargs['image_tchw'])
        return SimpleNamespace(logits=features)

    def generate(self, **kwargs):
        return self(**kwargs)


def test_configuration_only_replaces_checkpoint_identities():
    import json
    from pathlib import Path
    original = json.loads((Path(__file__).parents[1]/'configs/reference_qwen.json').read_text())
    new = spatial_config(original)
    assert new != original
    for section in ('perception', 'qa'):
        for key in ('model_id', 'revision'):
            assert new[section]['mllm'][key] != original[section]['mllm'][key]
            new[section]['mllm'][key] = original[section]['mllm'][key]
    assert new == original


def test_inherited_algorithms_are_not_reimplemented():
    assert SpatialDiscovery.discover_batch.__name__ == 'discover_batch'
    assert SpatialDiscovery._repeat_guard is QwenDiscovery._repeat_guard
    assert SpatialAnswerer.answer.__name__ == QwenAnswerer.answer.__name__
    # Selection and score computation are the original implementation.
    from sandbox_vlm.backbones.spatial_transport import SpatialCoverage
    assert SpatialCoverage.encode is SharedVisionCoverage.encode
    assert SpatialCoverage.reuse is SharedVisionCoverage.reuse


def test_nested_spatial_tensors_move_with_token_tensors():
    values = SpatialBatch(input_ids=torch.ones(1, 2), image_tchw=[torch.zeros(2, 3, 4, 4)],
                          video_tchw=None, second_per_grid_ts=[1.0])
    moved = values.to('cpu')
    assert moved['image_tchw'][0].device.type == 'cpu'
    assert moved['second_per_grid_ts'] == [1.0] and moved['video_tchw'] is None


def test_spatial_config_context_is_used_without_old_qwen_context():
    wrapper = SpatialModel(RawModel(), 2)
    assert wrapper.config.text_config.max_position_embeddings == 128000


def test_scoped_tower_reuse_preserves_spatial_forward_and_restores(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    raw = RawModel()
    wrapper = SpatialModel(raw, 2)
    processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    encoder = SharedVisionCoverage(SimpleNamespace(model=wrapper, processor=processor, device='cpu'))
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = torch.arange(128, dtype=torch.float32).reshape(8, 16) / 10
    batch = encoder.encode((SpatialBatch(pixel_values=pixels, image_grid_thw=grid),
                            dict(image_grid_thw=grid.tolist())))
    original = raw.visual.forward
    with encoder.reuse(batch, [1]) as calls:
        result = wrapper(pixel_values=pixels[4:], image_grid_thw=grid[1:],
                         image_tchw=[torch.zeros(2, 3, 28, 28)], logits_to_keep=1)
        assert torch.equal(result.logits, pixels[4:].reshape(1, 64).bfloat16())
    assert calls == [1] and len(raw.calls) == 1
    assert raw.visual.forward == original
    assert batch['deepstack'] == []


def test_reuse_rejects_changed_pixels_and_restores_after_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    raw = RawModel()
    wrapper = SpatialModel(raw, 2)
    encoder = SharedVisionCoverage(SimpleNamespace(model=wrapper,
        processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)), device='cpu'))
    grid = torch.tensor([[1, 2, 2]])
    pixels = torch.ones(4, 16)
    batch = encoder.encode((SpatialBatch(pixel_values=pixels, image_grid_thw=grid),
                            dict(image_grid_thw=grid.tolist())))
    original = raw.visual.forward
    with pytest.raises(ValueError, match='different pixels'):
        with encoder.reuse(batch, [0]):
            wrapper.generate(pixel_values=pixels+1, image_grid_thw=grid,
                             image_tchw=[torch.zeros(2, 3, 28, 28)])
    assert raw.visual.forward == original


def test_original_relevance_uses_same_last_token_margin(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    class Model:
        def __call__(self, **kwargs):
            assert kwargs['use_cache'] is False and kwargs['logits_to_keep'] == 1
            return SimpleNamespace(logits=torch.tensor([[[0., 1., 4.], [0., 8., 3.]]]))
    scorer = object.__new__(FrozenRelevanceScorer)
    scorer.model, scorer.device, scorer.token_ids = Model(), 'cpu', [1, 2]
    margins, stats = scorer.forward((SpatialBatch(input_ids=torch.ones(1, 2)), {}))
    assert margins == [5.0]


def test_alignment_exception_does_not_allow_larger_source_pixels():
    from PIL import Image
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.backbones.spatial_transport import spatial_limits
    original = QABudget(33, 1280, 8192, 32768, 256)
    adjusted = spatial_limits(original, [Image.new('RGB', (1280, 960))])
    assert adjusted.max_image_side == 1288
    assert {k:v for k,v in vars(original).items() if k != 'max_image_side'} == {
        k:v for k,v in vars(adjusted).items() if k != 'max_image_side'}
    assert spatial_limits(original, [Image.new('RGB', (1288, 960))]) is original


def test_official_imports_do_not_replace_external_geometry_namespace():
    import sys
    from sandbox_vlm.backbones.spatial_transport import isolated_vggt_imports
    original = SimpleNamespace(identity='external_geometry')
    previous = sys.modules.get('vggt')
    previous_path = list(sys.path)
    sys.modules['vggt'] = original
    try:
        with pytest.raises(RuntimeError, match='load failure'):
            with isolated_vggt_imports():
                assert 'vggt' not in sys.modules
                sys.modules['vggt'] = SimpleNamespace(identity='internal_spatial')
                sys.path.insert(0, '/test/official/external')
                raise RuntimeError('load failure')
        assert sys.modules['vggt'] is original and sys.path == previous_path
    finally:
        if previous is None:
            sys.modules.pop('vggt', None)
        else:
            sys.modules['vggt'] = previous
