"""Check bounded coordinate metadata against full uncompressed coordinates."""
from pathlib import Path
from types import SimpleNamespace
import torch

from causal_adapter import SpatialCausalMem


torch.manual_seed(17)
runtime = SimpleNamespace(device='cpu', model=SimpleNamespace(config=SimpleNamespace(
    hidden_size=16, vision_config=SimpleNamespace(tokens_per_second=25))))
adapter = SpatialCausalMem(runtime, Path('work/baselines_vsibench_20261105/repos/CausalMem'),
                          budget=23, sample_fps=1)
all_coordinates = []
for grid in [(2, 6, 8), (1, 8, 8), (3, 4, 10)]:
    t, h, w = grid
    h //= 2
    w //= 2
    first_pair = adapter.pair_count
    values = torch.randn(t * h * w, 16)
    for index in range(t):
        all_coordinates.append(torch.stack([
            torch.full((h * w,), (first_pair + index) * 50),
            torch.arange(h).repeat_interleave(w), torch.arange(w).repeat(h)]))
    adapter.add(values, grid)
    expected = torch.cat(all_coordinates, dim=1).index_select(1, adapter.cache.mem1_positions)
    assert torch.equal(adapter.positions, expected)
    assert adapter.positions.shape[1] <= 23
    assert torch.equal(adapter.position_token_ids, adapter.cache.mem1_positions)
print('PASS: bounded metadata equals full original M-RoPE coordinates after every chunk, including changing grids')
