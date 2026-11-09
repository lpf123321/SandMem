"""Run the same fixed four-frame CPU replay with the isolated v2 parser."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import sandbox_vlm.backbones.spatial_fast_syntax as original
    from sandbox_vlm.backbones.spatial_fast_syntax_v2 import install_fast_discovery_grammar
    original.install_fast_discovery_grammar = install_fast_discovery_grammar
    from profile_spatial_discovery_speed import replay
    replay(a.output.resolve(), optimized=True)
    receipt = a.output / 'receipt.json'
    data = json.loads(receipt.read_text())
    data.update(variant='v2: skip impossible raw-control trie heads and cache immutable stack lookups',
        files_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'sandbox_vlm/backbones/spatial_fast_syntax_v2.py',
            'scripts/profile_spatial_discovery_speed_v2.py', 'scripts/profile_spatial_discovery_speed.py')})
    receipt.write_text(json.dumps(data, indent=2))
