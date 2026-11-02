"""Optional 294-query CPU equivalence replay using an external artifact workspace.

The data root changes; all algorithm imports remain in this release directory.
No weights, generation, scoring, or writes to the reference workspace are used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    reference = args.reference_workspace.resolve()
    if output.exists():
        raise FileExistsError(output)
    if reference in output.parents:
        raise ValueError('write the verification into the new checkout, not the reference workspace')
    spec = importlib.util.spec_from_file_location('frozen_equivalence', ROOT / 'scripts/check_d116_full_equivalence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = reference
    from sandbox_vlm.qa import d116_full
    assert ROOT in Path(d116_full.__file__).resolve().parents
    module.check(output)
    result = json.loads(output.read_text())
    result.update(algorithm_imports_from_new_checkout=True, reference_workspace_read_only=True)
    output.write_text(json.dumps(result, indent=2))
