"""Verify the immutable method snapshot without accessing the old workspace."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check():
    receipt = json.loads((ROOT / 'provenance/frozen_files.json').read_text())
    errors = []
    for name, expected in receipt['source_sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name)
    dependencies = json.loads((ROOT / 'provenance/third_party.json').read_text())
    for component in dependencies.values():
        for name, expected in component['files'].items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(name)
    import sandbox_vlm
    from sandbox_vlm.memory import retention
    from sandbox_vlm.qa import d116_full, two_pass
    origins = [Path(module.__file__).resolve() for module in (sandbox_vlm, retention, d116_full, two_pass)]
    assert all(ROOT in origin.parents for origin in origins), 'imports escaped the release directory'
    return dict(status='passed' if not errors else 'failed',
                frozen_files=len(receipt['source_sha256']),
                dependency_files=sum(len(c['files']) for c in dependencies.values()),
                mismatches=errors, imports_from_release=True, new_model_calls=0)


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'passed')
