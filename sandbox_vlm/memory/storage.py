"""Measure reachable CPU structure storage; never label this as GPU usage."""
import sys

import numpy as np


def cpu_storage_report(root) -> dict:
    """Unique reachable Python sizes plus owned NumPy buffers, including view bases.

    sys.getsizeof includes data for owning ndarrays, only a header for views.
    This measures retained object storage, not allocator fragmentation/RSS or a
    process peak. No persisted index is built; traversal state is temporary.
    """
    seen = set()
    numpy_buffer_bytes = 0

    def visit(value):
        nonlocal numpy_buffer_bytes
        if id(value) in seen:
            return 0
        seen.add(id(value))
        size = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object-dtype arrays are unsupported structural storage")
            if value.flags.owndata:
                numpy_buffer_bytes += value.nbytes
            elif value.base is not None:
                size += visit(value.base)
        elif isinstance(value, dict):
            size += sum(visit(k) + visit(v) for k, v in value.items())
        elif isinstance(value, (list, tuple, set, frozenset)):
            size += sum(visit(v) for v in value)
        elif hasattr(value, "__dict__"):
            size += visit(vars(value))
        return size

    total = visit(root)
    return {"cpu_structural_bytes": total, "numpy_owned_buffer_bytes": numpy_buffer_bytes,
            "gpu_structural_bytes_measured": None,
            "storage_measurement": "unique reachable CPython sizes and NumPy owning buffers; not RSS/allocator peak"}
