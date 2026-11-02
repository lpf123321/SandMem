"""D042 bounded CPU input producer in a spawn process; no model weights/CUDA.

The processor and decoder have one owner. Only current-query batches enter the
queue, and the parent waits for their completion before QA. CPU tensors use
PyTorch's shared-memory transport; this is temporary inference workspace.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

_state = {}


def _initialize(model_id, revision, image_token_id):
    import time
    started = time.perf_counter()
    import torch
    import torch.multiprocessing  # register CPU tensor shared-memory reducers
    import cv2
    from transformers import AutoProcessor
    from types import SimpleNamespace
    from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
    stub = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id))
    _state['scorer'] = FrozenRelevanceScorer(SimpleNamespace(model=stub, processor=processor, device='cpu'))
    _state['initialization_seconds'] = time.perf_counter() - started


def _prepare(video_path, mapping_path, question, ids, query, image_side, audit):
    import hashlib
    import resource
    import time
    from sandbox_vlm.data.video import FrameMapping
    from compare_dev_frame_evidence import SequentialPrefixReader
    if not ids or ids != sorted(set(ids)) or ids[-1] > query:
        raise ValueError('invalid/future CPU producer request')
    identity = (video_path, mapping_path)
    if _state.get('video') != identity:
        if 'reader' in _state:
            _state['reader'].close()
        _state['reader'] = SequentialPrefixReader(video_path, FrameMapping.load(mapping_path))
        _state['video'] = identity
    started = time.perf_counter()
    images = [_state['reader'].read(fid, allowed_through=query, max_side=image_side) for fid in ids]
    decode = time.perf_counter() - started
    inputs, stats = _state['scorer'].prepare(question, images)
    if any(t.device.type != 'cpu' for t in inputs.values() if hasattr(t, 'device')):
        raise ValueError('CPU producer must not allocate model inputs on CUDA')
    if audit:
        stats['processor_tensor_sha256'] = {k: hashlib.sha256(t.numpy().tobytes()).hexdigest()
                                            for k, t in inputs.items() if hasattr(t, 'numpy')}
    stats['cpu_producer_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    stats['cpu_producer_initialization_seconds_once_per_process'] = _state['initialization_seconds']
    return ids, (inputs, stats), decode


class ProcessInputStream:
    def __init__(self, config, image_token_id):
        import torch.multiprocessing
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize, initargs=(config['model_id'], config['revision'], image_token_id))

    def batches(self, video_path, mapping_path, question, ids, *, query, batch_size=64,
                image_side=256, audit=False):
        if not ids:
            return
        if ids != sorted(set(ids)) or min(ids) < 0 or max(ids) > query:
            raise ValueError('future or unordered input request')
        chunks = iter(ids[i:i + batch_size] for i in range(0, len(ids), batch_size))

        def submit(chunk):
            return self.pool.submit(_prepare, str(video_path), str(mapping_path), question,
                                    chunk, query, image_side, audit)

        pending = submit(next(chunks))
        try:
            while pending is not None:
                current = pending.result()
                following = next(chunks, None)
                pending = submit(following) if following is not None else None
                yield current
        finally:
            if pending is not None:
                pending.result()  # no background preprocessing may survive this query

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
