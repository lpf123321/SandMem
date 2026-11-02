"""D043 frozen visual coverage, scoped feature reuse and causal shortlists.

Descriptors are visual candidates, never scene labels or instance identities.
Only the current batch keeps full Qwen embeddings. Compact necessary state is
serialized by the caller into the same bounded backend as the sandbox.
"""
import base64
from contextlib import contextmanager
import hashlib
import time

import numpy as np


def pack_descriptor(value, *, quantized=False):
    a = np.asarray(value, dtype=np.float16)
    if a.shape != (16, 64) or not np.isfinite(a).all():
        raise ValueError('invalid compact spatial descriptor')
    if quantized:
        return 'i8:'+base64.b64encode(np.rint(np.clip(a,-1,1)*127).astype(np.int8).tobytes()).decode('ascii')
    return base64.b64encode(a.tobytes()).decode('ascii')


def unpack_descriptor(value):
    quantized=value.startswith('i8:')
    a = np.frombuffer(base64.b64decode(value[3:] if quantized else value, validate=True),
                      dtype=np.int8 if quantized else np.float16)
    if a.size != 1024 or not np.isfinite(a).all():
        raise ValueError('invalid packed descriptor')
    return a.reshape(16, 64).astype(np.float32)/(127 if quantized else 1)


def context_similarity(a, b):
    x, y = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom else 0.


class NoveltyShortlist:
    def __init__(self, *, block_size=16, candidates_per_block=4):
        if not 1 <= candidates_per_block <= block_size:
            raise ValueError('invalid bounded coverage block')
        self.block_size, self.candidates_per_block = block_size, candidates_per_block
        self.last_id, self.last_descriptor = -1, None

    def select(self, ids, descriptors, *, query):
        if (not ids or ids != list(range(self.last_id + 1, self.last_id + 1 + len(ids)))
                or ids[-1] > query or np.asarray(descriptors).shape != (len(ids), 16, 64)):
            raise ValueError('missing, unordered or future coverage frame')
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if not np.isfinite(descriptors).all():
            raise ValueError('nonfinite visual descriptor')
        descriptors /= np.maximum(np.linalg.norm(descriptors, axis=-1, keepdims=True), 1e-12)
        previous = np.concatenate(([self.last_descriptor] if self.last_descriptor is not None
                                   else descriptors[:1], descriptors[:-1]), axis=0)
        change = np.maximum(0., 1. - (descriptors * previous).sum(-1)).max(-1)
        # Both sides of a one-frame event contribute; no threshold discards
        # a frame before the fixed block's coverage/diversity ranking.
        novelty = np.maximum(change, np.r_[change[1:], 0.])
        chosen = []
        for block in sorted({fid // self.block_size for fid in ids}):
            indices = [i for i, fid in enumerate(ids) if fid // self.block_size == block]
            selected = [indices[0]]  # explicit temporal coverage anchor
            while len(selected) < min(self.candidates_per_block, len(indices)):
                def priority(i):
                    diversity = min(max(0., 1. - context_similarity(descriptors[i], descriptors[j])) for j in selected)
                    return (float(novelty[i]) * (diversity + 1e-6), float(novelty[i]), -ids[i])
                selected.append(max((i for i in indices if i not in selected), key=priority))
            # Preserve a bounded three-frame burst around the strongest change.
            # The first probe missed a one-frame development observation when
            # global diversity preferred the frame immediately after the event.
            peak = max(indices, key=lambda i: (float(novelty[i]), -ids[i]))
            selected = sorted(set(selected) | {i for i in (peak-1, peak, peak+1) if i in indices})
            chosen.extend(selected)
        self.last_id, self.last_descriptor = ids[-1], descriptors[-1].copy()
        return sorted(chosen), novelty.tolist()

    def to_dict(self):
        return dict(block_size=self.block_size, candidates_per_block=self.candidates_per_block,
                    peak_burst_radius=1,
                    last_id=self.last_id, last_descriptor=None if self.last_descriptor is None
                    else pack_descriptor(self.last_descriptor))


class SharedVisionCoverage:
    def __init__(self, answerer):
        self.model, self.processor, self.device = answerer.model, answerer.processor, answerer.device
        # The CPU producer must not observe the consumer's scoped processor cache.
        self.image_processor = self.processor.image_processor

    def prepare(self, images):
        started = time.perf_counter()
        values = self.image_processor(images=images, return_tensors='pt')
        return values, dict(processor_seconds=time.perf_counter() - started,
                            logical_images=len(images), image_grid_thw=values['image_grid_thw'].tolist())

    def encode(self, prepared):
        import torch
        import torch.nn.functional as F
        inputs, stats = prepared
        inputs = inputs.to(self.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            embeddings, layers = self.model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])
            descriptors = []
            merge = self.processor.image_processor.merge_size
            for embedding, (t, h, w) in zip(embeddings, stats['image_grid_thw']):
                if t != 1 or embedding.shape[0] != h * w // merge ** 2 or embedding.shape[1] % 64:
                    raise ValueError('unexpected independent-image visual grid')
                grid = embedding.float().reshape(h // merge, w // merge, -1).permute(2, 0, 1)
                pooled = F.adaptive_avg_pool2d(grid, (4, 4)).permute(1, 2, 0).reshape(16, 64, -1).mean(-1)
                descriptors.append(F.normalize(pooled, dim=-1).half())
            # Preserve each image's arithmetic; transfer the compact batch once
            # instead of synchronizing the GPU for every independent image.
            descriptors = torch.stack(descriptors).cpu().numpy()
        torch.cuda.synchronize()
        return dict(inputs=inputs, embeddings=embeddings, deepstack=layers,
                    descriptors=descriptors, stats=dict(stats,
                    vision_seconds=time.perf_counter() - started, vision_forward_calls=1))

    @contextmanager
    def reuse_pixels(self, batch, indices, images):
        """Skip repeated image transforms, checking actual image bytes and order."""
        import torch
        original=self.processor.image_processor
        raw_sizes=batch['inputs']['image_grid_thw'].prod(-1).tolist()
        chunks=batch['inputs']['pixel_values'].split(raw_sizes)
        values=dict(pixel_values=torch.cat([chunks[i] for i in indices]),
                    image_grid_thw=torch.tensor([batch['stats']['image_grid_thw'][i] for i in indices]))
        expected=[hashlib.sha256(i.convert('RGB').tobytes()).hexdigest()for i in images]
        class CachedProcessor:
            def __getattr__(self,name):return getattr(original,name)
            def __call__(self,images,**kwargs):
                def flatten(items):
                    for item in items:
                        if isinstance(item,(list,tuple)):yield from flatten(item)
                        else:yield item
                actual=[hashlib.sha256(i.convert('RGB').tobytes()).hexdigest()for i in flatten(images)]
                if actual!=expected:raise ValueError('cached processor received different source images')
                return values
        self.processor.image_processor=CachedProcessor()
        try:yield
        finally:self.processor.image_processor=original

    @contextmanager
    def reuse(self, batch, indices):
        """Reuse only exactly identical pixels/grid in the current Qwen batch.

        DeepStack outputs must follow the same selected image/token order.
        The monkeypatch is scoped to one sequential consumer and always restored.
        """
        import torch
        core = self.model.model
        original = core.get_image_features
        grid = batch['inputs']['image_grid_thw']
        raw_sizes = grid.prod(-1).tolist()
        token_sizes = [e.shape[0] for e in batch['embeddings']]
        raw_chunks = batch['inputs']['pixel_values'].split(raw_sizes)
        expected_pixels = torch.cat([raw_chunks[i] for i in indices])
        expected_grid = grid[indices]
        selected_embeddings = tuple(batch['embeddings'][i] for i in indices)
        selected_layers = [torch.cat([layer.split(token_sizes)[i] for i in indices]) for layer in batch['deepstack']]
        calls = []

        def cached(pixel_values, image_grid_thw):
            if (not torch.equal(image_grid_thw, expected_grid)
                    or not torch.equal(pixel_values, expected_pixels)):
                raise ValueError('refusing visual reuse for different pixels or image grids')
            calls.append(1)
            return selected_embeddings, selected_layers

        core.get_image_features = cached
        try:
            yield calls
            if calls != [1]:
                raise ValueError('expected exactly one scoped visual reuse')
        finally:
            core.get_image_features = original


def coverage_batches(reader, encoder, ids, *, query, batch_size=64, image_side=256):
    """At most two CPU input batches, joined before the query's discovery/QA."""
    from concurrent.futures import ThreadPoolExecutor
    if ids != sorted(set(ids)) or ids and (ids[0] < 0 or ids[-1] > query) or batch_size < 1:
        raise ValueError('invalid causal coverage input')
    if not ids:
        return
    chunks = iter(ids[i:i + batch_size] for i in range(0, len(ids), batch_size))

    def prepare(chunk):
        started = time.perf_counter()
        images = reader.read_batch(chunk, allowed_through=query, max_side=image_side)
        decode = time.perf_counter() - started
        return chunk, images, encoder.prepare(images), decode

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bounded-coverage-cpu') as pool:
        pending = pool.submit(prepare, next(chunks))
        while pending is not None:
            current = pending.result()
            following = next(chunks, None)
            pending = pool.submit(prepare, following) if following is not None else None
            yield current


def score_visual_batch(vision, scorer, encoded, images, question, indices, *, weak_margin=1., force_full=False):
    """Adaptive language coverage, with identical current-batch pixel reuse.

    Weak visual/semantic hits trigger a full language check of this batch, so
    sparse brief observations are not irreversibly discarded. This is a score
    rule shared by all inputs, not a benchmark/task branch.
    """
    stats=[];total_requests=0;reuse_calls=0
    def score(selected):
        nonlocal total_requests,reuse_calls
        selected_images=[images[i]for i in selected]
        with vision.reuse_pixels(encoded,selected,selected_images):
            prepared=scorer.prepare(question,selected_images)
        with vision.reuse(encoded,selected) as calls:
            values,record=scorer.forward(prepared)
        stats.append(record);total_requests+=len(selected);reuse_calls+=len(calls)
        return values
    if force_full:indices=list(range(len(images)))
    values=score(indices)
    refined=len(indices)<len(images)and max(values)<=weak_margin
    if refined:
        indices=list(range(len(images)));values=score(indices)
    return indices,values,dict(language_calls=stats,logical_language_image_requests=total_requests,
        vision_reuse_calls=reuse_calls,weak_batch_full_check=refined or force_full,pixel_processor_reuse=True,
        next_batch_prefer_full=max(values)<=weak_margin)
