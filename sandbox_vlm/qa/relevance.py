"""D040: frozen single-forward relevance and bounded chronological selection.

Scores are Yes/No logit margins, not calibrated probabilities or identities.
No targets, task parser, extra encoder, generation, or geometry is used here.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def uniform_frames(frame_ids, count):
    if frame_ids != sorted(set(frame_ids)) or count < 1:
        raise ValueError('ordered unique frames and positive budget required')
    if len(frame_ids) <= count:
        return list(frame_ids)
    return [frame_ids[round(i * (len(frame_ids) - 1) / (count - 1))]
            for i in range(count)] if count > 1 else [frame_ids[len(frame_ids) // 2]]


def select_relevant(frame_ids, margins, *, query, count=32, separation=3):
    """Half temporal-bin winners, half global winners with bounded suppression.

    Prefix validation is strict: passing even one future score is an error.
    Filling remaining slots relaxes spacing, never the query boundary.
    """
    if (len(frame_ids) != len(margins) or frame_ids != sorted(set(frame_ids))
            or not frame_ids or min(frame_ids) < 0 or max(frame_ids) > query
            or count < 1 or separation < 0 or not all(math.isfinite(float(s)) for s in margins)):
        raise ValueError('invalid or future relevance evidence')
    count = min(count, len(frame_ids))
    ranking = sorted(range(len(frame_ids)), key=lambda i: (-float(margins[i]), frame_ids[i]))
    selected = set()
    bins = max(1, count // 2)
    for indices in np.array_split(np.arange(len(frame_ids)), bins):
        winner = min(indices.tolist(), key=lambda i: (-float(margins[i]), frame_ids[i]))
        selected.add(winner)
    for i in ranking:
        if len(selected) >= count:
            break
        if all(abs(frame_ids[i] - frame_ids[j]) >= separation for j in selected):
            selected.add(i)
    for i in ranking:
        if len(selected) >= count:
            break
        selected.add(i)
    return sorted(frame_ids[i] for i in selected)


def relevance_messages(question, image):
    return [dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=(
        'Look at this actual video frame. Is the main object or entity asked about in the '
        'following original question visibly present? Consider its stated visual attributes. '
        'Ignore objects mentioned only as answer choices or background context. '
        'Do not solve the question, infer unseen objects, or count across frames. '
        'Reply with exactly Yes or No.\nOriginal question:\n' + question))])]


class FrozenRelevanceScorer:
    def __init__(self, answerer):
        self.model, self.processor, self.device = answerer.model, answerer.processor, answerer.device
        tokenizer = self.processor.tokenizer
        tokens = [tokenizer.encode(s, add_special_tokens=False) for s in ('Yes', 'No')]
        if any(len(t) != 1 for t in tokens) or tokens[0] == tokens[1]:
            raise ValueError('Yes/No must be distinct single tokens for this pinned tokenizer')
        self.token_ids = [t[0] for t in tokens]

    def score(self, question, images, *, processor_device=None):
        prepared = self.prepare(question, images, processor_device=processor_device)
        return self.forward(prepared)

    def prepare(self, question, images, *, processor_device=None):
        """Unchanged processor computation; may run in a single CPU producer."""
        import torch
        if not images:
            raise ValueError('empty relevance batch')
        started = time.perf_counter()
        tokenizer = self.processor.tokenizer
        previous = tokenizer.padding_side
        try:
            tokenizer.padding_side = 'left'
            image_kwargs = dict(images_kwargs=dict(device=processor_device)) if processor_device else {}
            inputs = self.processor.apply_chat_template(
                [relevance_messages(question, image) for image in images],
                tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors='pt', padding=True, **image_kwargs)
        finally:
            tokenizer.padding_side = previous
        if not bool(inputs['attention_mask'][:, -1].all()):
            raise ValueError('last-token relevance logits would refer to padding')
        visual_tokens = (inputs['input_ids'] == self.model.config.image_token_id).sum(dim=1).tolist()
        token_counts = inputs['attention_mask'].sum(dim=1).tolist()
        grid = inputs['image_grid_thw'].tolist()
        if len(grid) != len(images):
            raise ValueError('expected one image per independent prompt')
        expected = [t * h * w // self.processor.image_processor.merge_size ** 2 for t, h, w in grid]
        if visual_tokens != expected:
            raise ValueError('relevance image token accounting mismatch')
        processor_seconds = time.perf_counter() - started
        stats = dict(logical_single_image_requests=len(images), forward_calls=1,
            processor_seconds=processor_seconds, visual_tokens=visual_tokens, input_tokens=token_counts,
            padded_input_length=int(inputs['input_ids'].shape[-1]), image_grid_thw=grid,
            yes_no_token_ids=self.token_ids, use_cache=False, generated_tokens=0)
        return inputs, stats

    def forward(self, prepared):
        """Consume one already prepared independent-prompt batch, without a KV cache."""
        import torch
        inputs, stats = prepared
        inputs = inputs.to(self.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, logits_to_keep=1)
            pair = output.logits[:, -1, self.token_ids].float()
            margins = (pair[:, 0] - pair[:, 1]).cpu().tolist()
        torch.cuda.synchronize()
        model_seconds = time.perf_counter() - started
        return margins, dict(stats, model_seconds=model_seconds)


class PrefixScores:
    """One fixed-capacity numeric index; all live retrieval state is charged."""
    def __init__(self, capacity, *, max_bytes):
        if capacity < 1 or capacity * 8 > max_bytes:
            raise ValueError('relevance index exceeds structural budget')
        self.ids = np.empty(capacity, dtype=np.int32)
        self.margins = np.empty(capacity, dtype=np.float32)
        self.size = 0

    @property
    def array_bytes(self):
        return self.ids.nbytes + self.margins.nbytes

    def append(self, ids, margins, *, query):
        if (len(ids) != len(margins) or ids != sorted(set(ids)) or not ids
                or min(ids) < 0 or max(ids) > query or self.size + len(ids) > len(self.ids)
                or (self.size and ids[0] <= int(self.ids[self.size - 1]))
                or not all(math.isfinite(float(s)) for s in margins)):
            raise ValueError('invalid, duplicate, overflowing or future score update')
        end = self.size + len(ids)
        self.ids[self.size:end], self.margins[self.size:end] = ids, margins
        self.size = end

    def select(self, *, query, count, separation):
        return select_relevant(self.ids[:self.size].tolist(), self.margins[:self.size].tolist(),
                               query=query, count=count, separation=separation)

    def select_adaptive(self, *, query, sparse_fraction=0.05, sparse_count=8,
                        dense_count=32, separation=3):
        ids, scores = self.ids[:self.size].tolist(), self.margins[:self.size].tolist()
        if not ids or max(ids) > query or not 0 <= sparse_fraction <= 1:
            raise ValueError('invalid or future adaptive relevance evidence')
        positive_count = sum(s > 0 for s in scores)
        sparse = positive_count / len(ids) <= sparse_fraction
        if sparse:
            # Adjacent source frames may contain different brief observations;
            # do not suppress these merely because their timestamps are close.
            ranking = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
            selected = sorted(ids[i] for i in ranking[:sparse_count])
        else:
            selected = select_relevant(ids, scores, query=query, count=dense_count, separation=separation)
        return selected, dict(mode='sparse_top_score' if sparse else 'dense_temporal_coverage',
            positive_logit_margin_count=positive_count, screened_frames=len(ids),
            positive_fraction=positive_count / len(ids), sparse_fraction_threshold=sparse_fraction,
            calibrated_probability=False)


def prefetched_batches(reader, frame_ids, *, query, batch_size, image_side):
    """Overlap one future decode batch with GPU work, within this query only.

    At most two batches of resized images are live. Exiting joins the worker;
    no worker may advance the video while answering this or the next query.
    """
    if (batch_size < 1 or frame_ids != sorted(set(frame_ids))
            or (frame_ids and (min(frame_ids) < 0 or max(frame_ids) > query))):
        raise ValueError('invalid or future prefetch request')
    if not frame_ids:
        return
    chunks = iter(frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size))

    def load(ids):
        started = time.perf_counter()
        images = [reader.read(fid, allowed_through=query, max_side=image_side) for fid in ids]
        return ids, images, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bounded-frame-decode') as pool:
        pending = pool.submit(load, next(chunks))
        while pending is not None:
            current = pending.result()
            following = next(chunks, None)
            pending = pool.submit(load, following) if following is not None else None
            yield current


def prepared_batches(reader, scorer, question, frame_ids, *, query, batch_size, image_side):
    """D042: pipeline identical CPU decode+processor with one GPU consumer.

    One CPU worker owns the decoder and tokenizer until the query ends. Exact
    frame/batch order is preserved. At most two prepared CPU batches are live;
    callers release each consumed batch before requesting the next. Joining at
    exit prevents preprocessing or reads during QA or beyond this watermark.
    """
    if (batch_size < 1 or frame_ids != sorted(set(frame_ids))
            or (frame_ids and (min(frame_ids) < 0 or max(frame_ids) > query))):
        raise ValueError('invalid or future prepared-batch request')
    if not frame_ids:
        return
    chunks = iter(frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size))

    def load(ids):
        started = time.perf_counter()
        images = [reader.read(fid, allowed_through=query, max_side=image_side) for fid in ids]
        decode_seconds = time.perf_counter() - started
        prepared = scorer.prepare(question, images)
        return ids, prepared, decode_seconds

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bounded-cpu-input') as pool:
        pending = pool.submit(load, next(chunks))
        while pending is not None:
            current = pending.result()
            following = next(chunks, None)
            pending = pool.submit(load, following) if following is not None else None
            yield current
