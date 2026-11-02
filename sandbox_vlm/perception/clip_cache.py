"""Bounded, lossless reuse of frozen SAM image features within one clip only."""
from contextlib import contextmanager


def tensor_storage_bytes(value):
    """Actual reachable tensor storage, with shared/view storage counted once."""
    seen, total = set(), 0
    def visit(item):
        nonlocal total
        if hasattr(item, 'untyped_storage'):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key); total += storage.nbytes()
        elif isinstance(item, dict):
            for entry in item.values(): visit(entry)
        elif isinstance(item, (tuple, list)):
            for entry in item: visit(entry)
    visit(value)
    return total


@contextmanager
def clip_image_feature_cache(predictor, max_frames, *, enabled=False):
    """Keep the official per-image outputs, never tracking or prompt state.

    The official getter replaces its cache on every miss. Preserve its result
    and restore earlier entries after that getter returns. No dtype conversion,
    learned computation, cross-clip cache, or changes to mask propagation.
    """
    if type(max_frames) is not int or not 1 <= max_frames <= 16 or type(enabled) is not bool:
        raise ValueError('clip cache requires a boolean flag and at most 16 frames')
    original = predictor._get_image_feature
    had_override = '_get_image_feature' in predictor.__dict__
    state_owner = None
    stats = dict(enabled=enabled, image_encoder_forward_calls=0, cache_hits=0,
                 max_cached_frames=0, peak_feature_storage_bytes=0)
    def get(state, frame_idx, batch_size):
        nonlocal state_owner
        if state_owner is None: state_owner = state
        if state is not state_owner or state['num_frames'] > max_frames or not 0 <= frame_idx < state['num_frames']:
            raise ValueError('image feature cache crossed its bounded clip')
        cache = state['cached_features']
        hit = frame_idx in cache
        result = original(state, frame_idx, batch_size)
        stats['cache_hits' if hit else 'image_encoder_forward_calls'] += 1
        if enabled:
            cache.update(state['cached_features'])
            state['cached_features'] = cache
        current = state['cached_features']
        stats['max_cached_frames'] = max(stats['max_cached_frames'], len(current))
        # Storage grows only on a miss. Counting it on every hit would add CPU work.
        if not hit:
            stats['peak_feature_storage_bytes'] = max(stats['peak_feature_storage_bytes'], tensor_storage_bytes(current))
        return result
    predictor._get_image_feature = get
    try:
        yield stats
    finally:
        if had_override: predictor._get_image_feature = original
        else: del predictor._get_image_feature
        if enabled and state_owner is not None:
            state_owner['cached_features'].clear()
