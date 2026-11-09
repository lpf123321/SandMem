"""A common frozen dual-stream Spatial visual encoder for baseline adapters."""
import hashlib
import json
from pathlib import Path
import time


def sample_frames(path, fps):
    import decord
    import numpy as np
    video = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    source_fps = float(video.get_avg_fps())
    # Time-based sampling, with no question, type or answer dependence.
    times = np.arange(0, len(video) / source_fps, 1 / fps)
    indices = sorted(set(min(len(video) - 1, int(t * source_fps)) for t in times))
    return video, indices, source_fps


def encode_chunk(runtime, frames, source_ids, *, sample_fps):
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from sandbox_vlm.backbones.spatial_inputs import spatial_tensors, input_record
    images = [Image.fromarray(frame) for frame in frames]
    content = dict(type='video', video=images)
    messages = [dict(role='user', content=[content])]
    _, videos = process_vision_info(messages)
    processor = runtime.processor
    batch = processor(text=[''], videos=videos, return_tensors='pt')
    grid = batch['video_grid_thw'].tolist()
    _, spatial = spatial_tensors(None, videos, [], grid,
        patch_size=processor.image_processor.patch_size,
        temporal_patch_size=processor.image_processor.temporal_patch_size)
    pixels = batch['pixel_values_videos'].to(runtime.device, runtime.model.visual.dtype)
    spatial = [value.to(runtime.device, runtime.model.visual.dtype) for value in spatial]
    grid_tensor = batch['video_grid_thw'].to(runtime.device)
    started = time.perf_counter()
    torch.cuda.synchronize()
    with torch.inference_mode():
        visual = runtime.model.visual(pixels, grid_thw=grid_tensor)
        spatial_embeddings, patch_start = runtime.model.spatial_encoder(spatial, grid_thw=grid_tensor)
        fused = runtime.model.connector(video_embeds=visual, spatial_embeds_list=spatial_embeddings,
                                        patch_start_idx=patch_start, grid_thw=grid_tensor)
    torch.cuda.synchronize()
    expected = sum(t * h * w // processor.image_processor.merge_size**2 for t, h, w in grid)
    if fused.shape != (expected, runtime.model.config.hidden_size):
        raise ValueError('fused features do not match native visual grid')
    padded_ids = list(source_ids)
    while len(padded_ids) < grid[0][0] * processor.image_processor.temporal_patch_size:
        padded_ids.append(source_ids[-1])
    receipt = dict(source_frame_ids=list(source_ids), padded_source_frame_ids=padded_ids,
                   video_grid_thw=grid[0], sample_fps=sample_fps,
                   video_pixels=input_record(batch['pixel_values_videos']),
                   spatial_input=[dict(shape=list(v.shape), dtype=str(v.dtype)) for v in spatial],
                   fused_shape=list(fused.shape), visual_tokens=expected,
                   visual_encoder_calls=1, spatial_encoder_calls=1, connector_calls=1,
                   encoder_seconds=time.perf_counter() - started)
    return fused, receipt, dict(pixel_values_videos=pixels, video_grid_thw=grid_tensor, video_tchw=spatial)


def check_native_equivalence(runtime, frames, source_ids, *, sample_fps):
    """Compare adapter features to the actual official model connector output."""
    import torch
    features, receipt, media = encode_chunk(runtime, frames, source_ids, sample_fps=sample_fps)
    count = features.shape[0]
    ids = runtime.processor.tokenizer('<|im_start|>user\n<|vision_start|>', add_special_tokens=False).input_ids
    ids += [runtime.model.config.video_token_id] * count
    ids += runtime.processor.tokenizer('<|vision_end|>\n<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False).input_ids
    ids = torch.tensor([ids], device=runtime.device)
    observed = []
    handle = runtime.model.connector.register_forward_hook(lambda module, args, output: observed.append(output.detach()))
    try:
        with torch.inference_mode(), runtime.trace_media() as trace:
            runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          second_per_grid_ts=[2 / sample_fps], **media, use_cache=False)
    finally:
        handle.remove()
    if len(observed) != 1 or not torch.equal(observed[0], features):
        raise AssertionError('adapter fused features differ from official forward')
    return receipt, dict(bit_exact_fused_embeddings=True, actual_forward_trace=trace)
