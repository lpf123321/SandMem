"""Auditable inputs for the official Spatial-MLLM v1.1 architecture.

This module does not load weights or choose an experiment configuration. Static
images remain independent observations, including the sandbox. Repeating each
image in its own temporal patch is processor padding, not new scene evidence.
"""
from copy import deepcopy
import hashlib


def validate_media(messages, media, *, watermark, require_sandbox=False):
    """Validate explicit media provenance before any image/video is opened.

    Videos must be explicit lists of already selected frames: accepting a full
    movie path here could silently expose future frames in streaming queries.
    """
    if type(watermark) is not int or watermark < 0:
        raise ValueError('explicit nonnegative query watermark required')
    blocks = [c for m in messages for c in m['content'] if c['type'] in ('image', 'video')]
    if len(blocks) != len(media):
        raise ValueError('one provenance record per media block required')
    sandbox_count = 0
    for block, source in zip(blocks, media):
        kind = source['kind']
        if kind not in ('source_frame', 'source_video', 'sandbox'):
            raise ValueError('unknown media provenance')
        ids = source['frame_ids']
        if (not ids or any(type(i) is not int or i < 0 or i > watermark for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError('missing, duplicated, unordered or future source frame IDs')
        if kind == 'source_video':
            if (block['type'] != 'video' or not isinstance(block.get('video'), list)
                    or len(block['video']) != len(ids)):
                raise ValueError('video requires the exact explicit causal frame list')
            if 'nframes' in block or 'fps' in block:
                raise ValueError('selected explicit frames must not be resampled')
        elif block['type'] != 'image':
            raise ValueError('sandbox and single source frames must be image inputs')
        if kind == 'source_frame' and len(ids) != 1:
            raise ValueError('single image needs one source frame ID')
        if kind == 'sandbox':
            sandbox_count += 1
    if require_sandbox and sandbox_count != 1:
        raise ValueError('joint answering requires exactly one actual sandbox image')
    return blocks


def tensor_record(tensor):
    data = tensor.detach().cpu().contiguous()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=hashlib.sha256(data.numpy().tobytes()).hexdigest())


def input_record(value):
    """Processor outputs include scalar video timing metadata as well as tensors."""
    if isinstance(value, (list, tuple)):
        return [input_record(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return tensor_record(value)


def spatial_tensors(image_inputs, video_inputs, image_grid, video_grid, *, patch_size, temporal_patch_size):
    """Match the spatial stream to the *same* RGB pixels and visual patch grids.

    Upstream prepare_spatial_mllm_inputs emits CHW for images, whereas its
    encoder unpacks TCHW and its connector expects temporal_patch_size slots.
    Mirror Qwen's repeated-image padding explicitly, without joining images.
    """
    import numpy as np
    import torch
    image_inputs, video_inputs = image_inputs or [], video_inputs or []
    if len(image_inputs) != len(image_grid) or len(video_inputs) != len(video_grid):
        raise ValueError('media/grid correspondence mismatch')
    images, videos = [], []
    for image, grid in zip(image_inputs, image_grid):
        t, h, w = grid
        value = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1)
        if t != 1 or list(value.shape) != [3, h*patch_size, w*patch_size]:
            raise ValueError('image pixels do not match the processor grid')
        images.append(value.float().div(255).unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1))
    for video, grid in zip(video_inputs, video_grid):
        if isinstance(video, list):
            video = torch.stack([torch.from_numpy(np.array(i.convert('RGB'), copy=True)).permute(2, 0, 1)
                                 for i in video])
        t, h, w = grid
        if list(video.shape) != [t*temporal_patch_size, 3, h*patch_size, w*patch_size]:
            raise ValueError('video pixels/temporal padding do not match the processor grid')
        videos.append(video.float().div(255))
    return images or None, videos or None


def prepare_inputs(processor, model_config, messages, media, *, watermark, require_sandbox=False):
    """Prepare one query using the checkpoint chat template and official utility.

    Returns CPU inputs plus a serializable receipt. No truncation, hidden frame
    selection, answer parsing or model forward is performed here.
    """
    import time
    from qwen_vl_utils import process_vision_info
    started = time.perf_counter()
    blocks = validate_media(messages, media, watermark=watermark, require_sandbox=require_sandbox)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                      return_tensors='pt', padding=True, padding_side='left')
    image_grid = batch['image_grid_thw'].tolist() if 'image_grid_thw' in batch else []
    video_grid = batch['video_grid_thw'].tolist() if 'video_grid_thw' in batch else []
    ip = processor.image_processor
    image_tchw, video_tchw = spatial_tensors(image_inputs, video_inputs, image_grid, video_grid,
        patch_size=ip.patch_size, temporal_patch_size=ip.temporal_patch_size)
    batch.update(image_tchw=image_tchw, video_tchw=video_tchw)
    image_tokens = int((batch['input_ids'] == model_config.image_token_id).sum())
    video_tokens = int((batch['input_ids'] == model_config.video_token_id).sum())
    expected_images = sum(t*h*w//ip.merge_size**2 for t, h, w in image_grid)
    expected_videos = sum(t*h*w//ip.merge_size**2 for t, h, w in video_grid)
    if (image_tokens, video_tokens) != (expected_images, expected_videos):
        raise ValueError('actual image/video token count disagrees with patch grids')
    tensors = {k: input_record(v)
               for k, v in batch.items() if v is not None}
    processed_media = []
    ii = vi = 0
    for block, source in zip(blocks, media):
        if block['type'] == 'image':
            value = image_inputs[ii]; grid = image_grid[ii]; ii += 1
            processed_media.append(dict(**deepcopy(source), grid_thw=grid,
                spatial_shape=[ip.temporal_patch_size, 3, value.height, value.width],
                spatial_repeated_image_padding=ip.temporal_patch_size-1))
        else:
            grid = video_grid[vi]; vi += 1
            processed_media.append(dict(**deepcopy(source), grid_thw=grid,
                spatial_shape=[grid[0]*ip.temporal_patch_size, 3, grid[1]*ip.patch_size, grid[2]*ip.patch_size],
                encoded_frames=grid[0]*ip.temporal_patch_size))
    input_tokens = int(batch['attention_mask'].sum())
    receipt = dict(prompt=prompt, media=processed_media, tensors=tensors,
        input_tokens=input_tokens, visual_tokens=image_tokens+video_tokens,
        text_and_special_input_tokens=input_tokens-image_tokens-video_tokens,
        image_tokens=image_tokens, video_tokens=video_tokens, watermark=watermark,
        image_grid_thw=image_grid, video_grid_thw=video_grid,
        sandbox_in_inputs=any(s['kind'] == 'sandbox' for s in media),
        spatial_encoder_present=True, learned_model_calls=0,
        processor_seconds=time.perf_counter()-started)
    return batch, receipt
