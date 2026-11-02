"""D048 deterministic actual-region evidence, without identity or count inference."""
import json
import math

from PIL import Image, ImageDraw, ImageFont


def uniform_indices(length, count):
    if count <= 0 or length <= 0:
        return []
    count = min(length, count)
    if count == 1:
        return [0]
    return [math.floor(i * (length - 1) / (count - 1) + .5) for i in range(count)]


def plan_regions(state, original_frames, *, watermark, tiles_per_panel=4):
    if tiles_per_panel != 4 or not original_frames or len(set(original_frames)) != len(original_frames):
        raise ValueError('requires unique original frames and four-tile panels')
    if sorted(original_frames) != original_frames or max(original_frames) > watermark:
        raise ValueError('nonchronological or future original frame')
    known = {i['visual_instance_id']: i for i in state['instances']}
    observations = {gid: [] for gid in known}
    for frame, links in state['viewpoints'].items():
        fid = int(frame)
        if fid > watermark:
            raise ValueError('future actual observation')
        for link in links:
            gid, evidence = link['visual_instance_id'], link['evidence']
            if gid not in known or evidence['frame_id'] != fid or not evidence['sam_mask_nonempty']:
                raise ValueError('unsupported or dangling crop')
            box, hw = evidence['bbox_xyxy'], evidence['source_hw']
            if (len(box) != 4 or len(hw) != 2 or min(hw) <= 0 or
                any(not math.isfinite(x) for x in box) or
                not (0 <= box[0] < box[2] <= hw[1] and 0 <= box[1] < box[3] <= hw[0])):
                raise ValueError('invalid actual observation box')
            observations[gid].append(dict(visual_id=gid, frame_id=fid, name=known[gid]['name'],
                bbox_xyxy=box, source_hw=hw))
    if any(not links for links in observations.values()):
        raise ValueError('hypothesis without actual source evidence')
    anchor_count = math.ceil(len(original_frames) / 2)
    anchors = [original_frames[i] for i in uniform_indices(len(original_frames), anchor_count)]
    full_supported = {gid for gid, links in observations.items() if any(o['frame_id'] in anchors for o in links)}
    available = []
    for gid, links in observations.items():
        if gid not in full_supported:
            best = max(links, key=lambda o: ((o['bbox_xyxy'][2]-o['bbox_xyxy'][0]) *
                (o['bbox_xyxy'][3]-o['bbox_xyxy'][1]) / (o['source_hw'][0]*o['source_hw'][1]), -o['frame_id']))
            available.append(best)
    available.sort(key=lambda o: (o['frame_id'], o['visual_id']))
    capacity = (len(original_frames) - anchor_count) * tiles_per_panel
    crops = [available[i] for i in uniform_indices(len(available), capacity)]
    crop_ids = {c['visual_id'] for c in crops}
    baseline_supported = {gid for gid, links in observations.items() if any(o['frame_id'] in original_frames for o in links)}
    return dict(full_frame_ids=anchors, crops=crops,
        panels=[crops[i:i+4] for i in range(0, len(crops), 4)],
        full_supported_visual_ids=sorted(full_supported), crop_supported_visual_ids=sorted(crop_ids),
        text_only_visual_ids=sorted(set(known)-full_supported-crop_ids),
        baseline_image_supported_visual_ids=sorted(baseline_supported),
        candidate_image_supported_visual_ids=sorted(full_supported | crop_ids),
        source_frame_ids=sorted(set(anchors) | {c['frame_id'] for c in crops}),
        layout_has_no_spatial_meaning=True, identity_decisions_changed=False)


def crop_bounds(box, source_hw, *, context_fraction=.10):
    if not 0 <= context_fraction <= 1:
        raise ValueError('invalid crop context')
    x0, y0, x1, y1 = box
    h, w = source_hw
    return [max(0, math.floor(x0-(x1-x0)*context_fraction)),
            max(0, math.floor(y0-(y1-y0)*context_fraction)),
            min(w, math.ceil(x1+(x1-x0)*context_fraction)),
            min(h, math.ceil(y1+(y1-y0)*context_fraction))]


def render_panel(crops, images_by_frame, size_wh):
    regions = {}
    for item in crops:
        source = images_by_frame[item['frame_id']]
        if [source.height, source.width] != item['source_hw']:
            raise ValueError('source dimensions differ from actual box grid')
        regions[item['visual_id']] = source.crop(crop_bounds(item['bbox_xyxy'], item['source_hw']))
    return render_crop_panel(crops, regions, size_wh)


def render_crop_panel(crops, regions_by_id, size_wh):
    if not 1 <= len(crops) <= 4 or min(size_wh) < 128:
        raise ValueError('invalid panel')
    canvas = Image.new('RGB', tuple(size_wh), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    w, h = size_wh
    for index, item in enumerate(crops):
        x0, y0 = (index % 2)*w//2, (index//2)*h//2
        x1, y1 = ((index % 2)+1)*w//2, ((index//2)+1)*h//2
        image = regions_by_id[item['visual_id']].copy()
        # Fit small actual regions as well as large ones. Upsampling is ordinary
        # interpolation of source pixels, not generated detail or new evidence.
        scale = min((x1-x0-8)/image.width, (y1-y0-32)/image.height)
        image = image.resize((max(1, math.floor(image.width*scale)),
                              max(1, math.floor(image.height*scale))), Image.Resampling.LANCZOS)
        canvas.paste(image, (x0+(x1-x0-image.width)//2, y0+30+(y1-y0-30-image.height)//2))
        draw.rectangle((x0, y0, x1-1, y1-1), outline='gray')
        draw.text((x0+4, y0+3), f"H{item['visual_id']} F{item['frame_id']}", fill='black', font=font)
        name = item['name'].encode('ascii', errors='replace').decode()[:24]
        draw.text((x0+4, y0+15), name, fill='black', font=font)
    return canvas


def panel_messages(question, images, labels, original_memory_text, plan):
    if len(images) != len(labels):
        raise ValueError('missing image provenance label')
    support = {k: plan[k] for k in ('full_frame_ids', 'full_supported_visual_ids',
        'crop_supported_visual_ids', 'text_only_visual_ids')}
    introduction = (
        'Full video frames below are in chronological order. Region panels contain actual crops; '
        'use each F label as its source frame ID and observation order. H labels refer to visual '
        'hypotheses in the unchanged historical memory, not confirmed physical identities. '
        'Panel tile placement has no spatial meaning. Views and different H labels may depict '
        'the same physical object. Crops include box context and are not new detections. '
        'The support lists distinguish full-frame, crop and text-only records. '
        'Possible correspondences in memory remain uncertain. Use the visible evidence and '
        'historical memory to answer the original question. Return only the requested short answer, '
        'without explanation.\nVisual support (JSON data):\n' + json.dumps(support, separators=(',', ':')))
    content = [dict(type='text', text=introduction + original_memory_text)]
    for image, label in zip(images, labels):
        content.extend([dict(type='text', text=label), dict(type='image', image=image)])
    content.append(dict(type='text', text='Original question (verbatim):\n' + question))
    return [dict(role='user', content=content)]
