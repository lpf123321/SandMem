"""Verify and archive the joint input actually handed to the answer transport.

Integrity checks only: no selection, prompting, geometry or identity rules. An
empty confirmed map remains visible as such; passing is not model acceptance.
"""
import hashlib
import json
from pathlib import Path

from sandbox_vlm.memory import Sandbox
from sandbox_vlm.qa.render import SPACE_NOTICE


def state_sha256(sandbox):
    encoded = json.dumps(sandbox.to_dict(), sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def pixel_sha256(image):
    header = f'{image.mode}:{image.width}:{image.height}:'.encode('ascii')
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def serial_messages(messages, images):
    """Index the very same image objects consumed by the model's processor."""
    result, index = [], 0
    for message in messages:
        content = []
        for item in message['content']:
            if item['type'] == 'image':
                if index >= len(images) or item.get('image') is not images[index]:
                    raise ValueError('joint message images differ from the charged image sequence')
                content.append(dict(type='image', evidence_image_index=index,
                                    pixel_sha256=pixel_sha256(images[index])))
                index += 1
            else:
                content.append(dict(item))
        result.append(dict(role=message['role'], content=content))
    if index != len(images):
        raise ValueError('charged image absent from the actual model messages')
    return result


def verify_joint_input(sandbox, question, images, metadata, messages, limits):
    if not isinstance(sandbox, Sandbox) or sandbox.validate():
        raise ValueError('joint QA requires a valid spatial sandbox')
    records, selection = metadata['image_records'], metadata['selection']
    if (not images or len(records) != len(images) or records[0]['kind'] != 'sandbox'
            or any(r['kind'] != 'source_frame' for r in records[1:])):
        raise ValueError('joint QA requires one sandbox canvas followed by source frames')
    frame_ids = [r['frame_id'] for r in records[1:]]
    if frame_ids != selection['frame_ids'] or len(set(frame_ids)) != len(frame_ids):
        raise ValueError('selected source-frame identities differ from actual input')
    if (metadata['original_question'] != question or selection['question'] != question
            or messages[-1]['content'][-1] != dict(type='text', text='Original question (verbatim):\n' + question)):
        raise ValueError('original question changed in joint QA')
    if metadata['space_notice'] != SPACE_NOTICE or SPACE_NOTICE not in messages[0]['content'][0]['text']:
        raise ValueError('independent-reference notice absent from model input')
    if len(images) > limits.max_images or any(max(i.size) > limits.max_image_side for i in images):
        raise ValueError('joint image count or resolution exceeds the configured budget')
    for image, record in zip(images, records):
        if record.get('input_image_size_hw', [image.height, image.width]) != [image.height, image.width]:
            raise ValueError('archived image dimensions differ from actual input')
    selected = set(selection['instance_ids'])
    if {r['instance_id'] for r in metadata['legend']} != selected:
        raise ValueError('selected instances and supplied legend differ')
    panels = metadata['render']['panels']
    if {gid for p in panels for gid in p['instance_ids']} != selected:
        raise ValueError('selected instances absent from the actual sandbox rendering')
    extents = {e.extent_id: e for e in sandbox.spatial_extents}
    for panel in panels:
        extent = extents[panel['extent_id']]
        if (extent.coordinate_reference != panel['coordinate_reference']
                or not set(panel['instance_ids']) <= extent.instance_ids()):
            raise ValueError('rendered geometry crosses independent coordinate references')
        actual_views = {v.frame_id for v in extent.viewpoints
                        if v.frame_id in frame_ids and selected.intersection(v.observed_instance_ids)}
        if set(panel['viewpoint_frame_ids']) != actual_views:
            raise ValueError('rendered cameras differ from selected actual observation links')
    serialized = serial_messages(messages, images)
    return dict(status='ok', scope='input_integrity_only_not_geometry_or_QA_quality',
                video_id=sandbox.video_id, sandbox_sha256=state_sha256(sandbox),
                combined_images=len(images), sandbox_images=1, actual_frame_ids=frame_ids,
                confirmed_instance_ids=sorted(selected),
                has_confirmed_3d=bool(selected), has_actual_frames=bool(frame_ids),
                image_pixel_sha256=[pixel_sha256(i) for i in images],
                messages_sha256=hashlib.sha256(json.dumps(serialized, ensure_ascii=False,
                    sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(),
                processor_tokens_measured=False)


def save_joint_input(folder, images, metadata, messages):
    """Write-only evidence archive; never an online memory or input cache."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = ['sandbox.png'] + [f"frame_{r['frame_id']}.png" for r in metadata['image_records'][1:]]
    payloads = {'evidence.json': metadata, 'messages.json': serial_messages(messages, images)}
    if any((folder / name).exists() for name in [*names, *payloads]):
        raise FileExistsError('refusing to replace a saved joint QA input')
    for name, image in zip(names, images):
        image.save(folder / name)
    for name, value in payloads.items():
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
