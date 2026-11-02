"""D065 input replay with exactly one supplemental spatial image block."""
import json

from sandbox_vlm.qa.prompt import messages_for
from sandbox_vlm.qa.render import SPACE_NOTICE

MAP_NOTICE = ('Supplemental estimated 3D sandbox: arbitrary scale, displayed local reference only, '
              'incomplete inventory.\n' + SPACE_NOTICE)


def original_messages(snapshot, images, memory_notice):
    messages = messages_for(snapshot['question'], images, snapshot['frame_ids'],
        'brief_evidence' if snapshot['sparse'] else 'question_last')
    if snapshot['compact_memory'] is not None:
        messages[0]['content'][0]['text'] += memory_notice + json.dumps(
            snapshot['compact_memory'], separators=(',', ':'))
    return messages


def add_map(messages, original_images, diagram):
    result = [dict(role=message['role'],content=[dict(item)for item in message['content']])
              for message in messages]
    source_index = 0
    for message in result:
        for item in message['content']:
            if item['type'] == 'image':
                item['image'] = original_images[source_index]
                source_index += 1
    if source_index != len(original_images):
        raise ValueError('original image cardinality differs')
    result[0]['content'][1:1] = [dict(type='text', text=MAP_NOTICE), dict(type='image', image=diagram)]
    return result, [diagram, *original_images]


def verify_only_map(base, augmented, original_images, joint_images):
    if len(joint_images) != len(original_images)+1 or any(
            original is not actual for original, actual in zip(original_images, joint_images[1:])):
        raise ValueError('source images changed')
    restored = []
    for index, message in enumerate(augmented):
        content = list(message['content'])
        if index == 0:
            if content[1] != dict(type='text', text=MAP_NOTICE) or content[2]['type'] != 'image':
                raise ValueError('supplemental map block differs')
            del content[1:3]
        restored.append(dict(role=message['role'], content=content))
    def protected(messages):
        return [dict(role=m['role'],content=[dict(type='image',object_identity=id(c['image']))
            if c['type']=='image'else dict(c)for c in m['content']])for m in messages]
    if protected(base) != protected(restored):
        raise ValueError('D065 text, source images or order changed')


def indexed_messages(messages, images, pixel_hashes):
    """Serialize once with trusted archive hashes or hashes of actual loaded pixels."""
    if len(images)!=len(pixel_hashes):raise ValueError('pixel hash cardinality')
    result=[];index=0
    for message in messages:
        content=[]
        for item in message['content']:
            if item['type']=='image':
                if index>=len(images)or item['image']is not images[index]:raise ValueError('image order differs')
                content.append(dict(type='image',evidence_image_index=index,pixel_sha256=pixel_hashes[index]));index+=1
            else:content.append(dict(item))
        result.append(dict(role=message['role'],content=content))
    if index!=len(images):raise ValueError('unreferenced source image')
    return result


def text_tokens(tokenizer, messages, number_of_images, chat_template=None):
    kwargs = dict(tokenize=True, add_generation_prompt=True)
    if chat_template is not None:
        kwargs['chat_template'] = chat_template
    ids = tokenizer.apply_chat_template(messages, **kwargs)
    image_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    if ids.count(image_id) != number_of_images:
        raise ValueError('unexpected image placeholders')
    return len(ids)-number_of_images
