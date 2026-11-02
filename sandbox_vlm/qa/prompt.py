"""Unchanged D053/D063 QA templates, extracted from the development scripts."""
def frame_messages(question, images, ids):
    content = [dict(type='text', text=(
        'The following actual video frames are in chronological order. Views may revisit '
        'the same physical object. Use the visible evidence to answer the original question. '
        'Return only the requested short answer, without explanation.\nOriginal question:\n' + question))]
    for image, fid in zip(images, ids):
        content += [dict(type='text', text=f'Actual source frame ID {fid}:'), dict(type='image', image=image)]
    return [dict(role='user', content=content)]


def messages_for(question, images, ids, arm):
    messages = frame_messages(question, images, ids)
    content = messages[0]['content']
    # Preserve the original question verbatim, move it after the visual input.
    content[0]['text'] = (
        'The following actual video frames are in chronological order. Views may revisit '
        'the same physical object. Use the visible evidence to answer the original question. '
        'Return only the requested short answer, without explanation.')
    content.append(dict(type='text', text='Original question (verbatim):\n' + question))
    if arm == 'brief_evidence':
        content[0]['text'] = (
            'The following actual video frames are in chronological order. Views may revisit '
            'the same physical object. Inspect the relevant evidence across all provided frames. '
            'Write brief evidence notes in chronological order, at most 100 words; distinguish '
            'observed facts from uncertainty. Then give the requested short answer on a separate '
            'final line exactly as FINAL_ANSWER: <answer>.')
        content.append(dict(type='text', text=(
            'Give brief evidence notes first, then the final line FINAL_ANSWER: <answer>.')))
    return messages

