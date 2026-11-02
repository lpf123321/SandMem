"""D113: real frames stay contiguous; supplemental sandbox follows them."""
from copy import deepcopy


def evidence_first(messages, snapshot):
    result, snap = deepcopy(messages), deepcopy(snapshot)
    content = result[0]['content']
    n = len(snap['paths']) - 1
    frames = content[5:5 + 2*n]
    assert len(frames) == 2*n
    for i in range(n):
        assert frames[2*i]['type'] == 'text'
        assert frames[2*i]['text'].startswith('Actual source frame ID ')
        assert frames[2*i+1]['evidence_image_index'] == i+1
    assert content[2]['evidence_image_index'] == 0
    # No block's text or pixel identity changes; only block/image ordering does.
    result[0]['content'] = [content[0], content[4], *frames,
                            *content[1:4], *content[5 + 2*n:]]
    for item in result[0]['content']:
        if item['type'] == 'image':
            old = item['evidence_image_index']
            item['evidence_image_index'] = n if old == 0 else old-1
    for key in ['paths', 'pixel_sha256', 'grids']:
        snap[key] = snap[key][1:] + snap[key][:1]
    return result, snap
