"""Controlled evidence removal and question-linked observation presentation.

No task-specific parsing, new identity decision, or perception model is used.
"""
from copy import deepcopy
import json

from sandbox_vlm.qa.complete_sandbox import complete_messages
from sandbox_vlm.qa.prompt import messages_for
from sandbox_vlm.qa.render import SPACE_NOTICE
from sandbox_vlm.qa.selection import terms


def diagnostic_messages(snapshot, images, diagram, arm):
    """The four cells share D105 question, pixels, short answer, and decoder."""
    if arm == 'both':
        return complete_messages(snapshot, images, diagram, snapshot['payload'])
    base = messages_for(snapshot['question'], images, snapshot['frame_ids'], 'question_last')
    if arm == 'frames_only':
        return base, list(images)
    content = base[0]['content']
    if arm == 'catalog_only':
        header = ('Stored instance records are estimated, not verified physical identities. '
                  'Image regions and local SAM links refer to actual observations; local '
                  'SAM links are not independently verified global identities.\n')
        # Keep the entire original catalog, including provenance, verbatim.
        content[0]['text'] += '\n' + header
        content[1:1] = [dict(type='text', text='Complete sandbox records and observation evidence (JSON data):\n'
            + json.dumps(snapshot['payload'], ensure_ascii=False, separators=(',', ':')))]
        return base, list(images)
    if arm != 'diagram_only':
        raise ValueError('unknown diagnostic arm')
    content[0]['text'] += ('\nThe supplemental diagram shows estimated local VGGT boxes, '
        'record IDs and actual camera frame IDs. Arbitrary scale; an unplotted object is '
        'not evidence of absence.\n' + SPACE_NOTICE)
    content[1:1] = [dict(type='text', text='Estimated 3D sandbox diagram:'), dict(type='image', image=diagram)]
    return base, [diagram, *images]


# Protocol/function words only; no object names, task classification, or option parser.
PROTOCOL_WORDS = terms('these frames video following correctly represents order appeared '
    'options answer letter given choices directly different there using single word phrase')


def linked_selection(original_question, payload):
    """Use the original input field, before adapter-added choices/instructions."""
    query = terms(original_question) - PROTOCOL_WORDS
    records = payload['records']
    anchors = {r[0] for r in records if query & terms(r[1])}
    if not anchors:
        anchors = {r[0] for r in records if query & terms(r[2])}
    fallback = not anchors
    if fallback:
        anchors = {r[0] for r in records}
    actual_frames = {fid for r in records if r[0] in anchors for fid in r[3]}
    context = {r[0] for r in payload['provided_frame_regions'] if r[1] in actual_frames}
    return dict(anchor_ids=sorted(anchors), relevant_ids=sorted(anchors | context),
                actual_anchor_frames=sorted(actual_frames), lexical_fallback=fallback)


def bound_messages(snapshot, images, diagram, selection):
    """Bind relevant native records to their actual source image, without deletion.

    A record's full text appears once at its first provided actual observation;
    subsequent observations use its stable ID. All other original native fields
    remain verbatim in an explicitly supplemental catalog. Source pixels and
    their ordering remain unchanged.
    """
    payload = snapshot['payload']
    base, joint = complete_messages(snapshot, images, diagram, payload)
    relevant = set(selection['relevant_ids'])
    # Do not change already wholly relevant inputs or their predictions.
    if relevant == {r[0] for r in payload['records']}:
        return base, joint
    content = base[0]['content']
    content[3]['text'] = ('Supplemental complete stored catalog, including records from other '
        'observations. Each actual-frame block below identifies its linked records. '
        'A background object record does not assert that another object appeared there.\n'
        + json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    content[4]['text'] = 'Actual video frames in chronological order, with query-linked observation evidence:'
    by_id = {r[0]: r for r in payload['records']}
    regions = payload['provided_frame_regions']
    emitted = set()
    for item in content[5:]:
        if item['type'] != 'text' or not item['text'].startswith('Actual source frame ID '):
            continue
        fid = int(item['text'].split('ID ')[1].rstrip(':'))
        actual = sorted({r[0] for r in regions if r[1] == fid and r[0] in relevant})
        first = [by_id[gid] for gid in actual if gid not in emitted]
        emitted.update(actual)
        item['text'] += '\nQuery-linked actual record IDs: ' + json.dumps(actual)
        if first:
            item['text'] += '\nNative records [id,name,complete description,actual frame IDs]: ' + json.dumps(
                first, ensure_ascii=False, separators=(',', ':'))
    return base, joint


def selected_catalog(payload, selection):
    """Proposed D109: retain complete native fields of retrieved records only.

    Removing unrelated records from actual QA is a user-visible change to the
    previous full-catalog requirement. This helper does not authorize execution.
    """
    keep = set(selection['relevant_ids'])
    result = deepcopy(payload)
    result['records'] = [r for r in result['records'] if r[0] in keep]
    result['provided_frame_regions'] = [r for r in result['provided_frame_regions'] if r[0] in keep]
    result['plotted_ids'] = [gid for gid in result['plotted_ids'] if gid in keep]
    result['unplotted_ids'] = [gid for gid in result['unplotted_ids'] if gid in keep]
    result['box_observation_sources']['rows'] = [r for r in result['box_observation_sources']['rows'] if r[0] in keep]
    result['local_SAM_correspondences']['rows'] = [r for r in result['local_SAM_correspondences']['rows']
                                               if r[0] in keep and r[1] in keep]
    if {r[0] for r in result['records']} != keep:
        raise ValueError('retrieval contains unknown instance')
    return result
