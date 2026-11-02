"""Actual-region QA input with explicit question provenance and existing 3D.

This is input wiring only: no new identity edges, geometry, temporal quotas,
model calls or expansion of the 24-entry legend budget.
"""
from PIL import Image

from sandbox_vlm.memory.identity_evidence import region_key
from sandbox_vlm.qa.artifact import serial_messages
from sandbox_vlm.qa.render import SPACE_NOTICE
from sandbox_vlm.qa.result import ANSWER_SCHEMA
from sandbox_vlm.qa.selection import terms
from sandbox_vlm.qa.spatial_restore import restore_spatial_input


def prepare_identity_input(base, regions, groups, relations, frame_ids, question,
                           candidate, source_images, source_records, *, geometry,
                           decisions, model_question=None, structure_observer=None):
    """Retrieve using the dataset question; retain the model's answer wrapper.

    Source images are already resized actual frames. Every record and omission
    is audited, without treating a track or local identity as a physical count.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError('explicit original retrieval question required')
    if len(frame_ids) != len(set(frame_ids)) or len(source_images) != len(frame_ids):
        raise ValueError('unique actual source frames required')
    if [r['frame_id'] for r in source_records] != frame_ids:
        raise ValueError('source image records differ from selection')
    query = terms(question)
    by_key = {r['region_key']: r for r in regions}
    relevant = [g for g in groups if any(query & terms(by_key[k]['name']) for k in g)] if candidate else []
    first_identity_id = max((r['instance_id'] for r in base['records']), default=-1) + 1
    identity_for = {k: first_identity_id+i for i, g in enumerate(relevant) for k in g}
    display = {}
    for fid in frame_ids:
        for link in base['viewpoints'].get(str(fid), []):
            key = region_key(link['evidence'])
            gid = identity_for.get(key, link['instance_id'])
            record = base['records'][link['instance_id']]
            if gid not in display:
                display[gid] = dict(instance_id=gid, name=record['name'],
                    description=record['description'][:64], coordinate_reference=None,
                    identity_status=('rule_supported_local_identity' if key in identity_for
                                     else 'observation_identity_unknown'),
                    selected_actual_frame_ids=[], identity_evidence=[])
            display[gid]['selected_actual_frame_ids'].append(fid)
    for i, group in enumerate(relevant):
        gid = first_identity_id+i
        if not {by_key[k]['region']['frame_id'] for k in group} <= set(frame_ids):
            raise ValueError('relevant identity lacks actual QA endpoints')
        if gid not in display:
            raise ValueError('identity endpoints missing from original actual links')
        display[gid]['identity_evidence'] = [dict(
            source_frame=by_key[r['region_keys'][0]]['region']['frame_id'],
            target_frame=by_key[r['region_keys'][1]]['region']['frame_id'],
            source_region=by_key[r['region_keys'][0]]['region']['bbox_xyxy'],
            target_region=by_key[r['region_keys'][1]]['region']['bbox_xyxy'],
            basis='independent_SAM_shared_pixels_and_unique_actual_endpoints')
            for r in relations if r['status'] == 'supported_local_relation'
            and set(r['region_keys']) <= set(group)]
    ranked = sorted(display.values(), key=lambda r: (
        not bool(r['identity_evidence']), -len(query & terms(r['name']+' '+r['description'])), r['instance_id']))
    legend = ranked[:24]
    if any(r['identity_evidence'] for r in ranked[24:]):
        raise ValueError('identity evidence omitted by target budget')
    ranking = []
    for index, row in enumerate(ranked):
        row['selected_actual_frame_ids'] = sorted(set(row['selected_actual_frame_ids']))
        matches = sorted(query & terms(row['name']+' '+row['description']))
        ranking.append(dict(instance_id=row['instance_id'], name=row['name'],
            matched_terms=matches, score=len(matches), rank=index+1,
            selected_actual_frame_ids=row['selected_actual_frame_ids'],
            displayed=index < 24, omission_reason=None if index < 24 else 'max_structure_targets_24'))
    omitted = [r['instance_id'] for r in ranked[24:]]
    omitted_relevant = sum(bool(r['matched_terms']) for r in ranking[24:])
    meta = dict(original_question=model_question if model_question is not None else question,
        retrieval_question=question,
        question_provenance=dict(retrieval='dataset.original_question', model='dataset.model_question'),
        selection=dict(question=question, instance_ids=[r['instance_id'] for r in legend],
                       frame_ids=frame_ids, lexical_fallback=not any(r['score'] for r in ranking),
                       uncovered_targets=[]),
        legend=legend, unresolved_notes=[], memory_has_unresolved_observations=True,
        memory_may_be_incomplete=True, scale_is_metric=False, space_notice=SPACE_NOTICE,
        image_records=[dict(kind='sandbox', input_image_size_hw=[576, 768]), *source_records],
        omission_audit=dict(query_terms=sorted(query), ranked_supported_records=ranking,
                            omitted_supported_ids=omitted, omitted_relevant_count=omitted_relevant,
                            track_coverage_is_not_physical_object_coverage=True),
        model_selection_limitations=dict(max_structure_targets=24,
            omitted_displayed_targets=len(omitted), omitted_relevant_count=omitted_relevant,
            notice='I links support only the displayed checked actual regions, not all parent track '
                   'observations. Other O identities remain unknown. The legend is a selected subset; '
                   'omitted records may be relevant and are not absent or confirmed distinct objects.'))
    # Only a slot placeholder; it is always replaced by the original 3D renderer.
    images = [Image.new('RGB', (768, 576), 'white'), *source_images]
    working = [display, ranked, ranking, meta, identity_for, by_key]
    def observe(label, values):
        if structure_observer is not None:
            structure_observer(label, [working, values])
    observe('legend_selection', [])
    joint, meta, messages = restore_spatial_input(images, meta, geometry, base, decisions,
        output_schema=ANSWER_SCHEMA, structure_observer=observe)
    verify_identity_transport(joint, meta, messages)
    return joint, meta, messages


def verify_identity_transport(images, metadata, messages):
    """Fail before generation if a directory silently replaces spatial input."""
    render = metadata['render']
    if (render.get('geometry_source') != 'frozen geometric sandbox'
            or render.get('panel_layout_has_physical_meaning') is not False
            or metadata.get('space_notice') != SPACE_NOTICE):
        raise ValueError('actual identity QA requires the existing spatial renderer')
    legend = {r['instance_id']: r for r in metadata['legend']}
    plotted = {gid for p in render['panels'] for gid in p['instance_ids']}
    boxed = {gid for gid, row in legend.items() if 'bbox_3d' in row}
    if plotted != boxed or bool(plotted) != render['contains_3D_boxes']:
        raise ValueError('spatial diagram and legend boxes differ')
    frames = metadata['selection']['frame_ids']
    if [r['frame_id'] for r in metadata['image_records'][1:]] != frames:
        raise ValueError('actual frame order differs from selection')
    if len(metadata['image_records']) != len(images):
        raise ValueError('actual image count differs from metadata')
    for panel in render['panels']:
        if not set(panel['viewpoint_frame_ids']) <= set(frames):
            raise ValueError('rendered camera lacks a supplied actual frame')
        if any(legend[gid]['bbox_3d']['frame'] != panel['coordinate_reference']
               for gid in panel['instance_ids']):
            raise ValueError('mixed spatial coordinate references')
    for row in legend.values():
        if not set(row['selected_actual_frame_ids']) <= set(frames):
            raise ValueError('legend claims an absent source frame')
        for edge in row.get('identity_evidence', []):
            if not {edge['source_frame'], edge['target_frame']} <= set(frames):
                raise ValueError('identity proof lacks actual source endpoints')
    # These are the same image objects passed to the processor, not filenames.
    serial_messages(messages, images)
    if messages[-1]['content'][-1]['text'] != 'Original question (verbatim):\n'+metadata['original_question']:
        raise ValueError('model question changed during input transport')
    return dict(status='passed_input_integrity_only', plotted_instance_ids=sorted(plotted),
                actual_frame_ids=frames, model_calls=0)
