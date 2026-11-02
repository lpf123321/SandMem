"""Dependency-exact reuse rules for the isolated Spatial-MLLM migration."""

BACKBONE_DEPENDENT = frozenset({
    'visual_descriptors', 'relevance_scores', 'discovery', 'instance_state',
    'observation_links', 'frame_selection', 'qa_payload', 'render', 'prediction',
})
BACKBONE_INDEPENDENT = frozenset({'decoded_frame', 'frame_mapping', 'sam_image_features', 'sam_mask', 'vggt_geometry'})


def reuse_decision(kind, cached_dependencies, required_dependencies, *, mode):
    """Unknown/missing provenance fails closed; matching kind alone never suffices.

    Dependencies must include content hashes of all arguments, model revision,
    code and preprocessing. For a SAM mask this includes the prompt box and for
    VGGT the entire ordered frame window. Even independent caches are invalid
    if the new discovery/schedule changes those exact inputs.
    """
    if mode not in ('full_spatial', 'fixed_qwen_memory_qa_swap'):
        raise ValueError('explicit migration mode required')
    if kind not in BACKBONE_DEPENDENT | BACKBONE_INDEPENDENT:
        raise ValueError('unclassified cache type')
    required = {'input_sha256', 'model_revision', 'code_sha256', 'processor_sha256'}
    if not required.issubset(cached_dependencies) or not required.issubset(required_dependencies):
        return dict(reuse=False, reason='incomplete_provenance')
    if cached_dependencies != required_dependencies:
        return dict(reuse=False, reason='dependency_mismatch')
    if kind in BACKBONE_DEPENDENT:
        backbone = cached_dependencies.get('mllm_backbone')
        if mode == 'full_spatial' and (not backbone or not backbone.startswith('Diankun/Spatial-MLLM-')):
            return dict(reuse=False, reason='old_or_missing_backbone_provenance')
        if not backbone:
            return dict(reuse=False, reason='missing_backbone_provenance')
    return dict(reuse=True, reason='all_dependencies_exact', cost_must_be_inherited=True,
                result_scope=mode)
