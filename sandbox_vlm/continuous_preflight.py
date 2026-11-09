"""CPU-only start gate. Engineering prefixes cannot masquerade as evaluations."""
import math


def sample_plan(source_fps, source_frames, sampling_fps, prefix_limit=None):
    if (not math.isfinite(source_fps) or not math.isfinite(sampling_fps)
            or not 0 < sampling_fps <= source_fps or type(source_frames) is not int or source_frames < 1
            or prefix_limit is not None and (type(prefix_limit) is not int or prefix_limit < 1)):
        raise ValueError('invalid source/sampling schedule')
    complete = sorted({round(i * source_fps / sampling_fps)
        for i in range(math.ceil(source_frames / source_fps * sampling_fps))})
    complete = [fid for fid in complete if fid < source_frames]
    ids = complete if prefix_limit is None else complete[:prefix_limit]
    return dict(frame_ids=ids, full_video=ids == complete, full_sampled_frames=len(complete))


def validate_start(config, *, plan, purpose, protocol, answer_format, input_profile=None):
    problems = []
    if protocol != 'offline':
        problems.append('streaming_benchmark_adapter_not_validated: end-only runner must not read future frames')
    if answer_format not in ('choice', 'number', 'text'):
        problems.append('explicit_answer_contract_required')
    if config['qa']['budget']['max_images'] < config['qa']['real_frame_budget'] + 1:
        problems.append('combined_image_budget_must_include_3D_diagram')
    if purpose != 'wiring':
        if not plan['full_video']:
            problems.append('truncated_prefix_cannot_answer_full_video_evaluation_question')
        if config.get('configuration_role') != 'full_video_development':
            problems.append('wiring_only_8_discovery_configuration_is_not_an_evaluation_profile')
        if input_profile is None:
            problems.append('explicit_inherited_input_profile_required: frame_count/density/image_side/output_limit')
    if purpose == 'evaluation':
        problems.append('formal_benchmark_adapter_and_frozen_all_task_acceptance_gate_not_complete')
    return dict(status='blocked' if problems else 'passed', problems=problems,
                purpose=purpose, protocol=protocol, full_video=plan['full_video'],
                sampled_frames=len(plan['frame_ids']), formal_evaluation_ready=False)


def apply_input_profile(config, profile):
    """Only declared input-derived settings. No answer, task label or prediction."""
    required = {'sparse', 'real_frame_budget', 'real_image_side', 'max_output_tokens', 'origin'}
    if set(profile) != required or type(profile['sparse']) is not bool:
        raise ValueError('invalid whitelisted input profile')
    if (type(profile['real_frame_budget']) is not int or not 1 <= profile['real_frame_budget'] <= 32
            or type(profile['real_image_side']) is not int or not 1 <= profile['real_image_side'] <= 768
            or profile['max_output_tokens'] not in (64, 256)
            or not isinstance(profile['origin'], str) or not profile['origin']):
        raise ValueError('invalid inherited input budget')
    if profile['max_output_tokens'] != (256 if profile['sparse'] else 64):
        raise ValueError('inherited output limit disagrees with input-density branch')
    from copy import deepcopy
    result = deepcopy(config)
    result['qa']['real_frame_budget'] = profile['real_frame_budget']
    result['qa']['real_image_side'] = profile['real_image_side']
    result['qa']['budget']['max_images'] = profile['real_frame_budget'] + 1
    result['qa']['budget']['max_output_tokens'] = profile['max_output_tokens']
    return result
