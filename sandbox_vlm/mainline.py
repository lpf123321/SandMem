"""Proposal path: build one video memory before independently answering questions.

Reuses the existing approved geometry, association, memory and QA mechanisms.
There is no conversion from D065 hypotheses, per-task solver, new threshold or
extra model. Model transports remain injectable for explicit CPU contracts.
"""
from sandbox_vlm.alignment.association import MatchSettings
from sandbox_vlm.pipeline import build_video
from sandbox_vlm.qa.artifact import state_sha256
from sandbox_vlm.qa.budget import QABudget
from sandbox_vlm.qa.frames import PreparedFrameReader
from sandbox_vlm.qa.pipeline import run_question


def run_prepared_questions(sample_root, manifest, config, questions, *, backend,
                           discovery, tracker, geometry, answerer, audit_sink=None,
                           mapping_sink=None, evidence_sink=None, result_sink=None):
    """No benchmark answers or historical prediction files are accepted here.

    Sinks are write-only observers. One question's working state is discarded
    before the next one; the complete base and overlap stay in the backend.
    """
    if (not questions or any(set(q) != {'id', 'question'} or not isinstance(q['question'], str)
                            or not q['question'].strip() for q in questions)
            or len({q['id'] for q in questions}) != len(questions)):
        raise ValueError('unique question IDs and nonempty original questions required')
    if config['perception']['mllm'].get('question_focus', False):
        raise ValueError('Proposal base mapping must be independent of questions')
    for key in ('model_id', 'revision', 'dtype'):
        if config['qa']['mllm'][key] != config['perception']['mllm'][key]:
            raise ValueError('perception and QA must use the same frozen backbone')
    # Deliberately pass no questions or question-derived selections to mapping.
    base, mapping = build_video(sample_root, manifest, config, backend=backend,
        discovery=discovery, tracker=tracker, geometry=geometry, audit_sink=audit_sink)
    report = dict(path='proposal_four_module', question_conditioned_mapping=False,
        mapping=mapping, questions=[], base_state_unchanged=True,
        geometry_acceptance='existing_overlap_consensus_not_long_video_validation')
    if mapping_sink is not None:
        mapping_sink(base, mapping)
    if mapping['status']=='capacity_limited':
        report['status']='capacity_limited'
        return base, report
    baseline = state_sha256(base)
    report['base_sandbox_sha256']=baseline
    reader = PreparedFrameReader(sample_root, manifest, video_id=base.video_id)
    for index, item in enumerate(questions):
        def evidence(images, metadata, messages):
            if evidence_sink is not None:
                evidence_sink(index, item, images, metadata, messages)
        result, working = run_question(base, item['question'], reader,
            QABudget(**config['qa']['budget']), answerer,
            MatchSettings(**config['alignment']['match']), backend=backend,
            evidence_sink=evidence,
            max_targets=config['qa'].get('selection', {}).get('max_targets'),
            citation_policy=config['qa']['mllm'].get('citation_policy', 'implicit'),
            output_constraint=config['qa']['mllm'].get('output_constraint', 'none'))
        if state_sha256(base)!=baseline or state_sha256(backend.read_working_copy())!=baseline:
            raise ValueError('question revisions leaked into reusable video memory')
        if result_sink is not None:
            result_sink(index, item, result, working)
        report['questions'].append(dict(id=item['id'], status=result['status'],
            input_contract=result['evidence']['input_contract'],
            model_calls=result['generation'].get('model_calls')))
        del working, result
    report['status']='ok' if mapping['status']=='ok' and all(q['status']=='ok' for q in report['questions']) else 'incomplete'
    return base, report
