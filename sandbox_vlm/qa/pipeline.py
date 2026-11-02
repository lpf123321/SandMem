"""One selection and one joint answering call in an isolated per-question state."""
from copy import deepcopy

from sandbox_vlm.memory.storage import cpu_storage_report
from sandbox_vlm.qa.artifact import verify_joint_input
from sandbox_vlm.qa.input import answer_messages, prepare_question
from sandbox_vlm.qa.result import ANSWER_SCHEMA, parse_answer
from sandbox_vlm.qa.revisions import verify_match_requests


def run_question(base, question, reader, limits, model, settings, *, evidence_sink=None, backend=None, max_targets=None, citation_policy='implicit', output_constraint='none', retrieval_question=None):
    if backend is not None:
        admission = backend.commit(base)
        if admission['status'] != 'ok':
            raise ValueError('capacity_limited base state; cannot answer')
    working = deepcopy(base)
    images, metadata = prepare_question(working, question, reader, limits,
        max_targets=max_targets, retrieval_question=retrieval_question)
    messages = answer_messages(images, metadata, output_schema=ANSWER_SCHEMA,citation_policy=citation_policy)
    metadata['input_contract'] = verify_joint_input(working, question, images, metadata, messages, limits)
    if evidence_sink is not None:
        evidence_sink(images, metadata, messages)
    storage_before = cpu_storage_report([base, working])
    if output_constraint=='json_unique_citations':
        from sandbox_vlm.qa.grammar import answer_grammar
        generated=model.answer(messages,images,limits,format_constraint=answer_grammar(metadata))
    elif output_constraint=='none':
        generated = model.answer(messages, images, limits)
    else:
        raise ValueError('unsupported output syntax constraint')
    result = dict(original_question=question, evidence=metadata, generation=generated,
                  question_state_storage_before=storage_before, revisions=[], status=generated['status'])
    if generated['status'] == 'generated':
        try:
            answer = parse_answer(generated['raw_output'], metadata, working)
            if not working.retain_unresolved and answer['match_requests']:
                raise ValueError('instances-only QA has no observation-match requests')
            result['parsed_answer'] = answer
            result['revisions'] = (verify_match_requests(working, answer['match_requests'], metadata, settings)
                                   if working.retain_unresolved else [])
            result['status'] = 'ok'
        except ValueError as exc:
            result.update(status='answer_failed', error=str(exc))
    result['question_state_storage_after'] = cpu_storage_report([base, working])
    if backend is not None:
        result['question_backend_commit'] = backend.commit(working)
        if result['question_backend_commit']['status'] != 'ok':
            result['status'] = 'capacity_limited'
            result['revision_state_committed'] = False
        else:
            result['revision_state_committed'] = True
        # Read only the accepted bounded state; no oversized state survives.
        working = backend.read_working_copy()
        result['base_backend_restoration'] = backend.commit(base)
        if result['base_backend_restoration']['status'] != 'ok':
            raise ValueError('failed to restore accepted video base state')
    result['working_sandbox_validation'] = working.validate()
    result['answer_generated_before_revisions'] = True
    # No model-triggered second call; no QA state is written back to base.
    return result, working
