"""D052 selects evidence from the unmodified D043 base, then revises a QA copy."""
from sandbox_vlm.memory.coverage import select_answer_frames
from sandbox_vlm.qa.coverage_evidence import temporal_answer_frames


def select_base_frames(base, ledger, discovered_bank, *, sparse, config):
    settings = config['coverage']
    maximum = config['visual']['sparse_maximum_frames'] if sparse else settings['maximum_answer_images']
    if settings.get('answer_selection') == 'temporal_coverage':
        return list(discovered_bank) if sparse else temporal_answer_frames(base, ledger, maximum=maximum)
    return select_answer_frames(base, ledger, maximum=maximum)


def charge_question_copy(auxiliary, answer_state, short_track_state):
    # This is resident inference state, not a path to an offline artifact.
    return dict(auxiliary, answer_working_sandbox=answer_state.to_dict(),
                short_track=short_track_state, association_application='question_local_no_base_writeback')
