"""Read-only, answer-free D043 perception inputs for the approved D053 control.

Snapshots are declared experimental inputs, not an online memory paging backend.
Only the current query's complete base and auxiliary state are loaded.
"""
import hashlib
import json
from pathlib import Path

from sandbox_vlm.memory.coverage import CoverageSandbox
from sandbox_vlm.qa.question_local import select_base_frames


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_query(record, item, config):
    values = {}
    for kind in ('base', 'auxiliary'):
        path = Path(item[kind+'_path'])
        # Only explicitly whitelisted perception files can enter inference.
        expected = 'visual_sandbox.json' if kind == 'base' else 'ledger.json'
        if path.name != expected or file_sha(path) != item[kind+'_sha256']:
            raise ValueError('frozen perception file changed or is not an allowed input')
        values[kind] = json.loads(path.read_text())
    state = CoverageSandbox.from_dict(values['base'])
    auxiliary = values['auxiliary']
    allowed = {'gated_frame_ledger', 'discovered_frame_bank', 'last_shortlist_state',
               'prefer_full_next_batch', 'current_watermark', 'frame_mapping',
               'answer_frame_ids', 'question_key'}
    if set(auxiliary) != allowed:
        raise ValueError('unexpected frozen auxiliary fields')
    query = item['query_frame_id']
    if auxiliary['current_watermark'] != query or state.question_key != auxiliary['question_key']:
        raise ValueError('frozen question or prefix mismatch')
    if state.question_key != CoverageSandbox(state.video_id, record['original_question']).question_key:
        raise ValueError('frozen input question changed')
    if any(int(fid) > query for fid in state.viewpoints):
        raise ValueError('future frozen observation')
    ledger = {int(fid): entry for fid, entry in auxiliary['gated_frame_ledger'].items()}
    if not ledger or max(ledger) > query or min(ledger) < 0:
        raise ValueError('future or empty frozen coverage ledger')
    bank = auxiliary['discovered_frame_bank']
    if any(fid not in ledger or fid > query for fid in bank):
        raise ValueError('future or missing discovery input')
    sparse = sum(entry['margin'] > 0 for entry in ledger.values()) / len(ledger) <= config['coverage']['discovery']['sparse_fraction']
    chosen = select_base_frames(state, ledger, bank, sparse=sparse, config=config)
    if chosen != auxiliary['answer_frame_ids']:
        raise ValueError('frozen D043 evidence selection does not reproduce')
    return state, auxiliary, chosen, sparse
