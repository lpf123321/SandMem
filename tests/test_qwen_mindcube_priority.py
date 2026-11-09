import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_qwen_mindcube_priority import claim,selected


def test_only_one_worker_can_claim_a_reserved_question(tmp_path):
    (tmp_path/'reservation.json').write_text(json.dumps(dict(owner='mindcube_a100_priority')))
    assert claim(tmp_path)
    assert not claim(tmp_path)


def test_dataset_priority_preserves_selected_inputs(tmp_path):
    rows=[dict(id=1,dataset='scannet'),dict(id='among_x',dataset='mindcube',question='unchanged')]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(inputs=rows)))
    assert selected(tmp_path)==[rows[1]]
