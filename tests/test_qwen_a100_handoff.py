import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_qwen_a100_handoff import audit_results,validate_vgent_checkpoint


def put(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))


def test_no_retry_of_incomplete_question(tmp_path):
    put(tmp_path/'manifest.json',dict(inputs=[dict(id=1)]))
    (tmp_path/'question_1').mkdir()
    with pytest.raises(RuntimeError,match='incomplete'):audit_results(tmp_path,True)


def test_successful_qwen_results_are_kept_with_hash(tmp_path):
    put(tmp_path/'manifest.json',dict(inputs=[dict(id=1)]))
    put(tmp_path/'question_1/result.json',dict(question_id=1,status='ok',full_qwen_perception=True,old_mllm_cache_used=False,actual_sandbox_in_model=True))
    receipt=audit_results(tmp_path,True)
    assert receipt['completed']==1 and len(receipt['result_sha256']['1'])==64


def test_vgent_only_pauses_after_closed_video_checkpoint(tmp_path):
    put(tmp_path/'completion.json',dict(status='video_boundary_checkpoint'))
    (tmp_path/'videos/unclosed').mkdir(parents=True)
    with pytest.raises(RuntimeError,match='incomplete'):validate_vgent_checkpoint(tmp_path)
    put(tmp_path/'videos/unclosed/complete.json',dict(answers=[]))
    assert validate_vgent_checkpoint(tmp_path)['status']=='passed'
    put(tmp_path/'completion.json',dict(status='failed'))
    with pytest.raises(RuntimeError,match='successful'):validate_vgent_checkpoint(tmp_path)
