"""Independent CPU audit for a completed continuous-model development gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sandbox_vlm.perception.discovery import discovery_prompt,parse_discovery
from sandbox_vlm.qa.artifact import pixel_sha256
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    root=args.root.resolve();run=root/'run'
    def read(name):return json.loads((run/name).read_text())
    manifest=json.loads((root/'source_manifest.json').read_text())
    assert all(hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['files'])
    plan=read('input_plan.json');completion=read('completion.json')
    ids=plan['frame_ids'];rows=[read(f'frame_{n:05d}.json') for n in range(len(ids))]
    assert plan['full_video'] and not completion['prefix_only']
    assert [r['frame_id'] for r in rows]==ids
    assert len(list(run.glob('frame_*.json')))==len(ids)
    config=plan['config'];K=config['geometry']['cache_frames_including_current']
    discoveries=[];closed_records=0;boundary_seconds=0.
    for index,row in enumerate(rows):
        assert row['geometry_cost']['retained_frame_ids']==ids[max(0,index-K+1):index+1]
        assert row['geometry_cost']['cache_after_bytes']<=config['geometry']['cache_limit_bytes']
        assert row['memory']['capacity']['bytes']<=config['memory_limit_bytes']
        d=row['discovery']
        if d is None:continue
        assert d['status']=='ok' and not d['reached_output_limit']
        assert d['prompt']==discovery_prompt(row['frame_id'])
        assert len(d['output_token_ids'])==d['output_tokens']<=config['execution']['discovery_output_tokens']
        expected=json.loads(json.dumps([x.to_dict() for x in parse_discovery(d['raw_output'],row['frame_id'],tuple(d['image_size_hw']))]))
        unique=[]
        for detection in expected:
            if detection not in unique:unique.append(detection)
        assert unique==d['detections']
        contract=d['output_contract']
        assert not contract['field_values_changed'] and not contract['raw_output_modified']
        assert contract['original_duplicate_guard_retained']
        assert contract['removed_identical_records']==len(expected)-len(unique)
        closed_records+=contract['removed_identical_records'];boundary_seconds+=contract['check_seconds']
        discoveries.append(row['frame_id'])
    assert discoveries==[ids[n] for n in plan['discovery_indices']]
    attempts=[json.loads(line) for line in (run/'attempts.jsonl').read_text().splitlines()]
    counts=dict(Counter(r['kind'] for r in attempts))
    assert counts==completion['calls']
    assert counts['geometry']==counts['sam_encoder']==len(ids)
    assert counts['discovery']==len(discoveries)
    assert counts['qa'] in [1,2]
    prepared=read('prepared.json');joint=read('joint_input.json');generation=read('joint_generation.json')
    selected=prepared['selection']['frame_ids']
    assert prepared['question']==plan['question']
    assert len(selected)<=config['qa']['real_frame_budget']
    assert selected==sorted(set(selected)) and set(selected)<=set(ids)
    assert len(joint['images'])==len(selected)+1
    assert generation['budget']['status']=='ok' and generation['model_calls']==1
    assert len(generation['image_grid_thw'])==len(joint['images'])
    panels=prepared['rendering']['panels']
    assert sorted(fid for panel in panels for fid in panel['trajectory_frame_ids'])==ids
    for record in joint['images']:
        with Image.open(record['path']) as im:assert pixel_sha256(im.convert('RGB'))==record['pixel_sha256']
    # Independently bind each provided real image to its declared source ID.
    for fid, record in zip(selected, joint['images'][1:]):
        with Image.open(run/'frames'/f'{fid:08d}.png') as source:
            expected_image=source.convert('RGB')
            expected_image.thumbnail((config['qa']['real_image_side'],)*2)
            assert pixel_sha256(expected_image)==record['pixel_sha256']
    prediction=completion['answer'].get('prediction')
    qa_ok=completion['answer']['status']=='ok' and prediction in ['A','B','C','D']
    receipt=dict(status='passed_offline_development_gate' if qa_ok else 'blocked_invalid_QA',
        gate_directory=root.name,source_hashes_verified=len(manifest['files']),full_sampled_frames=len(ids),
        continuous_geometry_and_SAM_complete=True,FIFO_complete_frames_verified=True,
        full_camera_trajectory_in_render=True,actual_3D_and_real_frames_in_QA=True,
        selected_frame_ids=selected,removed_exact_repeated_records=closed_records,
        boundary_check_seconds=boundary_seconds,answer_contract_passed=qa_ok,
        prediction=prediction,calls=counts,wall_seconds=completion['wall_seconds'],
        memory_bytes=completion['structure_bytes'],device=completion['device'],
        score_not_computed=True,formal_all_task_evaluation_ready=False,
        formal_streaming_adapter_ready=False)
    with (root/'gate_audit_source_binding.json').open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    if not qa_ok:raise SystemExit(2)

if __name__=='__main__':main()
