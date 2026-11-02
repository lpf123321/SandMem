"""One post-completion official full-denominator scoring; no inference access to GT."""
import argparse
from collections import defaultdict
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_evidence_binding import read,write,sha


def score(out):
    import pyarrow.parquet as pq
    from sandbox_vlm.evaluation.vsi_super import official_module
    from sandbox_vlm.evaluation.complete_denominator import score_record
    m=read(out/'manifest.json');progress=read(out/'progress.json')
    assert progress['status']=='completed'and progress['records_finished']==700
    if(out/'scores.json').exists():raise FileExistsError('one post-completion scoring')
    project=Path(m['baseline']).parents[2];tables={};modules={};rows=[];all_reports=[]
    for provenance in m['official_scoring_provenance'].values():
        assert sha(project/'vendor/cambrian-s-official'/provenance['path'])==provenance['sha256']
    old=read(Path(m['baseline'])/'full_denominator_diagnostic.json')
    reference={(r['sample_id'],r['ordinal']):r['score_percent']for r in old['queries']}
    for meta in m['records']:
        rid=meta['record_index'];report=read(out/f'sample_{rid:02d}/record_report.json');record=read(meta['path'])
        actual={r['query_frame_id']:r for r in report['queries']};predictions=[actual.get(q['query_frame_id'],dict(status='missing',prediction=None))for q in meta['queries']]
        task=meta['task'];key=meta['variant'],meta['duration']
        if task not in modules:modules[task]=official_module(project/'vendor/cambrian-s-official',task)[0]
        if key not in tables:tables[key]=pq.read_table(project/'work/protocol/vsi_super'/key[0]/f'test_{key[1]}.parquet').to_pylist()
        doc=tables[key][int(meta['sample_id'].split(':')[1])];assert doc['video_path']==meta['sample_id'].split(':',2)[2]
        scored=score_record(task,doc,[p['prediction']for p in predictions],[p['status']for p in predictions],modules[task])
        assert len(scored)==len(meta['queries'])
        for query,p,s in zip(meta['queries'],predictions,scored):
            rows.append(dict(record=rid,sample_id=meta['sample_id'],query_frame_id=query['query_frame_id'],task=task,duration=meta['duration'],
                partition=meta['partition'],original_partition=meta['original_partition'],expanded_partition=meta['expanded_partition'],
                reused_development=meta['reused_development'],prediction=p['prediction'],D065_percent=reference[meta['sample_id'],s['ordinal']],
                two_pass=p.get('two_pass',False),**s))
        all_reports.append(dict(record=rid,failed_queries=sum(p['status']!='ok'for p in predictions)))
    assert len(rows)==2500 and sum(r['task']=='recall'for r in rows)==300
    def aggregate(xs):return dict(queries=len(xs),score_percent=sum(r['score_percent']for r in xs)/len(xs),
        D065_percent=sum(r['D065_percent']for r in xs)/len(xs),invalid=sum(not r['valid']for r in xs),
        two_pass_queries=sum(r['two_pass']for r in xs),reused_queries=sum(r['reused_development']for r in xs))
    summary={t:aggregate([r for r in rows if r['task']==t])for t in ['recall','count','count_streaming']}
    groups=[dict(task=t,duration=d,**aggregate([r for r in rows if(r['task'],r['duration'])==(t,d)]))
        for t,d in sorted({(r['task'],r['duration'])for r in rows})]
    partitions={}
    for field in ['partition','original_partition','expanded_partition']:
        partitions[field]={}
        for value in sorted({r[field]for r in rows}):
            selected=[r for r in rows if r[field]==value]
            partitions[field][value]={t:aggregate([r for r in selected if r['task']==t])for t in sorted({r['task']for r in selected})}
    input_checks=[];actual_QA=0;visual=text=outputs=0
    for meta in m['records']:
        rid=meta['record_index'];record_report=read(out/f'sample_{rid:02d}/record_report.json')
        for r in record_report['queries']:
            q=out/f'sample_{rid:02d}'/f'query_{r["query_frame_id"]:06d}'
            if r['status']!='ok':continue
            s=read(q/'input.json');messages=read(q/'messages.json');g=read(q/'generation.json')
            refs=[c for msg in messages for c in msg['content']if c['type']=='image']
            assert s['has_map']and refs[0]['pixel_sha256']==s['pixel_sha256'][0]
            assert s['budget']==g['budget']and s['grids']==g['image_grid_thw']and s['generation_kwargs']==g['generation_kwargs']
            assert [c['pixel_sha256']for c in refs]==s['pixel_sha256']
            if not r['reused_development']:
                stages=[q/'generation.json']+([q/'draft/generation.json']if r['two_pass']else[])
                for p in stages:
                    v=read(p);actual_QA+=v['model_calls'];visual+=v['budget']['visual_tokens'];text+=v['budget']['text_and_special_input_tokens'];outputs+=v.get('output_tokens',0)
            input_checks.append(dict(record=rid,query_frame_id=r['query_frame_id'],actual_sandbox_in_model=True))
    assert sha(out/'D073_exclusions.json')==m['D073_exclusions_sha256']
    all_new_calls=all_new_visual=all_new_text=all_new_output=0
    for meta in m['records']:
        if meta['reused_development']:continue
        for query in meta['queries']:
            q=out/f'sample_{meta["record_index"]:02d}'/f'query_{query["query_frame_id"]:06d}'
            for path in [q/'generation.json',q/'draft/generation.json']:
                if not path.exists():continue
                generation=read(path);calls=generation['model_calls'];all_new_calls+=calls
                if calls:
                    all_new_visual+=generation['budget']['visual_tokens'];all_new_text+=generation['budget']['text_and_special_input_tokens']
                    all_new_output+=generation.get('output_tokens',0)
    artifact=dict(method='D116_frozen_full_D118',summary=summary,task_duration_groups=groups,partitions=partitions,queries=rows,
        full_denominators=True,all_inputs_previously_exposed=True,formal_baselines_reproduced=False)
    write(out/'scores.json',artifact)
    write(out/'input_audit.json',dict(status='passed',valid_joint_inputs=len(input_checks),checks=input_checks,
        D073_sha256_preserved=True,source_identity_rules_unchanged=True,new_QA_calls_on_valid_queries=actual_QA,
        new_QA_visual_tokens_on_valid_queries=visual,new_QA_text_tokens_on_valid_queries=text,
        new_QA_output_tokens_on_valid_queries=outputs,full_cold_start_certified=False))
    write(out/'completion_status.json',dict(status='completed',records=700,queries=2500,summary=summary,
        raw_failures_preserved=True,no_next_experiment=True))
    write(out/'cost_report.json',dict(known_new_QA_model_calls_including_failures=all_new_calls,
        new_QA_attempts_upper_bound=progress['new_QA_calls'],new_VGGT_attempts=progress['new_VGGT_calls'],
        new_SAM_prompt_attempts=progress['new_SAM_prompts'],new_discovery_calls=0,new_track_head_calls=0,
        new_QA_visual_tokens_including_failures=all_new_visual,new_QA_text_tokens_including_failures=all_new_text,
        new_QA_output_tokens_including_failures=all_new_output,sessions=progress['sessions'],
        inherited_perception_reused=True,inherited_perception_cost_sources=[str(Path(m['baseline'])/'manifest.json'),
            str(Path(m['development_source'])/'manifest.json')],full_cold_start_certified=False))
    print(json.dumps(dict(status='completed',summary=summary),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    score(p.parse_args().output.resolve())
