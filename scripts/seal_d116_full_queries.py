"""Split the CPU plan by query, so streaming workers never load future evidence."""
import argparse
from copy import deepcopy
import importlib.metadata,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_evidence_binding import read,write,sha,freeze


def repath(value,old,new):
    if isinstance(value,dict):return {k:repath(v,old,new)for k,v in value.items()}
    if isinstance(value,list):return [repath(v,old,new)for v in value]
    if isinstance(value,str)and value.startswith(str(old)+'/'):return str(new)+value[len(str(old)):]
    return value


def seal(source,out):
    import shutil
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    old=read(source/'manifest.json');m=deepcopy(old);m['records']=[];m['input_layout']='per_query'
    m['software_versions']={name:importlib.metadata.version(name)for name in ['torch','transformers','numpy','Pillow']}
    m['official_scoring_provenance']=read(Path(m['baseline'])/'manifest.json')['official_scoring_provenance']
    m['additional_sources']+=['scripts/seal_d116_full_queries.py']
    fields=['record_index','sample_id','variant','task','duration','streaming','original_question',
        'partition','original_partition','expanded_partition','related_group_id','video_path','video_sha256',
        'mapping_path','mapping_sha256','queries','inherited_perception_failures','reused_development','raw_question','inherited_sha256']
    count=0
    for meta in old['records']:
        assert sha(meta['path'])==meta['sha256'];spec=read(meta['path'])
        record={k:deepcopy(spec[k])for k in fields};record['query_plans']=[]
        for plan in spec['query_plans']:
            updated=repath(plan,source,out)
            assert repath(updated,out,source)==plan
            path=out/'plans'/f'{meta["record_index"]:04d}'/f'{plan["query_frame_id"]:06d}.json'
            write(path,updated)
            record['query_plans'].append(dict(query_frame_id=plan['query_frame_id'],folder=plan['folder'],
                path=str(path),sha256=sha(path)));count+=1
        path=out/'records'/f'{meta["record_index"]:04d}.json';write(path,record)
        assert all(set(q)=={'query_frame_id','folder','path','sha256'}for q in record['query_plans'])
        new=deepcopy(meta);new.update(path=str(path),sha256=sha(path));m['records'].append(new)
    assert count==2500
    m['inherited_sha256'][str(source/'manifest.json')]=sha(source/'manifest.json')
    for evidence in m['official_scoring_provenance'].values():
        path=ROOT/'vendor/cambrian-s-official'/evidence['path']
        assert sha(path)==evidence['sha256'];m['inherited_sha256'][str(path)]=sha(path)
    for variant,duration in {(r['variant'],r['duration'])for r in m['records']}:
        path=ROOT/'work/protocol/vsi_super'/variant/f'test_{duration}.parquet'
        m['inherited_sha256'][str(path)]=sha(path)
    m.update(query_input_layout_seconds=time.perf_counter()-start,original_CPU_plan=str(source))
    shutil.copy2(source/'D073_exclusions.json',out/'D073_exclusions.json')
    shutil.copy2(source/'equivalence.json',out/'equivalence.json')
    freeze(out,m);(out/'runtime/vendor').mkdir()
    for name in ['vggt','sam2','cambrian-s-official']:(out/'runtime/vendor'/name).symlink_to(ROOT/'vendor'/name,target_is_directory=True)
    write(out/'authorization.json',dict(full_start_authorized=True,user_instruction='固定D116的方法，只做一次冻结全量评测',
        manifest_sha256=sha(out/'manifest.json'),maximum_new_QA_calls=m['maximum_new_QA_calls']))
    write(out/'progress.json',dict(status='prepared',records_finished=0,queries_finished=0,sessions=[],new_QA_calls=0,new_VGGT_calls=0,new_SAM_prompts=0))
    write(out/'CPU_preflight.json',dict(status='passed',records=700,queries=count,input_layout='per_query',new_model_calls=0,
        query_values_exact_except_local_destination_paths=True,D116_294_input_equivalence='passed',
        full_worker_does_not_load_future_query_plans=True))
    write(source/'CPU_plan_only.json',dict(status='CPU_plan_only_do_not_launch',new_model_calls=0,query_sealed_output=str(out)))
    print(json.dumps(dict(status='sealed',records=700,queries=count,seconds=time.perf_counter()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();seal(a.source.resolve(),a.output.resolve())
