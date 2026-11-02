"""D118 answer-free plan and immutable per-record inputs for frozen D116."""
import argparse
from copy import deepcopy
import ast,json,hashlib,shutil,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_evidence_binding import read,write,sha,freeze


def prepare(out,equivalence):
    from sandbox_vlm.memory.coverage import CoverageSandbox
    from sandbox_vlm.qa.d065_spatial import instances_only,joint_frame_plan
    from sandbox_vlm.qa.observation_sandbox import local_window
    from sandbox_vlm.data.vsi_super import load_super_inputs
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'work/bench/vsi_super_round3_d065_incremental_20261022_01'
    dev=ROOT/'work/bench/d116_two_pass_20261101_01';old=read(baseline/'manifest.json');dm=read(dev/'manifest.json')
    expanded=read(ROOT/'configs/vsi_super_expanded_dev_20261029.json')
    excluded_path=ROOT/'work/bench/d073_text_capacity_20261024_01/development_related_exclusions.json'
    excluded={r['index']:r['sample_id']for r in read(excluded_path)['records']}
    dev_records={r['record_index']:r for r in expanded['records']};dev_groups={r['related_group_id']for r in expanded['records']}
    dev_jobs={(j['record'],j['query_frame_id']):j for j in dm['jobs']}
    with zipfile.ZipFile(baseline/'code_snapshot.zip')as z:
        tree=ast.parse(z.read('sandbox_vlm/qa/coverage_evidence.py').decode())
    notice=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='MEMORY_NOTICE'for t in n.targets))
    m=dict(method='D116_frozen_full_D118',baseline=str(baseline),development_source=str(dev),config=old['config'],
        records=[],memory_notice=notice,structure_limit_bytes=64*2**20,maximum_session_seconds=39600,
        archive_limit_bytes=200*2**30,all_inputs_previously_exposed=True,new_discovery_calls=0,new_identity_decisions=0,
        new_track_head_calls=0,retries=0,D073_exclusions_sha256=sha(excluded_path),input_sha256={},inherited_sha256={},
        additional_sources=['scripts/prepare_d116_full.py','scripts/run_d116_full.py','scripts/score_d116_full.py',
            'scripts/run_d116_full.sbatch','scripts/check_d116_full_equivalence.py','tests/test_d116_full.py'])
    m['config']['memory']['retain_unresolved_observations']=False
    m['config']['coverage']['retain_ambiguities']=False
    fixed=['two_pass.py','minimal_d065_prompt.py','note_boundary.py','bounded_notes.py','complete_sandbox.py',
           'complete_sandbox_repair.py','retrieved_sandbox.py','evidence_binding.py','model.py','map_only.py','prompt.py']
    for name in fixed:
        rel='sandbox_vlm/qa/'+name;assert sha(ROOT/rel)==sha(dev/'runtime'/rel),rel
    tables={};vg=sam_restore=source_windows=prompt_upper=0;mask_hashes={}
    for i,r in enumerate(old['records']):
        spec=deepcopy(r);spec['record_index']=i;spec['reused_development']=i in dev_records
        key=r['variant'],r['duration']
        if key not in tables:tables[key]=load_super_inputs(ROOT/'work/protocol/vsi_super'/key[0]/f'test_{key[1]}.parquet',variant=key[0])
        example=tables[key][int(r['sample_id'].split(':')[1])];assert example.sample_id==r['sample_id']
        spec['raw_question']=example.original_question;assert example.model_question==r['original_question']
        spec['original_partition']=r['partition']
        if i in excluded:assert excluded[i]==r['sample_id'];spec['partition']='exposed_D073_related'
        spec['expanded_partition']=dev_records[i]['phase']if i in dev_records else(
            'expanded_development_related'if r['related_group_id']in dev_groups else'previously_exposed_remaining')
        catalog={};clips=[];geo_edges=[];spec['query_plans']=[];hashes={r['mapping_path']:r['mapping_sha256']}
        for item in r['frozen_queries']:
            q=item['query_frame_id'];folder=f'sample_{i:02d}/query_{q:06d}'
            qa_path=baseline/folder/'qa.json';qa=read(qa_path)
            original=dict(question=qa['original_question'],compact_memory=qa['compact_memory'],frame_ids=[v['frame_id']for v in qa['images']],
                image_records=qa['images'],original_budget=qa['generation']['budget'],original_image_grid_thw=qa['generation']['image_grid_thw'],
                generation_kwargs=qa['generation']['generation_kwargs'],sparse=item['sparse'])
            assert original['frame_ids']==item['answer_frame_ids']and max(original['frame_ids'])<=q
            assert original['question']==r['original_question']
            plan=dict(query_frame_id=q,folder=folder,original=original,frozen_query=deepcopy(item))
            if spec['reused_development']:
                j=dev_jobs[i,q];plan['development_job']=j;plan['development_folder']=str(dev/folder)
            else:
                control_path=baseline/folder/'visual_sandbox.json';state=instances_only(CoverageSandbox.from_dict(read(control_path)))
                plan['control_path']=str(control_path);hashes[str(control_path)]=sha(control_path)
                for k in ['base','sam_auxiliary']:
                    assert sha(item[k+'_path'])==item[k+'_sha256'];hashes[item[k+'_path']]=item[k+'_sha256']
                legacy=read(item['sam_auxiliary_path'])['short_track']
                clip_path=Path(item['sam_auxiliary_path']).parent/'short_track.json';hashes[str(clip_path)]=sha(clip_path)
                clips+=[{k:c[k]for k in ['source_frame','end_frame','status']}for c in read(clip_path)['clips']]
                for w in r['geometry_windows']:
                    if w['available_through']==q:
                        p=baseline/f'sample_{i:02d}/geometry'/w['name']/'memory.json';hashes[str(p)]=sha(p);geo_edges+=read(p)['edges']
                catalog.update(deepcopy(item['mask_catalog_additions']));fallback=[]
                for v in item['mask_catalog_additions'].values():
                    if v['path']not in mask_hashes:mask_hashes[v['path']]=sha(v['path'])
                    hashes[v['path']]=mask_hashes[v['path']]
                ids=item['answer_frame_ids']
                supported=any(f'F{f}_O{j}'in catalog for f in ids for j,_ in enumerate(state.viewpoints.get(str(f),[])))
                if not supported and state.instances:
                    fid=max(ids,key=lambda f:(len(state.viewpoints.get(str(f),[])),-f))
                    for j,link in enumerate(state.viewpoints.get(str(fid),[])):
                        oid=f'F{fid}_O{j}';entry=dict(path=str(out/folder/'shared_masks.npz'),key='prompt__'+oid,available_through=q)
                        catalog[oid]=entry;fallback.append(dict(observation_id=oid,evidence=link['evidence'],**entry))
                _,geometry_frames,frame_plan=joint_frame_plan(state,ids,catalog,watermark=q,allow_scene_only=True)
                assert not frame_plan['added_for_actual_geometry']
                observations=[]
                if item['sparse']:
                    for fid in ids:
                        rows=[]
                        for j,link in enumerate(state.viewpoints.get(str(fid),[])):
                            row=deepcopy(link);oid=f'F{fid}_O{j}';row.update(observation_id=oid,mask_source=catalog.get(oid));rows.append(row)
                            if row['mask_source']is None:sam_restore+=1
                        if rows:observations.append(dict(source_frame=fid,frame_ids=local_window(fid,q),rows=rows,reference=f'D106_record{i}_F{fid}'))
                sam_restore+=len(fallback);source_windows+=len(observations);vg+=1+len(observations)
                all_rows=[(int(f),len(v))for f,v in state.viewpoints.items()]
                # Static upper bound includes every possible endpoint prompt window;
                # actual fixed D102 schedule remains limited to 96 additional seeds.
                prompt_upper+=sum(sum(n for f,n in all_rows if sf<=f<=min(sf+15,q))for sf,_ in all_rows)
                plan.update(legacy=legacy,legacy_clips=deepcopy(clips),old_geometry_edges=deepcopy(geo_edges),
                    catalog=deepcopy(catalog),fallback_masks=fallback,geometry_frame_ids=geometry_frames,observation_plans=observations)
            hashes[str(qa_path)]=sha(qa_path)
            spec['query_plans'].append(plan)
        spec.pop('frozen_queries');spec['inherited_sha256']=hashes
        path=out/'records'/f'{i:04d}.json';write(path,spec)
        metadata={k:spec[k]for k in ['record_index','sample_id','variant','task','duration','partition','original_partition',
            'expanded_partition','related_group_id','reused_development']}
        metadata.update(path=str(path),sha256=sha(path),queries=r['queries'])
        m['records'].append(metadata)
        if(i+1)%25==0:print('PLAN',i+1,flush=True)
    count=sum(len(r['queries'])for r in m['records']);assert len(m['records'])==700 and count==2500
    m.update(queries=2500,reused_development_queries=294,maximum_new_QA_calls=2477,maximum_new_VGGT_calls=vg,
        maximum_new_SAM_seeds=622*96,maximum_restore_SAM_prompts=sam_restore,
        maximum_short_track_SAM_prompts=prompt_upper,source_observation_windows=source_windows,
        planned_new_joint_geometry_calls=2206,preparation_seconds=time.perf_counter()-start)
    result=read(equivalence);assert result['status']=='passed'and result['queries']==294
    for p in [baseline/'manifest.json',dev/'manifest.json',dev/'run_report.json',dev/'input_audit.json',equivalence,excluded_path]:
        m['inherited_sha256'][str(p)]=sha(p)
    shutil.copy2(excluded_path,out/'D073_exclusions.json');write(out/'equivalence.json',result)
    freeze(out,m)
    (out/'runtime/vendor').mkdir()
    for name in ['vggt','sam2','cambrian-s-official']:(out/'runtime/vendor'/name).symlink_to(ROOT/'vendor'/name,target_is_directory=True)
    write(out/'authorization.json',dict(full_start_authorized=True,user_instruction='固定D116的方法，只做一次冻结全量评测',
        manifest_sha256=sha(out/'manifest.json'),maximum_new_QA_calls=m['maximum_new_QA_calls']))
    write(out/'progress.json',dict(status='prepared',records_finished=0,queries_finished=0,sessions=[],new_QA_calls=0,new_VGGT_calls=0,new_SAM_prompts=0))
    print(json.dumps({k:m[k]for k in ['queries','maximum_new_QA_calls','maximum_new_VGGT_calls','maximum_new_SAM_seeds',
        'maximum_restore_SAM_prompts','maximum_short_track_SAM_prompts','preparation_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--equivalence',type=Path,required=True)
    a=p.parse_args();prepare(a.output.resolve(),a.equivalence.resolve())
