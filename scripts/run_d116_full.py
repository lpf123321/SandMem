"""One frozen D116 full measurement, serial resumptions at record boundaries."""
import argparse
from copy import deepcopy
import gc,hashlib,importlib.metadata,json,os,re,resource,signal,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'vendor/vggt')]
from run_evidence_binding import read,write,sha


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');write(temp,value);os.replace(temp,path)


def tracking_state(t):
    fields=['maximum_seeds','threshold','attempted','seed_attempts','history','accepted_edges','processed_through',
        'failed_source_keys','new_seed_attempts','new_edges','legacy_seed_attempts','last_watermark','skipped_seed_groups']
    return {k:sorted(getattr(t,k))if isinstance(getattr(t,k),set)else deepcopy(getattr(t,k))for k in fields}


def restore_tracking(t,data):
    for k,v in data.items():setattr(t,k,set(v)if k in ['attempted','failed_source_keys']else v)


def execute(out):
    import cv2,numpy as np,torch
    from sandbox_vlm.qa.model import QwenAnswerer
    from sandbox_vlm.perception.sam_tracks import SamCandidateTracker
    from sandbox_vlm.perception.vggt import VggtGeometry
    from sandbox_vlm.perception.coverage_short_track import CoverageShortTrack
    from sandbox_vlm.memory.coverage import CoverageSandbox
    from sandbox_vlm.memory.joint_geometry import observations,merge_correspondences
    from sandbox_vlm.qa.d065_spatial import instances_only,reconstruct
    from sandbox_vlm.qa.observation_sandbox import observation_packet
    from sandbox_vlm.qa.d116_full import compose,draw,joint_messages
    from sandbox_vlm.qa.map_only import original_messages,indexed_messages,text_tokens
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import QABudget,check_budget
    from sandbox_vlm.qa.bounded_notes import POLICY
    from sandbox_vlm.qa.two_pass import eligible,candidate_from_generation,verification_messages,decide
    from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
    from sandbox_vlm.memory.storage import cpu_storage_report
    from sandbox_vlm.runtime import VideoFrameReader
    from sandbox_vlm.data.video import FrameMapping
    if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    m=read(out/'manifest.json');auth=read(out/'authorization.json');progress=read(out/'progress.json')
    assert m.get('input_layout')=='per_query','seal query-time inputs before launching'
    assert auth['full_start_authorized']and auth['manifest_sha256']==sha(out/'manifest.json')
    if progress['status']=='completed':raise FileExistsError('one full measurement only')
    for group in ['source_sha256','inherited_sha256']:
        for p,h in m[group].items():assert sha(ROOT/p if group=='source_sha256'else p)==h,p
    start=time.perf_counter();torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(0)
    stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    journal=out/'attempts.jsonl';counters={'QA':0,'VGGT':0,'SAM_prompt':0}
    if journal.exists():
        for line in journal.read_text().splitlines():counters[json.loads(line)['kind']]+=1
    current={'record':None,'query':None,'stage':None}
    def attempt(kind):
        limits={'QA':m['maximum_new_QA_calls'],'VGGT':m['maximum_new_VGGT_calls'],
            'SAM_prompt':m['maximum_restore_SAM_prompts']+m['maximum_short_track_SAM_prompts']}
        if counters[kind]>=limits[kind]:raise RuntimeError('sealed call limit: '+kind)
        counters[kind]+=1
        with journal.open('a')as f:f.write(json.dumps(dict(kind=kind,ordinal=counters[kind],**current))+'\n')
    device=torch.cuda.get_device_properties(0)
    session=dict(job_id=os.environ['SLURM_JOB_ID'],status='running',device_name=device.name,total_memory_bytes=device.total_memory,
        first_record=progress['records_finished'],starting_attempts=dict(counters))
    versions={name:importlib.metadata.version(name)for name in ['torch','transformers','numpy','Pillow']}
    assert versions==m['software_versions'],'software changed after freezing'
    if progress['sessions']:assert versions==progress['sessions'][0]['software_versions'],'software changed across continuation'
    session['software_versions']=versions
    progress['sessions'].append(session);progress['status']='running';atomic(out/'progress.json',progress)
    model=QwenAnswerer(m['config']['qa']['mllm']);sam=SamCandidateTracker(m['config']['perception']['sam']);geo=VggtGeometry(m['config']['perception']['vggt'])
    original_sam=sam.predictor.add_new_points_or_box
    def counted_sam(*a,**kw):attempt('SAM_prompt');return original_sam(*a,**kw)
    sam.predictor.add_new_points_or_box=counted_sam
    verified_videos=set();peak_structure=0;geometry_workspace=0
    archive_bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file()and not p.is_symlink())

    def charge(*values,extra=0):
        nonlocal peak_structure
        size=cpu_storage_report(values)['cpu_structural_bytes']+extra
        peak_structure=max(peak_structure,size)
        if size>m['structure_limit_bytes']:raise MemoryError('complete live structure exceeds64MiB')
        return size

    def vg(paths,ids,ref):
        nonlocal geometry_workspace
        attempt('VGGT');raw=geo.infer(paths,ids,reference=ref)
        write(out/'costs'/f'VGGT_{counters["VGGT"]:06d}.json',dict(**current,frame_ids=ids,reference=ref,cost=raw['cost']))
        geometry_workspace=max(geometry_workspace,raw['depth'].nbytes+raw['confidence'].nbytes)
        return raw

    def mask_prompts(reader,rows,destination):
        if not rows:return {}
        fid=rows[0]['evidence']['frame_id'];result={}
        with tempfile.TemporaryDirectory(prefix='d118-sam-')as temp:
            data,_,_=reader.encoded_source(fid);(Path(temp)/f'{fid:06d}.jpg').write_bytes(data)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=sam.predictor.init_state(temp)
                try:
                    for row in rows:
                        assert row['evidence']['frame_id']==fid
                        sam.predictor.reset_state(state)
                        _,ids,logits=sam.predictor.add_new_points_or_box(state,frame_idx=0,obj_id=0,
                            box=np.asarray(row['evidence']['bbox_xyxy'],dtype=np.float32))
                        assert ids==[0];result[row['observation_id']]=(logits[0,0]>0).cpu().numpy().copy()
                finally:sam.predictor.reset_state(state)
        if destination is not None:np.savez_compressed(destination,**result)
        return result

    def measure(processor,messages,paths,hashes,limits,original,sparse,note=None):
        from PIL import Image
        sizes=[]
        for p in paths:
            with Image.open(p)as im:sizes.append([im.height,im.width])
        grids=deepcopy(original['original_image_grid_thw'])
        if len(paths)>len(grids):
            with Image.open(paths[0])as image:grid=processor.image_processor(images=[image.convert('RGB')],return_tensors='pt')['image_grid_thw'].tolist()[0]
            grids.insert(0,grid)
        visual=sum(g[0]*g[1]*g[2]//processor.image_processor.merge_size**2 for g in grids)
        tokens=text_tokens(processor.tokenizer,messages,len(paths),processor.chat_template);patch=processor.image_processor.patch_size
        budget=check_budget(QABudget(**limits),image_sizes_hw=sizes,processed_image_sizes_hw=[[g[1]*patch,g[2]*patch]for g in grids],
            input_tokens=tokens+visual,visual_tokens=visual,context_limit=original['original_budget']['model_context_limit'])
        if budget['status']!='ok':raise ValueError(budget)
        snap=dict(paths=paths,pixel_sha256=hashes,limits=limits,budget=budget,grids=grids,
            generation_kwargs=dict(do_sample=False,max_new_tokens=limits['max_output_tokens']),
            has_map=len(paths)>len(original['frame_ids']),sparse=sparse,response_style='D065_strict')
        if note:snap['note_boundary']=dict(note)
        return snap

    def qa(folder,snap,serial,live):
        from sandbox_vlm.qa.continuous_sandbox import open_rgb_images
        folder.mkdir(parents=True,exist_ok=True)
        if(folder/'generation.json').exists():raise FileExistsError('no repeated model answer')
        images=open_rgb_images(snap['paths'])
        try:
            assert [pixel_sha256(i)for i in images]==snap['pixel_sha256']
            messages=deepcopy(serial)
            for message in messages:
                for c in message['content']:
                    if c['type']=='image':
                        index=c.pop('evidence_image_index');assert c.pop('pixel_sha256')==snap['pixel_sha256'][index];c['image']=images[index]
            charge(m,progress,live,snap,serial,messages,extra=sum(i.width*i.height*3 for i in images)+8*len(json.dumps(serial).encode())+2**20)
            write(folder/'input.json',snap);write(folder/'messages.json',serial)
            attempt('QA');g=model.answer(messages,images,QABudget(**snap['limits']),note_boundary=snap.get('note_boundary'))
            write(folder/'generation.json',g)
            assert g['budget']==snap['budget']and g['image_grid_thw']==snap['grids']and g['generation_kwargs']==snap['generation_kwargs']
            return g
        finally:
            for image in images:image.close()

    try:
        for meta in m['records'][progress['records_finished']:]:
            rid=meta['record_index'];current['record']=rid
            assert sha(meta['path'])==meta['sha256'];record=read(meta['path']);folder=out/f'sample_{rid:02d}';folder.mkdir(exist_ok=True)
            previous_archive=sum(p.stat().st_size for p in folder.rglob('*')if p.is_file()and not p.is_symlink())
            rows=[];track=CoverageShortTrack(sam);track_path=folder/'tracking_checkpoint.json'
            if track_path.exists():restore_tracking(track,read(track_path))
            if not record['reused_development']:
                for p,h in record['inherited_sha256'].items():assert sha(p)==h,p
                if record['video_path']not in verified_videos:
                    assert sha(record['video_path'])==record['video_sha256'];verified_videos.add(record['video_path'])
                reader=VideoFrameReader(record['video_path'],FrameMapping.load(record['mapping_path']))
            interrupted=False
            for plan_meta in record['query_plans']:
                qid=plan_meta['query_frame_id'];current.update(query=qid,stage='preparation');dest=out/plan_meta['folder'];dest.mkdir(parents=True,exist_ok=True)
                if(dest/'result.json').exists():rows.append(read(dest/'result.json'));continue
                if not record['reused_development']:reader.allowed_through=qid
                assert sha(plan_meta['path'])==plan_meta['sha256'];plan=read(plan_meta['path'])
                row=dict(record=rid,sample_id=record['sample_id'],query_frame_id=qid,task=record['task'],duration=record['duration'],
                    partition=record['partition'],expanded_partition=record['expanded_partition'],status='query_failed',prediction=None)
                if(dest/'attempt.json').exists():interrupted=True
                if interrupted:
                    row['error']='record_interrupted_preserved_without_model_retry';write(dest/'result.json',row);rows.append(row);continue
                write(dest/'attempt.json',dict(job_id=session['job_id'],status='started'))
                before=dict(counters);query_start=time.perf_counter();diagram=None;image=None
                try:
                    if record['reused_development']:
                        src=Path(plan['development_folder']);j=plan['development_job']
                        # Reuse only exact frozen input equivalence established for all 294.
                        dev_report=read(Path(m['development_source'])/'run_report.json')
                        oldrow=next(r for r in dev_report['queries']if r['record']==rid and r['query_frame_id']==qid)
                        for name in ['input.json','messages.json','generation.json']:(dest/name).symlink_to((src/name).resolve())
                        row.update(status=oldrow['status'],prediction=oldrow['prediction'],reused_development=True,
                            two_pass=j['new_QA_required'],source_folder=str(src))
                    else:
                        reader.allowed_through=qid;original=plan['original'];item=plan['frozen_query']
                        state=instances_only(CoverageSandbox.from_dict(read(plan['control_path'])))
                        base=instances_only(CoverageSandbox.from_dict(read(item['base_path'])))
                        legacy_edges=plan['legacy']['accepted_edges']+plan['old_geometry_edges']
                        reproduced,_=merge_correspondences(base,observations(base),legacy_edges,watermark=qid)
                        assert reproduced.to_dict()==state.to_dict();del reproduced,_
                        track.import_legacy(base,plan['legacy'],plan['legacy_clips'],watermark=qid)
                        current['stage']='existing_SAM_coverage';discarded,audit=track.update(base,reader,watermark=qid,audit_dir=dest/'short_track')
                        del discarded
                        if track.new_seed_attempts>96:raise RuntimeError('D102 additional seed cap')
                        write(dest/'track_audit.json',audit)
                        charge(m,progress,record,plan,reader.mapping,base,state,track.to_dict(),audit,
                            extra=max((c.get('temporary_CPU_mask_bytes',0)for c in audit['clips']),default=0))
                        discarded,union=merge_correspondences(base,observations(base),legacy_edges+track.new_edges,watermark=qid)
                        del discarded,base;write(dest/'new_unions.json',union)
                        current['stage']='existing_box_masks';fallback=mask_prompts(reader,plan['fallback_masks'],None)
                        if fallback:
                            np.savez_compressed(dest/'shared_masks.npz',**{'prompt__'+k:v for k,v in fallback.items()})
                        del fallback
                        paths=[]
                        with tempfile.TemporaryDirectory(prefix='d118-base-')as temp:
                            for fid in plan['geometry_frame_ids']:
                                data,ref,_=reader.encoded_source(fid)
                                p=Path(temp)/f'{ref.frame_id:06d}.jpg';p.write_bytes(data);paths.append(p)
                            current['stage']='control_VGGT';raw=vg(paths,plan['geometry_frame_ids'],f'D102/{rid}/{qid}')
                            geometry,geo_audit=reconstruct(raw,state,plan['catalog'],plan['geometry_frame_ids'],include_scene_extent=True)
                            frozen=dict(sandbox=geometry.to_dict(),audit=geo_audit,cost=raw['cost']);del raw,geometry
                        write(dest/'control_geometry.json',frozen);packets=[]
                        for op in plan['observation_plans']:
                            sf=op['source_frame'];window=dest/f'window_F{sf:06d}';window.mkdir();masks={};missing=[]
                            for obs in op['rows']:
                                cache=obs['mask_source']
                                if cache:
                                    with np.load(cache['path'],allow_pickle=False)as z:masks[obs['observation_id']]=z[cache['key']].copy()
                                else:missing.append(obs)
                            masks.update(mask_prompts(reader,missing,None))
                            np.savez_compressed(window/'source_masks.npz',**masks)
                            with tempfile.TemporaryDirectory(prefix='d118-local-')as temp:
                                paths=[];source_hashes=[]
                                for data,ref,_ in reader.encoded_sequence(op['frame_ids']):
                                    p=Path(temp)/f'{ref.frame_id:06d}.jpg';p.write_bytes(data);paths.append(p)
                                    source_hashes.append(dict(frame_id=ref.frame_id,source_index=ref.source_index,
                                        jpeg_sha256=hashlib.sha256(data).hexdigest()))
                                    if ref.frame_id==sf:
                                        oldimage=next(i for i in original['image_records']if i['frame_id']==sf)
                                        assert hashlib.sha256(data).hexdigest()==oldimage['source_sha256']
                                current['stage']='source_window_VGGT';raw=vg(paths,op['frame_ids'],op['reference'])
                                packet=observation_packet(raw,sf,op['rows'],masks)
                                packet.update(window_frame_ids=op['frame_ids'],source_hashes=source_hashes,
                                    window_cameras=[c.to_dict()for c in raw['cameras']],cost=raw['cost'])
                                del raw,masks
                            write(window/'packet.json',packet);packets.append(packet)
                        payload,geometry,construction=compose(state.to_dict(),frozen,packets,union['edges'],original['frame_ids'],qid,record['raw_question'])
                        write(dest/'geometry.json',geometry);write(dest/'construction.json',construction)
                        diagram=draw(geometry);diagram.save(dest/'sandbox.png');raw_paths=[];hashes=[pixel_sha256(diagram)]
                        for info in original['image_records']:
                            image,actual=reader.read(info['frame_id'],max_side=max(info['input_image_size_hw']))
                            assert actual==info;path=dest/f'frame_{info["frame_id"]}.png';image.save(path)
                            raw_paths.append(str(path));hashes.append(pixel_sha256(image));image.close()
                        images=[object()for _ in raw_paths]
                        messages,joint=joint_messages(original['question'],original['sparse'],original['frame_ids'],payload,images,diagram)
                        serial=indexed_messages(messages,joint,hashes)
                        limits=deepcopy(original['original_budget']['limits']);limits.update(max_images=33,max_text_tokens=32768,
                            max_image_side=1280 if construction['geometry_repaired']else 768)
                        snap=measure(model.processor,messages,[str(dest/'sandbox.png'),*raw_paths],hashes,limits,original,
                            original['sparse'],POLICY if original['sparse']else None)
                        two=eligible(snap,geometry,payload);diagram.close()
                        live=dict(record=record,current_query=plan,frame_mapping=reader.mapping,
                            tracking=tracking_state(track),geometry=geometry,payload=payload,construction=construction)
                        charge(m,progress,live,state,packets,frozen,extra=2**20)
                        del state,packets,frozen,union,audit
                        if two:
                            current['stage']='D065_independent_candidate'
                            draft_messages=original_messages(original,images,m['memory_notice'])
                            draft_serial=indexed_messages(draft_messages,images,hashes[1:])
                            ds=measure(model.processor,draft_messages,raw_paths,hashes[1:],original['original_budget']['limits'],original,original['sparse'])
                            assert ds['budget']==original['original_budget']
                            draft=qa(dest/'draft',ds,draft_serial,live);candidate=candidate_from_generation(draft)
                            write(dest/'candidate.json',candidate);write(dest/'base_messages.json',serial);write(dest/'base_input.json',snap)
                            serial=verification_messages(serial,candidate)
                            snap=measure(model.processor,serial,snap['paths'],hashes,limits,original,original['sparse'],POLICY)
                            current['stage']='joint_verification';generation=qa(dest,snap,serial,live)
                            decision=decide(candidate,generation,original['frame_ids']);write(dest/'decision.json',decision)
                            answer=decision['prediction'];status=decision['status']
                        else:
                            current['stage']='joint_single';generation=qa(dest,snap,serial,live)
                            answer=None;status=generation['status']
                            if status=='generated':answer=extract_d065_answer(generation['raw_output'],sparse=original['sparse']);status='ok'
                        prediction=float(answer)if record['variant']=='count'and answer is not None else answer
                        if isinstance(prediction,float)and not np.isfinite(prediction):raise ValueError('nonfinite answer')
                        row.update(status=status,prediction=prediction,two_pass=two,reused_development=False,
                            stored_records=len(payload['records']),structure_upper_bytes=peak_structure)
                        del live,payload,geometry,messages,joint,images,serial,original
                except Exception as exc:
                    row.update(status='query_failed',prediction=None,error=repr(exc),traceback=traceback.format_exc(),failed_stage=current['stage'])
                    torch.cuda.empty_cache()
                finally:
                    row['new_attempts']={k:counters[k]-before[k]for k in counters}
                    row['wall_seconds']=time.perf_counter()-query_start
                    atomic(track_path,tracking_state(track));write(dest/'result.json',row);rows.append(row)
                    print('D118_QUERY',rid,qid,row['status'],row.get('two_pass'),row.get('error',''),flush=True)
                    if diagram is not None:diagram.close()
                    if image is not None:image.close()
                    live=state=base=payload=geometry=packets=frozen=raw=union=audit=messages=joint=images=serial=None
                    gc.collect()
            write(folder/'record_report.json',dict(status='completed',record=rid,queries=rows,tracking=tracking_state(track)))
            archive_bytes+=sum(p.stat().st_size for p in folder.rglob('*')if p.is_file()and not p.is_symlink())-previous_archive
            if archive_bytes+2**20>m['archive_limit_bytes']:raise MemoryError('sealed 200GiB archive cap')
            progress.update(records_finished=rid+1,queries_finished=progress['queries_finished']+len(rows),new_QA_calls=counters['QA'],
                new_VGGT_calls=counters['VGGT'],new_SAM_prompts=counters['SAM_prompt'],
                maximum_structure_upper_bytes=max(progress.get('maximum_structure_upper_bytes',0),peak_structure),archive_bytes=archive_bytes)
            atomic(out/'progress.json',progress);print('D118_RECORD',rid+1,'/700',flush=True)
            if stop[0]or time.perf_counter()-start>=m['maximum_session_seconds']:break
        completed=progress['records_finished']==700
        progress['status']='completed'if completed else'checkpointed'
        session.update(status='completed',wall_seconds=time.perf_counter()-start,ending_attempts=dict(counters),
            last_finished_record=progress['records_finished'],peak_structure_bytes=peak_structure,
            maximum_geometry_workspace_bytes=geometry_workspace,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        atomic(out/'progress.json',progress)
        if completed:
            return subprocess.call([sys.executable,str(ROOT/'scripts/score_d116_full.py'),'--output',str(out)])
        # Preserve the user's A100 preference with A30 when that partition is occupied.
        try:
            node=subprocess.check_output(['scontrol','show','node','gpu1','-o'],text=True)
            total=dict(x.split('=',1)for x in node.split('CfgTRES=')[1].split()[0].split(','))
            allocated=dict(x.split('=',1)for x in node.split('AllocTRES=')[1].split()[0].split(','))
            free=int(total['gres/gpu'])>int(allocated.get('gres/gpu',0))
            free=free and int(re.search(r'CPUAlloc=(\d+)',node)[1])<int(re.search(r'CPUTot=(\d+)',node)[1])
        except Exception:free=False
        hardware=[]if free else['--partition=gpu-a30','--gres=gpu:a30:1']
        launch=['sbatch',f'--dependency=afterok:{session["job_id"]}',*hardware,str(ROOT/'scripts/run_d116_full.sbatch'),str(out)]
        receipt=subprocess.check_output(launch,text=True).strip();write(out/f'continuation_{session["job_id"]}.json',dict(receipt=receipt,command=launch))
        print('CONTINUATION',receipt,flush=True);return 0
    except Exception as exc:
        session.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),ending_attempts=dict(counters))
        progress['status']='failed';atomic(out/'progress.json',progress);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    raise SystemExit(execute(p.parse_args().output.resolve()))
