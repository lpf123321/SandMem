"""Fresh Spatial perception -> frozen D116. Wiring IDs are fixed, never scored.

Archives are evidence only; each inference starts without Qwen perception state.
Logical 64-frame screening blocks keep their original novelty/weak-hit policy;
independent language prompts are executed one at a time to bound VRAM.
"""
import argparse
from copy import deepcopy
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/vggt'),str(ROOT/'vendor/sam2')]
DATA=ROOT/'work/sandmem_vsibench_20261106_01'


def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def d116_identity_and_correspondences(base,legacy_edges,additional_edges,*,watermark):
    """D118 lines187-199: extra D102 edges are QA evidence, not new IDs."""
    from sandbox_vlm.memory.joint_geometry import observations,merge_correspondences
    state,_=merge_correspondences(base,observations(base),legacy_edges,watermark=watermark)
    _,union=merge_correspondences(base,observations(base),list(legacy_edges)+list(additional_edges),watermark=watermark)
    return state,union


def equivalent_cached_identity(prior):
    path=Path(prior)/'sam_correspondences.json'
    if not path.exists() and (Path(prior)/'prior_evidence').exists():
        return equivalent_cached_identity(Path(prior)/'prior_evidence')
    if not path.exists():return False
    report=read(path)
    # Also preserves original pre-SAM answer-frame selection for these cases.
    return report['legacy']['summary']['removed_hypotheses']==0 and report['union']['removed_hypotheses']==0


def equivalent_cached_execution(prior):
    if not equivalent_cached_identity(prior):return False
    source=Path(prior)
    while not (source/'state.json').exists() and (source/'prior_evidence').exists():source=source/'prior_evidence'
    if not (source/'retention.json').exists():return False
    retention=read(source/'retention.json')
    if retention['compaction_applied'] or retention['evicted_instance_ids']:return False
    from sandbox_vlm.memory.coverage import CoverageSandbox
    from sandbox_vlm.memory.joint_geometry import observations
    from sandbox_vlm.qa.d065_spatial import joint_frame_plan
    state=CoverageSandbox.from_dict(read(source/'state.json'))
    frames=[r['frame_id'] for r in read(source/'image_records.json')['frames']]
    catalog={}
    for row in observations(state):
        fid=row['evidence']['frame_id'];oid=row['observation_id']
        path=source/f'observation_F{fid:06d}'/(oid+'.npz')
        if path.exists():catalog[oid]=dict(path=str(path),key='mask',available_through=read(source/'mapping.json')['frames'][-1]['frame_id'])
    watermark=read(source/'mapping.json')['frames'][-1]['frame_id']
    _,geometry_ids,_=joint_frame_plan(state,frames,catalog,watermark=watermark,allow_scene_only=True)
    return geometry_ids==read(source/'geometry_calls/01.json')['frame_ids']


def discovery_with_failure_receipt(bundle,path,fid,question):
    """Map a decoder syntax dead end to the existing visible perception failure."""
    prefix=[];width=[None]
    def capture(module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is None:return
        row=ids[0].detach().cpu().tolist()
        if width[0] is None:width[0]=len(row);prefix.extend(row)
        else:prefix.extend(row)
    handle=bundle.runtime.model.register_forward_pre_hook(capture,with_kwargs=True)
    try:return bundle.discovery.discover_batch([path],[fid],question=question)[0]
    except ValueError as error:
        if 'prefix_allowed_tokens_fn' not in str(error) or 'empty list' not in str(error):raise
        tail=prefix[width[0]:] if width[0] is not None else []
        return dict(status='perception_failed',frame_id=fid,detections=[],error=str(error),
            raw_output=bundle.runtime.processor.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            output_token_ids=tail,output_tokens=len(tail),partial_output=True,model_calls=int(width[0] is not None),
            decoder_exception='unsatisfiable syntax prefix; no repair or retry')
    finally:handle.remove()


def prepare(out,resume_from=None):
    if out.exists():raise FileExistsError(out)
    original=read(ROOT/'provenance/frozen_files.json')['source_sha256']
    for p,h in original.items():assert sha(ROOT/p)==h,p
    cfg=read(ROOT/'configs/spatial135k_output_compat.json')
    assert cfg['coverage']['retain_ambiguities'] is False
    assert cfg['memory']['retain_unresolved_observations'] is False
    fixed=read(DATA/'wiring_inputs.json')['inputs'];assert [r['id'] for r in fixed]==[69,1077]
    out.mkdir(parents=True)
    # The release excludes binary extensions. Preserve the original extension,
    # loading it with the release's same frozen SAM Python source, never a fallback.
    native=out/'native_sam';native.mkdir()
    source=Path('/home/u2024311136/sandbox/vendor/sam2/sam2/_C.so')
    shutil.copy2(source,native/'_C.so')
    files=set(original)
    for pattern in ['sandbox_vlm/backbones/*.py','scripts/run_spatial_vsibench.py','scripts/run_spatial_vsibench.sbatch','scripts/control_spatial_vsibench.py','configs/spatial135k_output_compat.json','requirements-spatial*.txt']:
        files.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
    hashes={p:sha(ROOT/p) for p in sorted(files)}
    for p in hashes:
        target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,target)
    videos=read(DATA/'videos.json')
    subset=[r for r in videos if (r['dataset'],r['scene_name']) in {(x['dataset'],x['scene_name']) for x in fixed}]
    for row in subset:row['source_sha256']=sha(row['source_video'])
    inherited={}
    if resume_from is not None:
        prior=read(resume_from/'manifest.json')
        assert prior['scope']=='wiring' and prior['inputs']==fixed and prior['config']==cfg
        assert prior['question_ids']==[69,1077]
        for p,h in prior['source_sha256'].items():
            if p not in ('scripts/run_spatial_vsibench.py','scripts/control_spatial_vsibench.py'):
                assert sha(ROOT/p)==h,'not the same frozen Spatial implementation: '+p
        inherited[str(resume_from/'manifest.json')]=sha(resume_from/'manifest.json')
        inherited.update(prior.get('inherited_fresh_spatial_files_sha256',{}))
        for path in resume_from.glob('question_*/*'):
            if path.is_file():inherited[str(path.resolve())]=sha(path)
            elif path.is_dir():
                for f in path.rglob('*'):
                    if f.is_file():inherited[str(f.resolve())]=sha(f)
    write(out/'manifest.json',dict(status='wiring_prepared',config=cfg,inputs=fixed,videos=subset,
        source_sha256=hashes,native_extension_sha256=sha(native/'_C.so'),
        scope='wiring',data_manifest_sha256=sha(DATA/'manifest.json'),question_ids=[69,1077],
        resume_from=str(resume_from) if resume_from else None,inherited_fresh_spatial_files_sha256=inherited,
        no_answers_or_task_labels=True,formal_start_authorized_after_wiring=True,
        method='D116 + first_last_then_oldest_observed',
        execution=dict(screening_logical_batch=64,relevance_microbatch=1,discovery_microbatch=1,
            model_offload='same frozen Spatial weights to CPU during SAM/VGGT, then back to GPU',
            maximum_QA_calls_per_question=2,maximum_control_VGGT_calls=1,
            maximum_observation_VGGT_calls=8,discovery_output_cap=512,
            sparse_QA_output_cap=256,dense_QA_output_cap=64,maximum_images=33,
            maximum_visual_tokens=8192,maximum_joint_text_tokens=32768,
            original_frame_only_text_tokens=4096,structure_limit_bytes=64*2**20),
        warning='No wiring accuracy or blind-test claim. No historical MLLM cache reuse.'))
    write(out/'progress.json',dict(status='prepared',completed=[],formal_started=False))
    print(json.dumps(dict(output=str(out),wiring_ids=[69,1077],model_calls=0)),flush=True)


class InputArchive:
    def __init__(self,root):self.root=Path(root);self.stage='initial';self.ordinal=0
    def __call__(self,texts,batch):
        import torch
        folder=self.root/'actual_inputs'/f'{self.ordinal:05d}_{self.stage}';folder.mkdir(parents=True,exist_ok=False)
        self.ordinal+=1
        with gzip.open(folder/'tensors.pt.gz','wb',compresslevel=1) as f:torch.save(dict(batch),f)
        write(folder/'texts.json',texts)
        write(folder/'receipt.json',dict(stage=self.stage,compressed_tensor_sha256=sha(folder/'tensors.pt.gz')))


def verification_with_original_images(messages,images,candidate):
    """D116 copies serialized messages, never live PIL images. Rebind by index."""
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.map_only import indexed_messages
    from sandbox_vlm.qa.two_pass import verification_messages
    serial=indexed_messages(messages,images,[pixel_sha256(im) for im in images])
    result=verification_messages(serial,candidate)
    for message in result:
        for block in message['content']:
            if block['type']=='image':
                index=block.pop('evidence_image_index')
                assert block.pop('pixel_sha256')==pixel_sha256(images[index])
                block['image']=images[index]
    return result


def resume_verification(item,video,cfg,bundle,folder,prior):
    """Resume one never-executed verifier; no perception or draft model retry."""
    import torch
    from PIL import Image
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.bounded_notes import POLICY
    from sandbox_vlm.qa.d116_full import joint_messages
    from sandbox_vlm.qa.map_only import indexed_messages
    from sandbox_vlm.qa.prompt import messages_for
    from sandbox_vlm.qa.two_pass import candidate_from_generation,decide,eligible
    from sandbox_vlm.memory.storage import cpu_storage_report
    assert read(prior/'input.json')==item and sha(video['source_video'])==video['source_sha256']
    assert not (prior/'verification_generation.json').exists(),'never repeat a verifier'
    # The failed call stopped before serializing the verifier or preparing its
    # tensors. There must be no hidden prior verification forward to repeat.
    assert not list((prior/'actual_inputs').glob('*_verification'))
    frames=read(prior/'image_records.json')['frames'];ids=[f['frame_id'] for f in frames]
    sparse=read(prior/'allocation.json')['sparse'];assert sparse
    payload=read(prior/'payload.json');geometry=read(prior/'geometry.json')
    construction=read(prior/'construction.json');assert eligible(dict(sparse=True),geometry,payload)
    with Image.open(prior/'sandbox.png') as im:diagram=im.convert('RGB').copy()
    images=[]
    for fid in ids:
        with Image.open(prior/f'qa_frame_{fid:06d}.png') as im:images.append(im.convert('RGB').copy())
    assert pixel_sha256(diagram)==read(prior/'image_records.json')['sandbox_pixel_sha256']
    draft_messages=messages_for(item['formatted_question'],images,ids,'brief_evidence')
    draft_serial=indexed_messages(draft_messages,images,[pixel_sha256(im) for im in images])
    assert draft_serial==read(prior/'draft_messages.json'),'candidate inputs changed'
    draft=read(prior/'draft_generation.json');assert draft['status']=='generated'
    candidate=candidate_from_generation(draft)
    messages,joint=joint_messages(item['formatted_question'],True,ids,payload,images,diagram)
    messages=verification_with_original_images(messages,joint,candidate)
    limits=deepcopy(cfg['qa']['budget']);limits.update(max_images=33,max_text_tokens=32768,
        max_image_side=1280 if construction['geometry_repaired'] else 768,
        max_output_tokens=cfg['qa']['sparse_max_output_tokens'])
    archive=InputArchive(folder);archive.stage='verification';bundle.runtime.transport_processor.audit_sink=archive
    serial=indexed_messages(messages,joint,[pixel_sha256(im) for im in joint])
    write(folder/'verification_messages.json',serial)
    started=time.perf_counter()
    checked=bundle.answerer.answer(messages,joint,QABudget(**limits),note_boundary=POLICY)
    write(folder/'verification_generation.json',checked)
    if checked['status'] not in ('generated','answer_failed') or not checked.get('model_calls'):
        raise RuntimeError('QA contract failure: '+str(checked['status']))
    assert any(t['component']=='actual_spatial_encoder' and len(t['shapes'])==len(joint) for t in checked['actual_forward_trace'])
    decision=decide(candidate,checked,ids)
    if decision['status']!='ok':raise RuntimeError('QA contract failure: no valid two-pass answer')
    write(folder/'decision.json',decision)
    source_discoveries=[read(f) for f in prior.glob('observation_*/discovery.json')]
    sam_prompts=sum(read(f).get('box_prompt_calls',0) for f in prior.glob('observation_*/sam.json'))
    sa=read(prior/'sam_correspondences.json')
    sam_prompts+=sa['legacy']['summary']['prompt_attempts']+sa['coverage']['summary']['prompt_attempts']
    calls=[read(f) for f in (prior/'geometry_calls').glob('*.json')]
    state=read(prior/'state.json');screen=read(prior/'screening.json')
    counters=dict(coverage_images=sum(len(r['frame_ids']) for r in screen),
        relevance_logical_requests=sum(len(r['calls']) for r in screen),
        discovery_calls=len(source_discoveries),discovery_failures=sum(r['status']!='ok' for r in source_discoveries),
        SAM_prompts=sam_prompts,VGGT_calls=len(calls),VGGT_frames=sum(len(r['frame_ids']) for r in calls),
        QA_calls=2,new_QA_calls_this_resume=checked.get('model_calls',0),reused_draft_QA_calls=1,
        perception_calls_this_resume=0)
    structure=cpu_storage_report(dict(state=state,payload=payload,geometry=geometry,ledger=read(prior/'ledger.json')))
    assert structure['cpu_structural_bytes']<=64*2**20
    for image in joint:image.close()
    bundle.runtime.transport_processor.audit_sink=None
    (folder/'prior_evidence').symlink_to(prior.resolve(),target_is_directory=True)
    write(folder/'reuse.json',dict(source=str(prior),scope='all fresh Spatial perception and existing draft; verifier was never started',
        actual_draft_input_exact=True,source_hashes='sealed by root manifest',new_perception_calls=0,new_draft_calls=0))
    result=dict(question_id=item['id'],status='ok',raw_output=decision['prediction'],prediction=decision['prediction'],
        full_spatial_perception=True,old_mllm_cache_used=False,actual_sandbox_in_model=True,
        source_video_sha256=video['source_sha256'],actual_frame_ids=ids,query_watermark=video['sampled_watermark'],
        sparse=True,two_pass=True,selected_records=len(payload['records']),instance_records=len(state['instances']),
        costs=counters,structure=structure,resume_verification_seconds=time.perf_counter()-started,
        prior_stage_costs=str(prior),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),scores_used=False)
    write(folder/'result.json',result);return result


def fresh_question(item,video,cfg,bundle,folder,native,resume_from=None):
    import numpy as np
    import torch
    from sandbox_vlm.data.video import FrameMapping,FrameRef
    from sandbox_vlm.runtime import VideoFrameReader
    from sandbox_vlm.perception.parallel_decode import CoverageReader
    from sandbox_vlm.perception.coverage import NoveltyShortlist,coverage_batches,pack_descriptor,unpack_descriptor
    from sandbox_vlm.perception.coverage_budget import allocate_discovery
    from sandbox_vlm.memory.coverage import CoverageSandbox,ContinuitySettings
    from sandbox_vlm.qa.coverage_evidence import temporal_answer_frames
    from sandbox_vlm.memory.storage import cpu_storage_report
    assert sha(video['source_video'])==video['source_sha256'],'video changed'
    mapping=FrameMapping(video_id=video['dataset']+'/'+video['scene_name'],fps=video['source_fps'],sample_fps=1.,
        clip_len=16,overlap=4,frames=[FrameRef(**r) for r in video['frame_mapping']])
    watermark=len(mapping.frames)-1;question=item['formatted_question'];settings=cfg['coverage']
    reader=VideoFrameReader(video['source_video'],mapping);reader.allowed_through=watermark
    mapping.save(folder/'mapping.json');write(folder/'input.json',item)
    archive=InputArchive(folder);bundle.runtime.transport_processor.audit_sink=archive
    counters=dict(coverage_images=0,relevance_logical_requests=0,relevance_forward_calls=0,
        discovery_calls=0,discovery_images=0,discovery_failures=0,SAM_prompts=0,VGGT_calls=0,VGGT_frames=0,QA_calls=0,
        reused_relevance_requests=0,reused_discovery_requests=0,inherited_coverage_images=0)
    prior=(Path(resume_from)/f'question_{item["id"]}') if resume_from else None
    if prior and equivalent_cached_execution(prior) and (prior/'draft_generation.json').exists() and (prior/'geometry.json').exists():
        return resume_verification(item,video,cfg,bundle,folder,prior)
    old_screen=read(prior/'screening.json') if prior and (prior/'screening.json').exists() else []
    if old_screen:
        assert read(prior/'input.json')==item and read(prior/'mapping.json')==mapping.to_dict()
        assert [f for row in old_screen for f in row['frame_ids']]==list(range(watermark+1)),'incomplete old coverage cannot be reused'
        old_towers=sorted((prior/'actual_inputs').glob('*_coverage_tower'))
        assert len(old_towers)==len(old_screen)
        counters['inherited_coverage_images']=watermark+1
    cached_ledger=read(prior/'ledger.json') if old_screen and (prior/'ledger.json').exists() else None
    started=time.perf_counter();ledger={};prefer_full=False
    shortlist=NoveltyShortlist(block_size=settings['block_size'],candidates_per_block=settings['candidates_per_block'])
    scan=CoverageReader(video['source_video'],mapping,workers=settings['decode_workers'])
    screening=[]
    try:
        if cached_ledger is not None:
            ledger={int(k):v for k,v in cached_ledger.items()}
            assert set(ledger)=={fid for row in old_screen for fid in row['selected_frame_ids']}
            counters['reused_relevance_requests']=sum(len(row['calls']) for row in old_screen)
            counters['reused_coverage_images']=watermark+1
            screening=old_screen;write(folder/'screening.json',screening)
        remaining_ids=[] if cached_ledger is not None else list(range(watermark+1))
        for block,(ids,images,prepared,decode_seconds) in enumerate(coverage_batches(scan,bundle.coverage,remaining_ids,
                query=watermark,batch_size=cfg['relevance']['batch_size'],image_side=cfg['relevance']['image_side'])):
            if old_screen:
                assert ids==old_screen[block]['frame_ids']
                with gzip.open(old_towers[block]/'tensors.pt.gz','rb') as f:old_inputs=torch.load(f,map_location='cpu',weights_only=True)
                assert set(prepared[0])==set(old_inputs)
                for key in old_inputs:assert torch.equal(prepared[0][key],old_inputs[key]),'different actual coverage input: '+key
                del old_inputs
            archive.stage='coverage_tower';archive([],prepared[0])
            archive.stage='screening';encoded=bundle.coverage.encode(prepared)
            indices,novelty=shortlist.select(ids,encoded['descriptors'],query=watermark)
            calls=[]
            def score(selected):
                values=[]
                for i in selected:
                    with bundle.coverage.reuse_pixels(encoded,[i],[images[i]]):prepared_prompt=bundle.relevance.prepare(question,[images[i]])
                    with bundle.runtime.trace_media() as trace:
                        with bundle.coverage.reuse(encoded,[i]):margins,stats=bundle.relevance.forward(prepared_prompt)
                    assert any(t['component']=='actual_spatial_encoder' for t in trace)
                    values.extend(margins);calls.append(dict(frame_id=ids[i],stats=stats,trace=trace))
                    counters['relevance_logical_requests']+=1;counters['relevance_forward_calls']+=1
                return values
            if old_screen:
                previous=old_screen[block];old_ids=previous['selected_frame_ids']
                assert {ids[i] for i in indices}<=set(old_ids),'recomputed visual shortlist differs from prior language coverage'
                indices=[ids.index(fid) for fid in old_ids];values=previous['margins'];refined=previous['weak_batch_full_check']
                counters['reused_relevance_requests']+=len(previous['calls'])
                calls=[dict(reused_from=str(prior/'screening.json'),previous_forward_calls=len(previous['calls']))]
            else:
                if prefer_full:indices=list(range(len(ids)))
                values=score(indices)
                refined=len(indices)<len(ids) and max(values)<=settings['weak_margin_full_check']
                if refined:indices=list(range(len(ids)));values=score(indices)
            prefer_full=max(values)<=settings['weak_margin_full_check']
            for i,value in zip(indices,values):ledger[ids[i]]=dict(margin=float(value),
                descriptor=pack_descriptor(encoded['descriptors'][i],quantized=settings['quantized_ledger']),
                novelty=novelty[i],selection='fixed_block_coverage_diversity_and_peak_burst')
            counters['coverage_images']+=len(ids)
            screening.append(dict(frame_ids=ids,selected_frame_ids=[ids[i] for i in indices],margins=values,
                weak_batch_full_check=refined,decode_seconds=decode_seconds,vision=encoded['stats'],calls=calls))
            write(folder/'screening.json',screening)
            for fid,image in zip(ids,images):image.save(folder/f'screen_frame_{fid:06d}.png');image.close()
            del encoded,prepared,images
    finally:scan.close()
    new_ids,allocation=allocate_discovery(ledger,query=watermark,total_frames=len(mapping.frames),
        selected=[],previous_query=-1,**settings['discovery'])
    sparse=allocation['sparse'];write(folder/'allocation.json',allocation)
    # Durable stage checkpoint. Archives never restore Qwen semantic state.
    write(folder/'ledger.json',ledger)
    discoveries=[]
    for fid in new_ids:
        sub=folder/f'observation_F{fid:06d}';sub.mkdir()
        data,ref,hw=reader.encoded_source(fid);path=sub/f'{fid:06d}.jpg';path.write_bytes(data)
        archive.stage='discovery'
        prior_result=prior/sub.name/'discovery.json' if prior else None
        if prior_result and prior_result.exists():
            assert sha(path)==sha(prior/sub.name/path.name),'different discovery pixels'
            result=read(prior_result);counters['reused_discovery_requests']+=1
            write(sub/'reuse.json',dict(source=str(prior_result),sha256=sha(prior_result),
                actual_inputs=str(prior/'actual_inputs'),failed_outputs_reused_as_failure=True,new_model_calls=0))
        else:
            abandoned=(prior/sub.name/path.name) if prior else None
            old_error=read(prior/'failure.json').get('error','') if prior and (prior/'failure.json').exists() else ''
            if abandoned and abandoned.exists() and 'prefix_allowed_tokens_fn' in old_error and 'empty list' in old_error:
                # The old worker stopped at this frame before writing a result.
                # Preserve it as a failed request, never generate it again.
                assert sha(path)==sha(abandoned)
                result=dict(status='perception_failed',frame_id=fid,detections=[],error=old_error,
                    raw_output=None,output_tokens=None,partial_output_unavailable=True,
                    model_calls=1,reused_failed_invocation=True,retry=False)
                counters['reused_discovery_requests']+=1
                write(sub/'reuse.json',dict(source=str(prior/'failure.json'),new_model_calls=0,
                    original_partial_output_unavailable=True,failed_invocation_preserved=True))
            else:
                result=discovery_with_failure_receipt(bundle,path,fid,question)
                counters['discovery_calls']+=1;counters['discovery_images']+=1
        write(sub/'discovery.json',result)
        if result['status']!='ok':
            # Same D043/D065 perception contract: failed discovery contributes
            # no object. Keep the failure; do not repair, retry, or kill the QA.
            counters['discovery_failures']+=1
            continue
        discoveries.append((fid,path,result))
    # Keep only the compact original coverage descriptors across model stages.
    del screening
    gc.collect();bundle.runtime.model.to('cpu');torch.cuda.empty_cache()
    import sam2
    if str(native) not in sam2.__path__:sam2.__path__.append(str(native))
    from sandbox_vlm.perception.sam_tracks import SamCandidateTracker
    from sandbox_vlm.perception.short_track import BoundedShortTrack
    from sandbox_vlm.perception.coverage_short_track import CoverageShortTrack
    from sandbox_vlm.memory.joint_geometry import observations,merge_correspondences
    from sandbox_vlm.memory.retention import observation_id
    from sandbox_vlm.qa.d065_spatial import instances_only,joint_frame_plan,reconstruct
    from sandbox_vlm.perception.vggt import VggtGeometry
    from sandbox_vlm.qa.observation_sandbox import local_window,observation_packet
    tracker=SamCandidateTracker(cfg['perception']['sam'])
    state=CoverageSandbox(mapping.video_id,question,retain_ambiguities=False,
        memory_limit_bytes=cfg['memory']['backend_max_bytes'])
    catalog={};associations=[]
    for fid,path,result in discoveries:
        if not result['detections']:
            write(path.parent/'sam.json',dict(status='no_prompt_needed',box_prompt_calls=0))
            continue
        tracks,sam_report=tracker.track_clip(path.parent,[fid],result['detections'],clip_id=fid,
            iou_threshold=cfg['perception']['sam']['local_match_iou'])
        write(path.parent/'sam.json',sam_report);counters['SAM_prompts']+=sam_report['box_prompt_calls']
        for track in tracks:
            if track.representative_feature is None:continue
            seed=track.seed_detection
            ev=dict(frame_id=fid,bbox_xyxy=list(seed['bbox_xyxy']),source_hw=result['image_size_hw'],
                sam_mask_nonempty=bool(track.masks[fid].any()))
            decision=state.ingest(video_id=mapping.video_id,name=seed['name'],description=seed['short_description'],
                evidence=ev,appearance=track.representative_feature.to_dict(),
                context_descriptor=unpack_descriptor(ledger[fid]['descriptor']),
                settings=ContinuitySettings(**settings['continuity']),allowed_through=watermark)
            associations.append(dict(frame_id=fid,**decision))
            if decision['status']=='duplicate_actual_observation':continue
            link=next((l for l in reversed(state.viewpoints.get(str(fid),[])) if l['evidence']==ev),None)
            if link is None:continue # capacity eviction is authoritative
            oid=observation_id(fid,link,0);p=path.parent/(oid+'.npz')
            np.savez_compressed(p,mask=track.masks[fid]);catalog[oid]=dict(path=str(p),key='mask',available_through=watermark)
        del tracks
    write(folder/'associations.json',associations)
    # D053/D065 selects source frames from the base before query-local unions.
    from sandbox_vlm.qa.question_local import select_base_frames
    chosen=select_base_frames(state,ledger,new_ids,sparse=sparse,config=cfg)
    # Original bounded SAM support, then the existing D102 coverage extension.
    legacy=BoundedShortTrack(tracker);_,legacy_audit=legacy.update(state,reader,watermark=watermark,audit_dir=folder/'legacy_sam')
    coverage=CoverageShortTrack(tracker)
    coverage.import_legacy(state,legacy.to_dict(),legacy_audit['clips'],watermark=watermark)
    _,coverage_audit=coverage.update(state,reader,watermark=watermark,audit_dir=folder/'coverage_sam')
    counters['SAM_prompts']+=legacy_audit['summary']['prompt_attempts']+coverage_audit['summary']['prompt_attempts']
    state,union=d116_identity_and_correspondences(state,legacy.accepted_edges,coverage.new_edges,watermark=watermark)
    write(folder/'sam_correspondences.json',dict(legacy=legacy_audit,coverage=coverage_audit,union=union))
    del coverage,legacy,tracker;gc.collect();torch.cuda.empty_cache()
    state=instances_only(state)
    from sandbox_vlm.memory.retention import fit_instance_memory
    state,aux,retention=fit_instance_memory(state,cfg['memory']['backend_max_bytes'],auxiliary_state=dict(
        gated_frame_ledger=ledger,mask_catalog=catalog,accepted_edges=union['edges'],
        discovered_frame_bank=new_ids,frame_mapping=mapping.to_dict(),last_shortlist_state=shortlist.to_dict()))
    write(folder/'retention.json',retention)
    if retention['status']!='ok':raise MemoryError('complete perception structure exceeds capacity')
    catalog=aux['mask_catalog'];union['edges']=aux['accepted_edges'];del aux
    current_obs={r['observation_id']:r for r in observations(state)}
    catalog={k:v for k,v in catalog.items() if k in current_obs}
    write(folder/'state.json',state.to_dict());write(folder/'ledger.json',ledger)
    # D116 preserves all original answer frames; the map has its additional slot.
    _,geometry_ids,frame_plan=joint_frame_plan(state,chosen,catalog,watermark=watermark,allow_scene_only=True)
    if frame_plan['added_for_actual_geometry']:raise RuntimeError('original answer frames lack SAM support')
    geometry_model=VggtGeometry(cfg['perception']['vggt'])
    def infer(ids,reference):
        with tempfile.TemporaryDirectory(prefix='spatial-geometry-') as temp:
            paths=[];sources=[]
            for data,ref,hw in reader.encoded_sequence(ids):
                p=Path(temp)/f'{ref.frame_id:06d}.jpg';p.write_bytes(data);paths.append(p)
                sources.append(dict(frame_id=ref.frame_id,source_index=ref.source_index,jpeg_sha256=hashlib.sha256(data).hexdigest()))
            raw=geometry_model.infer(paths,ids,reference=reference)
        counters['VGGT_calls']+=1;counters['VGGT_frames']+=len(ids)
        write(folder/'geometry_calls'/f'{counters["VGGT_calls"]:02d}.json',dict(reference=reference,frame_ids=ids,sources=sources,cost=raw['cost']))
        return raw
    raw=infer(geometry_ids,f'VSI/{item["id"]}/control')
    geometry,audit=reconstruct(raw,state,catalog,geometry_ids,include_scene_extent=True)
    frozen=dict(sandbox=geometry.to_dict(),audit=audit,cost=raw['cost']);del raw,geometry
    packets=[]
    if sparse:
        for fid in chosen:
            rows=[r for r in current_obs.values() if r['evidence']['frame_id']==fid]
            if not rows:continue
            masks={}
            for row in rows:
                source=catalog[row['observation_id']]
                with np.load(source['path'],allow_pickle=False) as z:masks[row['observation_id']]=z[source['key']].copy()
            raw=infer(local_window(fid,watermark),f'VSI/{item["id"]}/F{fid}')
            packets.append(observation_packet(raw,fid,rows,masks));del raw,masks
    del geometry_model;gc.collect();torch.cuda.empty_cache()
    from sandbox_vlm.qa.d116_full import compose,draw,joint_messages
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.map_only import indexed_messages
    from sandbox_vlm.qa.prompt import messages_for
    from sandbox_vlm.qa.two_pass import eligible,candidate_from_generation,verification_messages,decide
    from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
    from sandbox_vlm.qa.bounded_notes import POLICY
    payload,geometry,construction=compose(state.to_dict(),frozen,packets,union['edges'],chosen,watermark,item['question'])
    write(folder/'geometry.json',geometry);write(folder/'payload.json',payload);write(folder/'construction.json',construction)
    structure=cpu_storage_report(dict(state=state,ledger=ledger,catalog=catalog,payload=payload,geometry=geometry,
        frozen=frozen,packets=packets,mapping=mapping,shortlist=shortlist))
    if structure['cpu_structural_bytes']>64*2**20:raise MemoryError('complete live structure over64MiB')
    diagram=draw(geometry);diagram.save(folder/'sandbox.png')
    pairs=[reader.read(fid,max_side=cfg['qa']['sparse_image_side'] if sparse else cfg['qa']['image_side']) for fid in chosen]
    images=[p[0] for p in pairs]
    for fid,image in zip(chosen,images):image.save(folder/f'qa_frame_{fid:06d}.png')
    messages,joint=joint_messages(question,sparse,chosen,payload,images,diagram)
    write(folder/'image_records.json',dict(frames=[p[1] for p in pairs],sandbox_pixel_sha256=pixel_sha256(diagram)))
    bundle.runtime.model.to(bundle.runtime.device);torch.cuda.empty_cache()
    limits=deepcopy(cfg['qa']['budget']);limits.update(max_images=33,max_text_tokens=32768,
        max_image_side=1280 if construction['geometry_repaired'] else 768,
        max_output_tokens=cfg['qa']['sparse_max_output_tokens'] if sparse else cfg['qa']['budget']['max_output_tokens'])
    two=eligible(dict(sparse=sparse),geometry,payload)
    def answer(stage,msg,imgs,lim,note):
        archive.stage=stage
        hashes=[pixel_sha256(i) for i in imgs]
        serialized=indexed_messages(msg,imgs,hashes)
        write(folder/(stage+'_messages.json'),serialized)
        if prior and (prior/(stage+'_generation.json')).exists() and (prior/(stage+'_messages.json')).exists():
            if read(prior/(stage+'_messages.json'))==serialized:
                cached=read(prior/(stage+'_generation.json'))
                write(folder/(stage+'_generation.json'),cached)
                write(folder/(stage+'_reuse.json'),dict(source=str(prior),exact_serialized_input=True,new_model_calls=0))
                counters['reused_QA_calls']=counters.get('reused_QA_calls',0)+cached.get('model_calls',0)
                return cached
        result=bundle.answerer.answer(msg,imgs,QABudget(**lim),**({'note_boundary':POLICY} if note else {}))
        counters['QA_calls']+=result.get('model_calls',0)
        write(folder/(stage+'_generation.json'),result)
        if result['status'] not in ('generated','answer_failed') or not result.get('model_calls'):
            raise RuntimeError('QA contract failure: '+str(result['status']))
        assert any(t['component']=='actual_spatial_encoder' and len(t['shapes'])==len(imgs)
            for t in result['actual_forward_trace']),'not all supplied images reached the spatial encoder'
        return result
    if two:
        draft_messages=messages_for(question,images,chosen,'brief_evidence')
        dl=deepcopy(cfg['qa']['budget']);dl['max_output_tokens']=cfg['qa']['sparse_max_output_tokens']
        draft=answer('draft',draft_messages,images,dl,False);candidate=candidate_from_generation(draft)
        checked=answer('verification',verification_with_original_images(messages,joint,candidate),joint,limits,True)
        decision=decide(candidate,checked,chosen);write(folder/'decision.json',decision)
        prediction=decision['prediction']
        if decision['status']!='ok':raise RuntimeError('QA contract failure: no valid two-pass answer')
    else:
        checked=answer('joint',messages,joint,limits,sparse)
        if checked['status']!='generated':raise RuntimeError('QA contract failure: '+str(checked['status']))
        prediction=extract_d065_answer(checked['raw_output'],sparse=sparse)
    for im in joint:im.close()
    bundle.runtime.transport_processor.audit_sink=None
    result=dict(question_id=item['id'],status='ok',raw_output=prediction,prediction=prediction,
        full_spatial_perception=True,old_mllm_cache_used=False,source_video_sha256=video['source_sha256'],
        actual_frame_ids=chosen,query_watermark=watermark,sparse=sparse,two_pass=two,
        actual_sandbox_in_model=True,selected_records=len(payload['records']),
        instance_records=len(state.instances),costs=counters,structure=structure,
        wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),scores_used=False)
    write(folder/'result.json',result);return result


def run(out):
    import cv2,torch
    from sandbox_vlm.backbones.spatial_bundle import create_spatial_backbone
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    m=read(out/'manifest.json')
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in m.get('inherited_fresh_spatial_files_sha256',{}).items():assert sha(name)==h,name
    assert sha(out/'native_sam/_C.so')==m['native_extension_sha256']
    formal=m.get('scope')=='formal'
    attempt=out/('session_'+os.environ['SLURM_JOB_ID']+'.json' if formal else 'gpu_attempt.json')
    if attempt.exists():raise FileExistsError('preserve prior attempt; no automatic model retry')
    write(attempt,dict(job_id=os.environ['SLURM_JOB_ID'],device=torch.cuda.get_device_name(),started=time.time()))
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(0)
    bundle=create_spatial_backbone(m['config'],ROOT/'work/spatial-mllm-official')
    results=[];stop=[False];session_start=time.monotonic()
    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    try:
        for item in m['inputs']:
            folder=out/f'question_{item["id"]}'
            if formal and (folder/'result.json').exists():
                results.append(read(folder/'result.json'));continue
            prior=Path(m['resume_from'])/folder.name if m.get('resume_from') else None
            if prior and equivalent_cached_execution(prior) and (prior/'result.json').exists():
                cached=read(prior/'result.json')
                assert read(prior/'input.json')==item and cached['status']=='ok' and cached['full_spatial_perception'] and not cached['old_mllm_cache_used']
                folder.symlink_to(prior.resolve(),target_is_directory=True)
                results.append(cached)
                print('REUSED_COMPLETE',item['id'],flush=True)
                continue
            if formal and (stop[0] or time.monotonic()-session_start>=9*3600):break
            if formal and folder.exists():
                # A previously interrupted request is visible and never repeated.
                result=dict(question_id=item['id'],status='interrupted',raw_output='',error='prior incomplete attempt preserved without model retry')
                write(folder/'result.json',result);results.append(result);continue
            folder.mkdir(exist_ok=False)
            video=next(v for v in m['videos'] if (v['dataset'],v['scene_name'])==(item['dataset'],item['scene_name']))
            print('START',item['id'],flush=True)
            try:
                result=fresh_question(item,video,m['config'],bundle,folder,out/'native_sam',m.get('resume_from'))
            except Exception as error:
                write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
                if not formal:raise
                expected_QA_failure=(isinstance(error,RuntimeError) and str(error).startswith('QA contract failure:')) or (
                    isinstance(error,ValueError) and str(error) in ('missing exact final line; no repair','empty answer'))
                if not expected_QA_failure:
                    raise # stop on an unclassified engineering error; never score it silently
                # Preserve full scoring denominator, do not invent or retry answers.
                result=dict(question_id=item['id'],status='failed',raw_output='',error=str(error))
                write(folder/'result.json',result)
                # A failed stage might have left the shared weights on the CPU.
                bundle.runtime.transport_processor.audit_sink=None
                bundle.runtime.model.to(bundle.runtime.device);gc.collect();torch.cuda.empty_cache()
            results.append(result);write(out/'progress.json',dict(status='running',completed=[r['question_id'] for r in results]))
            print(result['status'].upper(),item['id'],flush=True)
        if formal:
            (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
            complete=len(results)==5130
            write(out/'progress.json',dict(status='completed' if complete else 'continuation_required',
                completed=[r['question_id'] for r in results],formal_started=True,session_job=os.environ['SLURM_JOB_ID']))
            if complete:write(out/'completion.json',dict(status='completed',questions=5130,manifest_sha256=sha(out/'manifest.json')))
        else:
            assert [r['question_id'] for r in results]==[69,1077]
            write(out/'completion.json',dict(status='passed',question_ids=[69,1077],results=results,
                formal_scores=None,source_sha256=m['source_sha256'],manifest_sha256=sha(out/'manifest.json')))
    except Exception as e:
        write(out/'failure.json',dict(status='failed',completed=[r['question_id'] for r in results],error=str(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve(),a.resume_from.resolve() if a.resume_from else None)
    else:run(a.output.resolve())
