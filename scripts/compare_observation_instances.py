"""D094: full frozen 346 replay, two genuine QA generations, no new perception."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import zipfile

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
LIMIT=64*1024**2


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def encoded(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def replay(source,spatial):
    from sandbox_vlm.memory import Sandbox,CameraParams
    from sandbox_vlm.alignment.association import MatchSettings
    from sandbox_vlm.memory.association import associate_observation
    from sandbox_vlm.perception.observations import LocalObservation
    from sandbox_vlm.perception.image_geometry import vggt_grid_transforms
    from sandbox_vlm.memory.storage import cpu_storage_report
    m=read(source/'manifest.json');cfg=m['config'];state=Sandbox(video_id=m['example']['video_path'])
    settings=MatchSettings(**cfg['alignment']['match'])
    if spatial:settings=replace(settings,birth_policy='spatial_separation')
    decisions={};rows=[];peak=0
    for folder in sorted((source/'inference').glob('clip_*')):
        report=read(folder/'report.json');raw=read(folder/'observations.json')
        cameras={c['frame_id']:CameraParams.from_dict(c) for c in read(folder/'cameras.json')}
        shapes={p['frame_id']:tuple(p['source_hw']) for o in raw for p in o['prompt_evidence']}
        grids={f:g for f,g in zip(shapes,vggt_grid_transforms(list(shapes.values())))}
        transient={};current=[]
        for value in raw:
            obs=LocalObservation.from_dict(value)
            result=associate_observation(state,obs,cameras,grids,settings,
                alignment_reliable=report['alignment_reliable'],current_clip_evidence=transient)
            decisions[obs.local_observation_id]=dict(result)
            current.append(dict(observation_id=obs.local_observation_id,**result))
        size=cpu_storage_report([state,transient,cameras,grids,raw,decisions])['cpu_structural_bytes']
        peak=max(peak,size)
        if size>LIMIT:raise ValueError('CPU replay complete structure exceeded 64MiB')
        rows.append(dict(clip_id=report['clip_id'],alignment_reliable=report['alignment_reliable'],
                         cpu_working_bytes=size,outcomes=current))
    if state.validate():raise ValueError('replay state invalid')
    return state,decisions,dict(clips=len(rows),rows=rows,cpu_replay_peak_bytes=peak)


def select_common(source,question):
    from sandbox_vlm.qa.selection import terms
    catalog={};frames=set()
    for folder in sorted((source/'inference').glob('clip_*')):
        for o in read(folder/'observations.json'):
            ps=[p for p in o['prompt_evidence'] if p['sam_mask_nonempty']]
            frames.update(p['frame_id'] for p in ps)
            catalog[o['local_observation_id']]=dict(name=o['name'],description=o['short_description'],
                prompt_evidence=ps,score=len(terms(question)&terms(o['name']+' '+o['short_description'])))
    frames=sorted(frames);n=min(31,len(frames))
    chosen=[frames[(i*(len(frames)-1)*2+(n-1))//(2*(n-1))] for i in range(n)] if n>1 else frames
    assert len(set(chosen))==len(chosen)
    selected=[]
    for quarter in range(4):
        pool=[]
        for oid,row in catalog.items():
            actual=sorted({p['frame_id'] for p in row['prompt_evidence']} & set(chosen))
            if not actual or row['score']<=0:continue
            # Each observation gets exactly one quarter, preventing double slots.
            anchor=actual[0];q=min(3,anchor*4//744)
            if q==quarter:pool.append((-row['score'],anchor,oid))
        selected.extend(x[2] for x in sorted(pool)[:6])
    return dict(frame_ids=chosen,observation_ids=selected,catalog=catalog,
        rule='31 quantiles of all frozen actual prompt frame IDs; six lexically relevant observations per temporal quarter',
        quarter_frame_counts=[sum(min(3,f*4//744)==q for f in chosen) for q in range(4)])


def render_input_map(sandbox,selection,rows):
    from PIL import Image,ImageDraw,ImageFont
    from sandbox_vlm.qa.render import render_sandbox
    image,metadata=render_sandbox(sandbox,selection,max_side=768)
    canvas=Image.new('RGB',(768,576),'white');image.thumbnail((500,430))
    canvas.paste(image,(0,65));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
    draw.text((8,8),'D094 | left: confirmed 3D | right: actual observation / instance directory',fill='black',font=font)
    draw.text((8,25),'Candidate/pending entries may repeat one object. Directory layout is not geometry.',fill='black',font=font)
    draw.line((501,50,501,560),fill='grey')
    for i,row in enumerate(rows):
        label=f"{row['display_id']} {row['name']} F"+'/'.join(map(str,row['frames'][:3]))
        draw.text((507,57+i*20),label[:35],fill='black',font=font)
    metadata.update(image_size_hw=[576,768],directory_rows=rows,
                    directory_layout_has_physical_meaning=False,candidate_boxes_not_drawn_in_global_reference=True)
    return canvas,metadata


def build_input(sandbox,registry,decisions,common,question,source_images,source_records):
    from sandbox_vlm.memory.relaxed import candidate_frames
    from sandbox_vlm.qa.selection import EvidenceSelection
    from sandbox_vlm.qa.input import answer_messages
    from sandbox_vlm.qa.result import ANSWER_SCHEMA
    from sandbox_vlm.qa.render import SPACE_NOTICE
    selected=common['observation_ids'];frames=common['frame_ids'];gids=set();pending=[];display=[]
    by_candidate={r['instance_id']:r for r in registry.get('records',[])}
    by_pending={o.local_observation_id:o for o in sandbox.pending_observations}
    for oid in selected:
        row=common['catalog'][oid];actual=sorted({p['frame_id'] for p in row['prompt_evidence']} & set(frames))
        decision=decisions[oid]
        gid=decision.get('matched_instance_id') if decision['status'] in ('created','matched') else registry.get('source_to_candidate',{}).get(oid)
        if gid is not None:
            if gid in gids:continue
            gids.add(gid);display.append(dict(display_id=('C' if gid in by_candidate else 'I')+str(gid),name=row['name'],frames=actual))
        elif oid in by_pending:
            pending.append(oid);display.append(dict(display_id=oid,name=row['name'],frames=actual))
    confirmed=sorted(gids&sandbox.instance_ids())
    selection=EvidenceSelection(question,confirmed,pending,frames,[],False,{},[],max_targets=24)
    canvas,render=render_input_map(sandbox,selection,display)
    legend=[]
    for gid in sorted(gids):
        if gid in by_candidate:
            r=by_candidate[gid];support=sorted(set(candidate_frames(registry,gid))&set(frames))
            legend.append(dict(instance_id=gid,name=r['name'],description=r['description'],
                identity_status=r['identity_status'],coordinate_reference=None,
                selected_actual_frame_ids=support))
        else:
            r=sandbox.get_instance(gid)
            legend.append(dict(instance_id=gid,name=r.name,description=r.short_description,
                identity_status='confirmed_by_existing_geometric_rules',coordinate_reference=sandbox.reference_of(gid),
                selected_actual_frame_ids=sorted(set(sandbox.frames_for_instance(gid))&set(frames))))
    notes={n.local_observation_id:n for n in sandbox.ambiguity_notes}
    unresolved=[dict(local_observation_id=oid,name=by_pending[oid].name,description=by_pending[oid].short_description,
                    coordinate_reference=by_pending[oid].coordinate_reference,
                    selected_actual_prompt_frame_ids=sorted({p['frame_id'] for p in by_pending[oid].prompt_evidence}&set(frames)),
                    candidate_instance_ids=[g for g in notes[oid].candidate_instance_ids if g in gids],
                    reason=notes[oid].reason) for oid in pending]
    images=[canvas,*source_images]
    metadata=dict(original_question=question,selection=dict(selection.to_dict(),instance_ids=sorted(gids)),
        render=render,image_records=[dict(kind='sandbox',input_image_size_hw=[576,768]),*source_records],
        legend=legend,unresolved_notes=unresolved,memory_has_unresolved_observations=bool(sandbox.pending_observations or registry.get('records')),
        memory_may_be_incomplete=True,scale_is_metric=False,space_notice=SPACE_NOTICE,
        model_selection_limitations=dict(max_structure_targets=24,displayed_observation_basis=len(selected),
            omitted_observation_basis=len(common['catalog'])-len(selected),
            notice='Instance IDs marked candidate or operational identity unconfirmed are hypotheses and may repeat one physical object. Do not treat ID totals as the answer. Only confirmed boxes share their stated reference. Directory layout expresses no spatial relation. All selected IDs have actual image support.'))
    messages=answer_messages(images,metadata,output_schema=ANSWER_SCHEMA,citation_policy='implicit')
    return images,metadata,messages


class CitationState:
    def __init__(self,metadata):self.links={r['instance_id']:r['selected_actual_frame_ids'] for r in metadata['legend']}
    def frames_for_instance(self,gid):return self.links[gid]


def prepare(out,source,frame_cache=None):
    from sandbox_vlm.memory import Sandbox
    from sandbox_vlm.memory.relaxed import candidate_registry
    from sandbox_vlm.memory.storage import cpu_storage_report
    from sandbox_vlm.qa.artifact import state_sha256,save_joint_input,pixel_sha256
    from sandbox_vlm.data.video import FrameMapping
    from sandbox_vlm.runtime import VideoFrameReader
    from sandbox_vlm.qa.budget import QABudget
    from check_proposal_revision import ProcessorOnly
    started=time.perf_counter();m=read(source/'manifest.json');question=m['example']['model_question']
    assert m['original_index']==346 and m['frames']==744 and m['clips']==62
    common=select_common(source,question);write(out/'selection.json',common)
    mapping=FrameMapping.load(source/'inference/frame_mapping.json')
    reader=VideoFrameReader(m['video_path'],mapping);reader.allowed_through=743
    images=[];records=[]
    (out/'source_frames').mkdir()
    for fid in common['frame_ids']:
        if frame_cache is not None:
            from PIL import Image
            cached={r['frame_id']:r for r in read(frame_cache/'records.json')}
            record=deepcopy(cached[fid]);record.pop('kind',None)
            assert record['source_index']==mapping.by_id(fid).source_index
            image=Image.open(frame_cache/f'frame_{fid}.png').convert('RGB')
            assert [image.height,image.width]==record['input_image_size_hw'] and max(image.size)<=384
        else:image,record=reader.read(fid,max_side=384)
        images.append(image);records.append(dict(kind='source_frame',**record))
        image.save(out/'source_frames'/f'frame_{fid}.png')
    write(out/'source_frames/records.json',records)
    cfg=deepcopy(m['config']);cfg['qa']['budget'].update(max_images=32,max_image_side=768,max_output_tokens=1024)
    cfg['qa']['mllm']['output_constraint']='json_unique_citations'
    limits=QABudget(**cfg['qa']['budget']);processor=ProcessorOnly(cfg,True)
    summary={};all_inputs=[]
    for arm,spatial in [('A',False),('B',True)]:
        folder=out/arm;folder.mkdir()
        if spatial:
            from sandbox_vlm.memory.observation_instances import observation_instances
            from sandbox_vlm.perception.observations import LocalObservation
            observations=[];reliability={}
            for clip in sorted((source/'inference').glob('clip_*')):
                reliable=read(clip/'report.json')['alignment_reliable']
                for raw_obs in read(clip/'observations.json'):
                    obs=LocalObservation.from_dict(raw_obs);observations.append(obs)
                    reliability[obs.local_observation_id]=reliable
            state,registry,decisions=observation_instances(m['example']['video_path'],observations,strict_decisions,reliability)
            peak=cpu_storage_report([state,registry,decisions,strict_decisions,observations,reliability,common])['cpu_structural_bytes']
            if peak>LIMIT:raise ValueError('D094 complete birth working state exceeded 64MiB')
            replay_report=dict(clips=62,local_tracks=len(observations),cpu_replay_peak_bytes=peak,
                numeric_cross_track_fusions=0,exact_prompt_unions=len(registry['source_to_candidate'])-len(registry['records']))
            del observations,reliability,strict_decisions
        else:
            state,decisions,replay_report=replay(source,False)
            strict_decisions=deepcopy(decisions)
        if arm=='A':
            original=Sandbox.from_dict(read(source/'inference/query_000743/base_sandbox.json'))
            assert state_sha256(original)==state_sha256(state),'strict replay differs from D083'
            del original
        if not spatial:registry=dict(records=[],viewpoints={},source_to_candidate={})
        converted=set(registry.get('source_to_candidate',{}))
        if spatial:
            state.pending_observations=[o for o in state.pending_observations if o.local_observation_id not in converted]
            state.ambiguity_notes=[n for n in state.ambiguity_notes if n.local_observation_id not in converted]
        state_payload=dict(sandbox=state.to_dict(),registry=registry)
        write(folder/'state.json',state_payload);write(folder/'replay.json',replay_report);write(folder/'decisions.json',decisions)
        joint,meta,messages=build_input(state,registry,decisions,common,question,images,records)
        assert all(r['selected_actual_frame_ids'] for r in meta['legend'])
        admission=processor.answer(messages,joint,limits)
        assert admission['status']=='cpu_input_verified_no_generation',admission
        body=encoded(state_payload);host=cpu_storage_report([state,registry,decisions,common,meta])['cpu_structural_bytes']
        charged=max(host,cpu_storage_report(state_payload)['cpu_structural_bytes'])+3*len(body)
        if charged>LIMIT:raise ValueError(f'{arm} complete structural upper bound exceeded: {charged}')
        save_joint_input(folder/'prepared_input',joint,meta,messages)
        write(folder/'preflight.json',dict(processor=admission,charged_complete_structure_upper_bound=charged,
            host_working_bytes=host,encoded_snapshot_bytes=len(body),transaction_copies=3,
            frames_pixel_sha256=[pixel_sha256(x) for x in joint[1:]]))
        summary[arm]=dict(confirmed_instances=len(state.instance_ids()),candidate_instances=len(registry['records']),
            pending_observations=len(state.pending_observations),registered_local_tracks=len(converted),
            cpu_replay_peak_bytes=replay_report['cpu_replay_peak_bytes'],structure_upper_bound=charged,
            selected_confirmed_ids=[r['instance_id'] for r in meta['legend'] if r['identity_status'].startswith('confirmed')],
            selected_candidate_ids=[r['instance_id'] for r in meta['legend'] if not r['identity_status'].startswith('confirmed')],
            displayed_pending=len(meta['unresolved_notes']),processor=admission)
        all_inputs.extend([folder/'state.json',folder/'decisions.json',folder/'preflight.json',*sorted((folder/'prepared_input').glob('*'))])
        del state,registry,state_payload,body,joint,meta,messages,decisions,replay_report;gc.collect()
    assert summary['A']['processor']['image_grid_thw']==summary['B']['processor']['image_grid_thw']
    assert read(out/'A/preflight.json')['frames_pixel_sha256']==read(out/'B/preflight.json')['frames_pixel_sha256']
    code=[*sorted((ROOT/'sandbox_vlm').rglob('*.py')),Path(__file__),ROOT/'scripts/check_proposal_revision.py']
    hashes={}
    with zipfile.ZipFile(out/'code_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in code:
            rel=p.relative_to(ROOT);dst=out/'runtime'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
            hashes[str(rel)]=sha(dst);z.write(dst,str(rel))
    (out/'runtime/venv').symlink_to(ROOT/'venv',target_is_directory=True)
    sources=[source/'manifest.json',source/'inference/frame_mapping.json',source/'inference/query_000743/base_sandbox.json',
             *sorted((source/'inference').glob('clip_*/*.json')),
             ROOT/'sandbox_vlm/qa/grammar.py',out/'selection.json',out/'source_frames/records.json',*all_inputs]
    if frame_cache is not None:sources.extend(sorted(frame_cache.glob('*')))
    manifest=dict(purpose='D094_full_video_actual_evidence_birth_QA_pair',source_root=str(source),source_project=str(ROOT),
        config=cfg,example=m['example'],parquet_path=m['parquet_path'],frame_ids=common['frame_ids'],
        shared_pixel_grid_checks_passed=True,arms=summary,maximum_new_QA=2,new_discovery_SAM_VGGT=0,
        program_limit_seconds=780,slurm_limit_seconds=900,backend_max_bytes=LIMIT,
        input_sha256={str(p.resolve()):sha(p) for p in sources},source_sha256=hashes,
        inherited_perception_cost_seconds=read(source/'run_report.json')['program_seconds'],
        cpu_prepare_seconds=time.perf_counter()-started,result_scope='one_exposed_complete_video_frozen_perception_development_QA_pair',
        D091_executed=False,D092_executed=False,targets_loaded=False)
    write(out/'manifest.json',manifest)
    write(out/'user_authorization.json',dict(user_instruction='执行（D094具体提案后）',
        scope='approved D094: full 346 strict A vs actual-evidence births without numeric cross-track fusion B; identical 31 real frames; at most two QA',
        manifest_sha256=sha(out/'manifest.json')))
    print(json.dumps(dict(status='prepared',frames=manifest['frame_ids'],arms=summary),ensure_ascii=False))


def verify(out):
    m=read(out/'manifest.json');assert read(out/'user_authorization.json')['manifest_sha256']==sha(out/'manifest.json')
    assert all(sha(p)==h for p,h in m['input_sha256'].items())
    assert all(sha(out/'runtime'/p)==h for p,h in m['source_sha256'].items())
    return m


def run(out,m):
    import torch
    import numpy as np
    from PIL import Image
    from sandbox_vlm.qa.model import QwenAnswerer
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.input import answer_messages
    from sandbox_vlm.qa.result import ANSWER_SCHEMA,parse_answer
    from sandbox_vlm.qa.grammar import answer_grammar
    from sandbox_vlm.qa.artifact import save_joint_input,pixel_sha256,serial_messages
    from sandbox_vlm.memory.storage import cpu_storage_report
    if (out/'run_report.json').exists():raise FileExistsError('no automatic model rerun')
    assert os.environ.get('SLURM_JOB_ID') and 'A100' in torch.cuda.get_device_name()
    started=time.perf_counter();torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
    report=dict(status='running',slurm_job_id=os.environ['SLURM_JOB_ID'],new_perception_calls=0,arms={},QA_attempts=0)
    write(out/'run_report.json',report)
    try:
        model=QwenAnswerer(m['config']['qa']['mllm']);limits=QABudget(**m['config']['qa']['budget'])
        for arm in ['A','B']:
            folder=out/arm
            if time.perf_counter()-started>m['program_limit_seconds']:raise RuntimeError('program budget before next QA')
            payload=read(folder/'state.json');raw=encoded(payload)
            meta=read(folder/'prepared_input/evidence.json')
            host=cpu_storage_report([payload,meta])['cpu_structural_bytes']
            # Include the host JSON bytes, temporary transfer array and GPU snapshot.
            upper=host+3*len(raw)
            if upper>LIMIT:raise ValueError('GPU-run complete structure upper bound exceeded')
            buffer=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()).cuda()
            images=[Image.open(folder/'prepared_input/sandbox.png').convert('RGB')]
            images += [Image.open(folder/'prepared_input'/f'frame_{fid}.png').convert('RGB') for fid in m['frame_ids']]
            messages=answer_messages(images,meta,output_schema=ANSWER_SCHEMA,citation_policy='implicit')
            assert serial_messages(messages,images)==read(folder/'prepared_input/messages.json')
            assert [pixel_sha256(i) for i in images[1:]]==read(folder/'preflight.json')['frames_pixel_sha256']
            save_joint_input(folder/'actual_input',images,meta,messages)
            generated=model.answer(messages,images,limits,format_constraint=answer_grammar(meta))
            result=dict(status=generated['status'],generation=generated,structure_upper_bound=upper,
                cpu_working_bytes=host,gpu_complete_structure_bytes=buffer.untyped_storage().nbytes())
            if generated['status']=='generated':
                try:
                    parsed=parse_answer(generated['raw_output'],meta,CitationState(meta))
                    result.update(status='ok',parsed_answer=parsed,
                        revision_requests=[dict(request=r,status='pending',reason='pair_diagnostic_no_postanswer_state_writeback') for r in parsed['match_requests']])
                except ValueError as exc:result.update(status='answer_failed',error=str(exc))
            write(folder/'qa_result.json',result)
            write(folder/'prediction.json',dict(status=result['status'],sample_id=m['example']['sample_id'],
                prediction=result.get('parsed_answer',{}).get('answer')))
            report['arms'][arm]=dict(status=result['status'],structure_upper_bound=upper)
            report['QA_attempts']=model.generation_attempts;write(out/'run_report.json',report)
            del payload,raw,buffer,images,messages,meta;gc.collect();torch.cuda.empty_cache()
        assert model.generation_attempts<=2
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        for arm in ['A','B']:
            if not(out/arm/'prediction.json').exists():write(out/arm/'prediction.json',dict(status='not_completed',sample_id=m['example']['sample_id'],prediction=None))
        report.update(QA_attempts=getattr(locals().get('model'),'generation_attempts',0),
            program_seconds=time.perf_counter()-started,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out/'run_report.json',report)
    return 0 if report['status']=='completed' else 1


def score(out,m):
    import pyarrow.parquet as pq
    from sandbox_vlm.evaluation.vsi_super import score_predictions
    if(out/'score.json').exists():raise FileExistsError('preserve scores')
    rows=pq.read_table(m['parquet_path'],filters=[('video_path','=',m['example']['video_path']),('split','=','10mins')]).to_pylist()
    assert len(rows)==1
    result=dict(target=rows[0]['answer'],arms={},scope=m['result_scope'],original_full_benchmark_unchanged=True)
    for arm in ['A','B']:
        prediction=read(out/arm/'prediction.json');answer=prediction['prediction']
        try:valid=math.isfinite(float(answer))
        except (ValueError,TypeError):valid=False
        s=score_predictions([dict(sample_id=prediction['sample_id'],prediction=answer,target_doc=rows[0])],
            task='count',official_repo=Path(m['source_project'])/'vendor/cambrian-s-official') if valid else dict(status='invalid_counted_incorrect',metrics={'Overall':0.0},n_scored=1)
        result['arms'][arm]=dict(prediction=answer,valid_prediction=valid,score=s)
    result['candidate_minus_control_pp']=result['arms']['B']['score']['metrics']['Overall']-result['arms']['A']['score']['metrics']['Overall']
    write(out/'score.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--frame-cache',type=Path)
    p.add_argument('--action',choices=['prepare','verify','run','score'],default='verify');a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out,a.source.resolve(),a.frame_cache.resolve() if a.frame_cache else None)
    else:
        manifest=verify(out)
        if a.action=='run':raise SystemExit(run(out,manifest))
        elif a.action=='score':score(out,manifest)
        else:print('D094_FROZEN_VERIFIED_NO_MODEL')
