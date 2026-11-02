"""D095: fixed identity-first gate, no model retries or automatic QA on failure."""
import argparse
from copy import deepcopy
import gc
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import zipfile

os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
LIMIT=64*1024**2
from sandbox_vlm.memory.identity_evidence import (region_key,parse_unit,unit_images,unit_messages,
 UNIT_PATTERN,mask_iou,unique_candidates,verify_pair,supported_components,pack_mask,unpack_mask)


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def encoded(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def make_tasks(proposal):
 controls=read(proposal/'known_unit_control_inputs.json')['endpoints'];tasks=[]
 for i,row in enumerate(controls):tasks.append(dict(task_id=f'unit_control_{i}',name='ottoman',region=row['evidence'],parent_track=row['observation_id']))
 pairs=read(proposal/'selected_pairs.json')
 for i,pair in enumerate(pairs):
  for side,region,parent in zip(['a','b'],pair['actual_endpoints'],[pair['source_observation_id'],pair['target_observation_id']]):
   tasks.append(dict(task_id=f'pair_{i:02d}_{side}',name=pair['name'],region=region,parent_track=parent))
 for i,control in enumerate(read(proposal/'co_visible_controls.json')):
  for side,region in zip(['a','b'],control['regions']):tasks.append(dict(task_id=f'different_{i}_{side}',name=region['name'],region=region,parent_track=None))
 assert len(tasks)==42 and len(pairs)==16
 for task in tasks:task['region_key']=region_key(task['region'])
 return tasks


def load_unit_images(folder):
 from PIL import Image
 return [Image.open(folder/name).convert('RGB') for name in ['marked.png','crop.png']]


def basis_records(tasks):
 return [dict(region_key=t['region_key'],name=t['name'],parent_track=t['parent_track'],region=t['region'],
              unit_status='not_executed',appearance=None,coordinate_reference=None,
              identity_scope='actual_region_only_not_parent_track') for t in tasks if t['task_id'].startswith('pair_')]


def qa_input(base,regions,units,components,frame_ids,images,records,question,arm):
 from PIL import Image,ImageDraw,ImageFont
 from sandbox_vlm.qa.input import answer_messages
 from sandbox_vlm.qa.result import ANSWER_SCHEMA
 from sandbox_vlm.qa.render import SPACE_NOTICE
 # Duplicate actual region keys are one observation; no whole-track identity inference.
 by_key={r['region_key']:r for r in regions};keys=list(dict.fromkeys(r['region_key'] for r in regions))
 groups=[list(g) for g in components] if arm=='B' else []
 covered={k for group in groups for k in group}
 groups.extend([[key] for key in keys if key not in covered]);groups.sort(key=lambda group:min(keys.index(k) for k in group))
 legend=[];gid_for={}
 for gid,group in enumerate(groups):
  rows=[by_key[k] for k in group];first=rows[0]
  for k in group:gid_for[k]=gid
  # No audit labels, parent features, geometry or hidden record count in QA input.
  legend.append(dict(instance_id=gid,name=first['name'],coordinate_reference=None,
   identity_status='supported_local_identity' if arm=='B' and len(group)>1 else 'observation_identity_unknown',
   unit_status=sorted({units.get(k,'not_executed') for k in group}),
   selected_actual_frame_ids=sorted({r['region']['frame_id'] for r in rows})))
 canvas=Image.new('RGB',(768,576),'white');draw=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
 draw.text((10,8),'Actual regions and local identity links | no verified 3D',fill='black',font=font)
 draw.text((10,27),'Layout is a directory, not physical position. Different O IDs may repeat one object.',fill='black',font=font)
 for i in range(0,len(regions),2):
  left,right=regions[i:i+2];ka,kb=left['region_key'],right['region_key'];ga,gb=gid_for[ka],gid_for[kb]
  label=f"O{ga} {left['name']} F{left['region']['frame_id']}  |  O{gb} {right['name']} F{right['region']['frame_id']}"
  if arm=='B' and ga==gb:label=f"I{ga} {left['name']} : F{left['region']['frame_id']} <-> F{right['region']['frame_id']}"
  draw.text((12,56+(i//2)*30),label[:96],fill='black',font=font)
 meta=dict(original_question=question,selection=dict(question=question,instance_ids=list(range(len(legend))),pending_ids=[],frame_ids=frame_ids,
  lexical_fallback=False,uncovered_targets=[],max_targets=32),legend=legend,unresolved_notes=[],
  render=dict(panels=[],directory_layout_has_physical_meaning=False,verified_3D=False),
  image_records=[dict(kind='sandbox',input_image_size_hw=[576,768]),*records],
  memory_has_unresolved_observations=True,memory_may_be_incomplete=True,scale_is_metric=False,space_notice=SPACE_NOTICE,
  model_selection_limitations=dict(max_structure_targets=32,displayed_actual_regions=len(keys),
   notice='O IDs are actual observations, not distinct confirmed objects. I links support only displayed actual regions, not all observations in a parent SAM track or all video instances. No verified 3D. Diagram layout has no spatial meaning. Other video identities remain unresolved.'))
 joint=[canvas,*images];messages=answer_messages(joint,meta,output_schema=ANSWER_SCHEMA,citation_policy='implicit')
 return joint,meta,messages


def charge_cpu(payload,extra=(),reserve=0):
 from sandbox_vlm.memory.storage import cpu_storage_report
 raw=encoded(payload)
 host=cpu_storage_report([payload,extra])['cpu_structural_bytes']
 return dict(host_working_bytes=host,encoded_bytes=len(raw),transaction_copies=3,
             charged_bytes=host+3*len(raw)+reserve,reserve_bytes=reserve)


def measure_unit(processor,messages,images,limits):
 from sandbox_vlm.qa.unit_budget import check_unit_budget
 inputs=processor.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 grid=inputs['image_grid_thw'].tolist();visual=int((inputs['input_ids']==processor.model_config.image_token_id).sum().item());patch=processor.processor.image_processor.patch_size
 assert visual==sum(t*h*w//processor.processor.image_processor.merge_size**2 for t,h,w in grid) and len(grid)==2
 budget=check_unit_budget(limits,image_sizes_hw=[[i.height,i.width] for i in images],processed_image_sizes_hw=[[h*patch,w*patch] for _,h,w in grid],
  input_tokens=int(inputs['input_ids'].shape[-1]),visual_tokens=visual,context_limit=processor.model_config.text_config.max_position_embeddings)
 return dict(status='cpu_input_verified_no_generation' if budget['status']=='ok' else 'budget_exceeded',model_calls=0,model_weights_loaded=False,budget=budget,image_grid_thw=grid,
  processor_tensor_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in inputs.items()})


def prepare(out,proposal):
 from PIL import Image
 from sandbox_vlm.data.video import FrameMapping
 from sandbox_vlm.runtime import VideoFrameReader
 from sandbox_vlm.qa.budget import QABudget
 from sandbox_vlm.qa.artifact import serial_messages,pixel_sha256
 from check_proposal_revision import ProcessorOnly
 started=time.perf_counter();cpu0=time.process_time()
 inp=read(proposal/'input_manifest.json');source=Path(inp['source']);m=read(source/'manifest.json')
 assert all(sha(p)==h for p,h in inp['source_sha256'].items())
 assert sha(proposal/'prepare_manifest.py')==inp['preparation_source_sha256']
 (out/'inputs').mkdir();files={}
 for p in sorted(proposal.iterdir()):
  if p.is_file() and p.suffix in ['.json','.txt','.py']:
   shutil.copy2(p,out/'inputs'/p.name);files[str(p)]=sha(p)
 tasks=make_tasks(out/'inputs');write(out/'tasks.json',tasks)
 frame_ids=read(proposal/'QA_frame_ids.json')['frame_ids']
 native_ids=sorted({t['region']['frame_id'] for t in tasks}|set(frame_ids))
 mapping=FrameMapping.load(source/'inference/frame_mapping.json');reader=VideoFrameReader(m['video_path'],mapping);reader.allowed_through=743
 (out/'native').mkdir();native_records=[]
 for fid in native_ids:
  cache=out.parent/'d095_identity_first_20261028_01/native'/f'{fid:06d}.jpg'
  if out.name.endswith(('_02','_03')) and cache.exists():
   content=cache.read_bytes();ref=mapping.by_id(fid)
   previous=read(cache.parent.parent/'native_records.json')
   assert hashlib.sha256(content).hexdigest()==next(r for r in previous if r['frame_id']==fid)['source_sha256']
   with Image.open(io.BytesIO(content)) as opened:size=[opened.height,opened.width]
  else:content,ref,size=reader.encoded_source(fid)
  (out/'native'/f'{fid:06d}.jpg').write_bytes(content)
  native_records.append(dict(frame_id=fid,source_index=ref.source_index,source_hw=size,source_sha256=hashlib.sha256(content).hexdigest()))
 write(out/'native_records.json',native_records)
 # Reproduce the exact exposed control image identity, without using its label in the input.
 old_controls=read(proposal/'known_unit_control_inputs.json')['endpoints']
 for row in old_controls:assert next(r for r in native_records if r['frame_id']==row['evidence']['frame_id'])['source_sha256']==row['record']['source_sha256']
 cfg=deepcopy(m['config']);cfg['qa']['budget'].update(max_images=32,max_image_side=768,max_output_tokens=1024)
 cfg['qa']['mllm']['max_new_tokens']=1024;cfg['qa']['mllm']['output_constraint']='json_unique_citations'
 processor=ProcessorOnly(cfg,True);unit_limits=QABudget(2,384,2048,1024,64)
 prompt_template=(proposal/'unit_prompt_template.txt').read_text();unit_preflight={}
 for task in tasks:
  folder=out/'units'/task['task_id'];folder.mkdir(parents=True)
  native=Image.open(out/'native'/f"{task['region']['frame_id']:06d}.jpg").convert('RGB')
  images=unit_images(native,task['region']);prompt=prompt_template.replace('{name_json}',json.dumps(task['name'],ensure_ascii=False))
  messages=unit_messages(images,prompt);admission=measure_unit(processor,messages,images,unit_limits)
  write(folder/'preflight.json',admission);write(folder/'messages.json',serial_messages(messages,images));write(folder/'task.json',task)
  (folder/'prompt.txt').write_text(prompt)
  for image,name in zip(images,['marked.png','crop.png']):image.save(folder/name)
  if admission['status']!='cpu_input_verified_no_generation':raise ValueError(f'unit input budget exceeded: {task["task_id"]}')
  unit_preflight[task['task_id']]=admission
 base_path=ROOT/'work/bench/d094_relaxed_birth_20261027_01/B/state.json';base=read(base_path)
 assert len(base['registry']['records'])==1575
 shutil.copy2(base_path,out/'base_state.json')
 regions=basis_records(tasks);unit_status={t['region_key']:'multiple_objects' for t in tasks}
 payload=dict(base=base,actual_region_layer=regions,unit_status=unit_status,pair_results=[],components={})
 # Reserve 1MiB additional serialized overlay + CPU Python overhead + packed current overlap/mask context.
 structural=charge_cpu(payload,extra=[tasks,inp],reserve=6*1024**2)
 if structural['charged_bytes']>LIMIT:raise ValueError('complete preflight structure upper bound exceeds 64MiB')
 images=[];records=[]
 for fid in frame_ids:
  image=Image.open(out/'native'/f'{fid:06d}.jpg').convert('RGB');image.thumbnail((384,384),Image.Resampling.LANCZOS)
  rec=next(r for r in native_records if r['frame_id']==fid)
  images.append(image);records.append(dict(kind='source_frame',frame_id=fid,source_index=rec['source_index'],source_sha256=rec['source_sha256'],input_image_size_hw=[image.height,image.width]))
 qa_preflight={}
 for arm,groups in [('A',[]),('B',[[r['region_key'] for r in regions[i:i+2]] for i in range(0,len(regions),2)])]:
  joint,meta,messages=qa_input(base,regions,unit_status,groups,frame_ids,images,records,m['example']['model_question'],arm)
  measured=processor.answer(messages,joint,QABudget(**cfg['qa']['budget']));qa_preflight[arm]=measured
  if measured['status']!='cpu_input_verified_no_generation':raise ValueError(f'full 32-region QA preflight exceeds fixed budget: {arm}')
 write(out/'cpu_preflight.json',dict(unit_inputs=unit_preflight,QA_shape_preflight=qa_preflight,complete_structure_upper_bound=structural,
  qualification='QA semantics/groups unknown; each actual generated QA input is checked again before generation.'))
 code=[*sorted((ROOT/'sandbox_vlm').rglob('*.py')),Path(__file__),ROOT/'scripts/check_proposal_revision.py',ROOT/'scripts/compare_observation_instances.py']
 hashes={}
 with zipfile.ZipFile(out/'code_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
  for path in code:
   rel=path.relative_to(ROOT);dest=out/'runtime'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);hashes[str(rel)]=sha(dest);z.write(dest,str(rel))
 (out/'runtime/venv').symlink_to(ROOT/'venv',target_is_directory=True)
 # All current inference inputs are frozen; future native source reads are tied to the immutable original video.
 frozen=[p for d in ['inputs','units','native'] for p in sorted((out/d).rglob('*')) if p.is_file()]
 frozen.extend([out/'base_state.json',out/'tasks.json',out/'native_records.json',out/'cpu_preflight.json',source/'manifest.json',source/'inference/frame_mapping.json',Path(m['video_path'])])
 manifest=dict(purpose='D095_identity_first_whole_fixed_candidate',proposal_root=str(proposal),source_root=str(source),source_project=str(ROOT),
  config=cfg,example=m['example'],video_path=m['video_path'],parquet_path=m['parquet_path'],frame_ids=frame_ids,
  source_sha256=hashes,input_sha256={str(p.resolve()):sha(p) for p in frozen},proposal_source_sha256=files,
  maximum_unit_calls=42,maximum_QA=2,maximum_SAM_prompts=512,maximum_SAM_tracks=32,maximum_SAM_encoder=1024,
  maximum_SAM_decoder=1024,maximum_propagation_steps=544,backend_max_bytes=LIMIT,
  maximum_source_frame_decodes=512,maximum_CPU_seconds=1800,mechanism_seconds=2580,mechanism_job_seconds=2700,
  maximum_GPU_allocated_bytes=32*1024**3,maximum_CPU_RSS_bytes=24*1024**3,maximum_archive_bytes=2*1024**3,
  prepared_native_frame_decodes=0 if out.name.endswith(('_02','_03')) else len(native_ids),inherited_initial_preparation_native_decodes=len(native_ids) if out.name.endswith(('_02','_03')) else 0,cpu_prepare_seconds=time.perf_counter()-started,cpu_prepare_process_seconds=time.process_time()-cpu0,
  inherited_perception_seconds=5870.569374939892,result_scope='one_exposed_full_video_fixed_16_short_overlap_relations',
  no_new_discovery=True,no_new_VGGT=True,QA_requires_machine_and_offline_gate=True,source_revision=read(source/'manifest.json')['config'])
 write(out/'manifest.json',manifest)
 write(out/'user_authorization.json',dict(user_instruction='执行（D095整体协议之后）',manifest_sha256=sha(out/'manifest.json'),
   scope='fixed16+4+2; identity first; gate failure closes candidate with zero QA; at most two A100 jobs and <=2 QA if passed'))
 print(json.dumps(dict(status='prepared_no_models',native_source_decodes=manifest['prepared_native_frame_decodes'],reused_native_frames=manifest.get('inherited_initial_preparation_native_decodes',0),unit_inputs=len(tasks),QA_preflight={a:r['budget'] for a,r in qa_preflight.items()},structure=structural),ensure_ascii=False))


def verify(out):
 m=read(out/'manifest.json');assert read(out/'user_authorization.json')['manifest_sha256']==sha(out/'manifest.json')
 assert all(sha(p)==h for p,h in m['input_sha256'].items())
 assert all(sha(out/'runtime'/p)==h for p,h in m['source_sha256'].items())
 return m


def unit_answer(model,messages,images,limits,report):
 import torch
 from lmformatenforcer import RegexParser
 from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
 from sandbox_vlm.qa.unit_budget import check_unit_budget
 started=time.perf_counter();inputs=model.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 nt=int(inputs['input_ids'].shape[-1]);visual=int((inputs['input_ids']==model.model.config.image_token_id).sum().item());grid=inputs['image_grid_thw'].tolist()
 assert visual==sum(t*h*w//model.processor.image_processor.merge_size**2 for t,h,w in grid) and len(grid)==2
 patch=model.processor.image_processor.patch_size
 budget=check_unit_budget(limits,image_sizes_hw=[[i.height,i.width] for i in images],processed_image_sizes_hw=[[h*patch,w*patch] for _,h,w in grid],input_tokens=nt,visual_tokens=visual,context_limit=model.model.config.text_config.max_position_embeddings)
 result=dict(status=budget['status'],budget=budget,image_grid_thw=grid,model_calls=0,processor_tensor_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in inputs.items()},generation_kwargs=dict(do_sample=False,max_new_tokens=64),output_pattern=UNIT_PATTERN)
 if budget['status']!='ok':return result
 if not hasattr(model,'_unit_tokens'):model._unit_tokens=build_token_enforcer_tokenizer_data(model.processor.tokenizer)
 prefix=build_transformers_prefix_allowed_tokens_fn(model._unit_tokens,RegexParser(UNIT_PATTERN))
 inputs=inputs.to(model.device);torch.cuda.synchronize();stamp=time.perf_counter();report['unit_attempts']+=1
 with torch.inference_mode():generated=model.model.generate(**inputs,do_sample=False,max_new_tokens=64,prefix_allowed_tokens_fn=prefix)
 torch.cuda.synchronize();ids=generated[0,nt:];eos=model.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
 capped=len(ids)>=64 and int(ids[-1]) not in eos
 result.update(raw_output=model.processor.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),model_calls=1,output_tokens=len(ids),generation_seconds=time.perf_counter()-stamp,wall_seconds=time.perf_counter()-started,status='output_cap_failure' if capped else 'generated')
 return result


class Mechanism:
 def __init__(self,out,m):
  import torch
  self.out=out;self.m=m;self.started=time.perf_counter();self.cpu0=time.process_time();self.buffer=None
  self.tasks=read(out/'tasks.json');self.pairs=read(out/'inputs/selected_pairs.json');self.refs=read(out/'inputs/same_frame_reference_prompts.json')
  self.base=read(out/'base_state.json');self.regions=basis_records(self.tasks);self.units={};self.results=[];self.groups=dict(accepted=[],rejected=[]);self.cannot=[]
  self.report=dict(status='running',unit_attempts=0,SAM_prompt_calls=0,SAM_tracks=0,SAM_encoder_calls=0,SAM_decoder_calls=0,propagation_steps=0,new_QA=0,new_discovery=0,new_VGGT=0,
   source_frame_decodes=m['prepared_native_frame_decodes']+m.get('inherited_initial_preparation_native_decodes',0),slurm_job_id=os.environ.get('SLURM_JOB_ID'),unit_slots={t['task_id']:dict(status='not_executed') for t in self.tasks},
   pair_slots=[dict(pair_index=i,status='not_executed') for i in range(16)],co_visible_controls=[dict(control_index=i,status='not_executed') for i in range(4)],structure_peak_bytes=0)
  torch.cuda.reset_peak_memory_stats();self.touch()
 def payload(self):return dict(base=self.base,actual_region_layer=self.regions,unit_status=self.units,pair_results=self.results,components=self.groups,cannot_links=self.cannot)
 def touch(self,extra=()):
  import torch,numpy as np
  self.buffer=None;gc.collect();torch.cuda.empty_cache()
  payload=self.payload();charge=charge_cpu(payload,[extra,self.tasks,self.pairs,self.refs,self.report,self.m],reserve=2*1024**2)
  if charge['charged_bytes']>LIMIT:raise ValueError('complete live structure exceeded 64MiB')
  raw=encoded(payload);self.buffer=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()).cuda()
  self.report['structure_peak_bytes']=max(self.report['structure_peak_bytes'],charge['charged_bytes']);self.report['GPU_complete_structure_bytes']=self.buffer.untyped_storage().nbytes()
  self.guard();write(self.out/'mechanism_report.json',self.report)
 def guard(self):
  import torch
  r=self.report;m=self.m
  for actual,limit in [('unit_attempts','maximum_unit_calls'),('SAM_prompt_calls','maximum_SAM_prompts'),('SAM_tracks','maximum_SAM_tracks'),('SAM_encoder_calls','maximum_SAM_encoder'),('SAM_decoder_calls','maximum_SAM_decoder'),('propagation_steps','maximum_propagation_steps'),('source_frame_decodes','maximum_source_frame_decodes')]:
   if r[actual]>m[limit]:raise ValueError('budget exceeded: '+actual)
  if time.perf_counter()-self.started>m['mechanism_seconds']:raise TimeoutError('mechanism wall budget')
  if time.process_time()-self.cpu0+m['cpu_prepare_process_seconds']>m['maximum_CPU_seconds']:raise TimeoutError('CPU budget')
  if torch.cuda.max_memory_allocated()>m['maximum_GPU_allocated_bytes']:raise MemoryError('GPU allocated budget')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>m['maximum_CPU_RSS_bytes']:raise MemoryError('CPU RSS budget')
  if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file() and 'runtime' not in p.parts)>m['maximum_archive_bytes']:raise ValueError('archive budget')
 def native(self,fid):
  p=self.out/'native'/f'{fid:06d}.jpg'
  if not p.exists():
   from sandbox_vlm.data.video import FrameMapping
   from sandbox_vlm.runtime import VideoFrameReader
   reader=VideoFrameReader(self.m['video_path'],FrameMapping.load(Path(self.m['source_root'])/'inference/frame_mapping.json'));reader.allowed_through=743
   content,ref,size=reader.encoded_source(fid);p.write_bytes(content);self.report['source_frame_decodes']+=1
   with (self.out/'additional_native_records.jsonl').open('a') as f:f.write(json.dumps(dict(frame_id=fid,source_index=ref.source_index,source_hw=size,source_sha256=hashlib.sha256(content).hexdigest()))+'\n')
  return p
 def image_dir(self,frame_ids,label):
  p=self.out/'tmp'/label;p.mkdir(parents=True,exist_ok=False)
  for fid in frame_ids:os.link(self.native(fid),p/f'{fid:06d}.jpg')
  return p
 def unit(self,model,task):
  from sandbox_vlm.qa.budget import QABudget
  from sandbox_vlm.qa.artifact import serial_messages,pixel_sha256
  folder=self.out/'units'/task['task_id'];images=load_unit_images(folder);messages=unit_messages(images,(folder/'prompt.txt').read_text())
  assert serial_messages(messages,images)==read(folder/'messages.json')
  generation=unit_answer(model,messages,images,QABudget(2,384,2048,1024,64),self.report)
  pre=read(folder/'preflight.json')
  assert generation.get('processor_tensor_sha256')==pre['processor_tensor_sha256']
  parsed=None
  if generation['status']=='generated':
   try:parsed=parse_unit(generation['raw_output'])
   except (ValueError,TypeError):generation['status']='unit_schema_failed'
  result=dict(task=task,generation=generation,unit_status=parsed,status='ok' if parsed else 'failed')
  if (folder/'result.json').exists():raise FileExistsError('unit rerun forbidden')
  write(folder/'result.json',result);self.report['unit_slots'][task['task_id']]=dict(status=result['status'],unit_status=parsed)
  # A conflicting unit interpretation for one actual region cannot become supported.
  key=task['region_key'];old=self.units.get(key)
  self.units[key]=parsed or 'unclear' if old is None or old==parsed else 'unclear'
  for region in self.regions:
   if region['region_key']==key:region['unit_status']=self.units[key]
  self.touch();return parsed
 def run_units(self):
  import torch
  from sandbox_vlm.qa.model import QwenAnswerer
  model=QwenAnswerer(self.m['config']['qa']['mllm'])
  for task in self.tasks[:2]:self.unit(model,task)
  controls=[self.report['unit_slots'][t['task_id']].get('unit_status') for t in self.tasks[:2]]
  self.report['unit_controls']=dict(observed=controls,expected=['one_object','multiple_objects'],passed=controls==['one_object','multiple_objects'])
  if not self.report['unit_controls']['passed']:
   self.report.update(status='stopped',stop_reason='fixed_unit_controls_failed',machine_gate_passed=False,QA_permitted=False)
   return False
  for task in self.tasks[2:]:self.unit(model,task)
  del model;gc.collect();torch.cuda.empty_cache()
  if any(self.report['unit_slots'][f'different_{i}_{side}']['unit_status']!='one_object' for i in range(4) for side in ['a','b']):
   self.report.update(status='stopped',stop_reason='co_visible_unit_controls_not_single_objects',machine_gate_passed=False,QA_permitted=False);return False
  return True
 def sam_hook(self,kind):
  def hook(module,args):
   self.report[kind]+=1;self.guard()
  return hook
 def track(self,tracker,task,pair,side):
  import numpy as np
  frames=pair[side+'_context_frame_ids'];directory=self.image_dir(frames,task['task_id'])
  self.report['SAM_prompt_calls']+=1;self.report['SAM_tracks']+=1
  self.guard()
  try:
   tracks,summary=tracker.track_clip(directory,frames,[dict(frame_id=task['region']['frame_id'],name=task['name'],short_description='',bbox_xyxy=task['region']['bbox_xyxy'])],clip_id=pair[side+'_clip'],iou_threshold=.70)
   if len(tracks)!=1:return None,None,summary
   track=tracks[0];self.report['propagation_steps']+=len(frames)+1;self.guard()
   packed={f:pack_mask(track.masks[f]) for f in pair['overlap_frame_ids']}
   feature=track.representative_feature.to_dict()
   for region in self.regions:
    if region['region_key']==task['region_key']:region['appearance']=feature
   return packed,feature,summary
  finally:shutil.rmtree(directory)
 def reference_scan(self,tracker,frame_id,consumer,label,only=None):
  import numpy as np,torch
  directory=self.image_dir([frame_id],label)
  try:
   predictor=tracker.predictor
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=predictor.init_state(str(directory));references=[r for r in self.refs if r['frame_id']==frame_id]
    # Original manifest deduplicates the exact frame/box key; do not filter by name.
    if only is not None:references=[r for r in references if region_key(r) in only]
    for region in references:
     self.report['SAM_prompt_calls']+=1;self.guard();predictor.reset_state(state)
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
      _,ids,logits=predictor.add_new_points_or_box(state,frame_idx=0,obj_id=0,box=np.asarray(region['bbox_xyxy'],dtype=np.float32))
     assert ids==[0];mask=(logits[0,0]>0).cpu().numpy();consumer(region,mask)
    predictor.reset_state(state)
  finally:shutil.rmtree(directory)
 def run_sam(self):
  import numpy as np
  from sandbox_vlm.perception.sam_tracks import SamCandidateTracker
  tracker=SamCandidateTracker(self.m['config']['perception']['sam'])
  h1=tracker.predictor.image_encoder.register_forward_pre_hook(self.sam_hook('SAM_encoder_calls'))
  h2=tracker.predictor.sam_mask_decoder.register_forward_pre_hook(self.sam_hook('SAM_decoder_calls'))
  try:
   negs=read(self.out/'inputs/co_visible_controls.json')
   for i,control in enumerate(negs):
    keys=[region_key(r) for r in control['regions']];masks={}
    self.reference_scan(tracker,control['regions'][0]['frame_id'],lambda r,mask:masks.update({region_key(r):pack_mask(mask)}),f'negative_{i}',set(keys))
    a,b=[unpack_mask(masks[k]) for k in keys];passed=bool(a.any() and b.any() and not (a&b).any())
    self.report['co_visible_controls'][i]=dict(control_index=i,status='passed' if passed else 'failed',region_keys=keys,mask_iou=mask_iou(a,b))
    np.savez_compressed(self.out/f'negative_{i}_masks.npz',**{f'mask_{j}':masks[k]['bits'] for j,k in enumerate(keys)})
    if not passed:self.report.update(status='stopped',stop_reason='co_visible_actual_masks_not_separate',machine_gate_passed=False,QA_permitted=False);return
    self.cannot.append(keys);self.touch()
   edges=[]
   for i,pair in enumerate(self.pairs):
    ta=next(t for t in self.tasks if t['task_id']==f'pair_{i:02d}_a');tb=next(t for t in self.tasks if t['task_id']==f'pair_{i:02d}_b')
    if self.units.get(ta['region_key'])!='one_object' or self.units.get(tb['region_key'])!='one_object':
     result=dict(pair_index=i,status='identity_unresolved',reason='endpoint_unit_not_one_object')
    else:
     a,fa,sa=self.track(tracker,ta,pair,'source');b,fb,sb=self.track(tracker,tb,pair,'target')
     if a is None or b is None:result=dict(pair_index=i,status='identity_unresolved',reason='independent_prompt_empty_mask',sam=[sa,sb])
     else:
      aa={f:unpack_mask(p) for f,p in a.items()};bb={f:unpack_mask(p) for f,p in b.items()}
      forward=[];backward=[]
      def scan_for(propagated,collector):
       def consume(region,mask):
        if mask_iou(propagated,mask)>=.70:collector.append(region_key(region))
       return consume
      self.reference_scan(tracker,tb['region']['frame_id'],scan_for(aa[tb['region']['frame_id']],forward),f'pair_{i:02d}_forward')
      self.reference_scan(tracker,ta['region']['frame_id'],scan_for(bb[ta['region']['frame_id']],backward),f'pair_{i:02d}_backward')
      result=dict(pair_index=i,**verify_pair(aa,bb,pair['overlap_frame_ids'],sorted(set(forward)),sorted(set(backward)),ta['region_key'],tb['region_key']),sam=[sa,sb])
      np.savez_compressed(self.out/f'pair_{i:02d}_overlap_masks.npz',**{f'{side}_{f}':p['bits'] for side,masks in [('a',a),('b',b)] for f,p in masks.items()})
      write(self.out/f'pair_{i:02d}_mask_shapes.json',dict(source={f:p['shape'] for f,p in a.items()},target={f:p['shape'] for f,p in b.items()}))
      if result['status']=='supported_local_relation':edges.append([ta['region_key'],tb['region_key']])
      del a,b,aa,bb
    self.results.append(result);self.report['pair_slots'][i]=result;self.touch()
   self.groups=supported_components(edges,self.cannot)
   supported={tuple(sorted((a,b))) for group in self.groups['accepted'] for a,b in itertools.combinations(group,2)}
   accepted=[i for i,row in enumerate(self.results) if row['status']=='supported_local_relation' and tuple(sorted([self.tasks[2+2*i]['region_key'],self.tasks[3+2*i]['region_key']])) in supported]
   quarters={self.pairs[i]['quarter'] for i in accepted};passed=len(accepted)>=4 and len(quarters)>=2
   self.report.update(status='awaiting_offline_review' if passed else 'stopped',machine_gate_passed=passed,
    accepted_pair_indices=accepted,accepted_quarters=sorted(quarters),stop_reason=None if passed else 'insufficient_supported_local_identity_relations',QA_permitted=False)
   self.touch();write(self.out/'actual_region_state.json',self.payload())
  finally:h1.remove();h2.remove()
 def finish(self):
  import torch
  self.report.update(program_seconds=time.perf_counter()-self.started,CPU_process_seconds=time.process_time()-self.cpu0,
   GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  write(self.out/'mechanism_report.json',self.report)
  write(self.out/'stage_slots.json',dict(unit_checks=self.report['unit_slots'],pairs=self.report['pair_slots'],co_visible_controls=self.report['co_visible_controls'],QA=[dict(arm=a,status='not_executed_identity_gate') for a in ['A','B']]))


def run_mechanism(out,m):
 import torch
 if (out/'mechanism_report.json').exists():raise FileExistsError('no automatic mechanism or model rerun')
 assert os.environ.get('SLURM_JOB_ID') and 'A100' in torch.cuda.get_device_name()
 torch.manual_seed(0);work=None
 try:
  work=Mechanism(out,m)
  if work.run_units():work.run_sam()
 except Exception as exc:
  if work is None:write(out/'mechanism_start_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),QA_permitted=False))
  else:work.report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),machine_gate_passed=False,QA_permitted=False)
 finally:
  if work:work.finish()
 return 1 if work is None or work.report['status']=='failed' else 0


def prepare_qa(out,m):
 from PIL import Image
 from sandbox_vlm.qa.budget import QABudget
 from sandbox_vlm.qa.artifact import save_joint_input,pixel_sha256
 from check_proposal_revision import ProcessorOnly
 gate=read(out/'offline_review.json');report=read(out/'mechanism_report.json')
 assert gate['passed'] and report['machine_gate_passed'] and gate['mechanism_report_sha256']==sha(out/'mechanism_report.json')
 folder=out/'QA';folder.mkdir(exist_ok=False);payload=read(out/'actual_region_state.json')
 frames=m['frame_ids'];images=[];records=[]
 for fid in frames:
  image=Image.open(out/'native'/f'{fid:06d}.jpg').convert('RGB');image.thumbnail((384,384),Image.Resampling.LANCZOS);images.append(image)
  records.append(dict(kind='source_frame',frame_id=fid,input_image_size_hw=[image.height,image.width]))
 processor=ProcessorOnly(m['config'],True);summary={};inputs=[]
 for arm in ['A','B']:
  dest=folder/arm;dest.mkdir();state=deepcopy(payload)
  if arm=='A':state['components']=dict(accepted=[],rejected=[])
  joint,meta,messages=qa_input(state['base'],state['actual_region_layer'],state['unit_status'],state['components']['accepted'],frames,images,records,m['example']['model_question'],arm)
  admission=processor.answer(messages,joint,QABudget(**m['config']['qa']['budget']))
  assert admission['status']=='cpu_input_verified_no_generation'
  charge=charge_cpu(state,extra=meta,reserve=0);assert charge['charged_bytes']<=LIMIT
  save_joint_input(dest/'prepared_input',joint,meta,messages);write(dest/'state.json',state)
  write(dest/'preflight.json',dict(processor=admission,frames_pixel_sha256=[pixel_sha256(x) for x in joint[1:]],structure=charge))
  summary[arm]=dict(processor=admission);inputs.extend([dest/'state.json',dest/'preflight.json',*sorted((dest/'prepared_input').glob('*'))])
 qm=dict(config=m['config'],example=m['example'],parquet_path=m['parquet_path'],source_project=m['source_project'],frame_ids=frames,program_limit_seconds=780,backend_max_bytes=LIMIT,
  source_sha256=m['source_sha256'],input_sha256={str(p.resolve()):sha(p) for p in inputs},result_scope='D095_identity_passed_one_exposed_full_video_QA_pair',arms=summary)
 # compare runner verifies source paths relative to this symlink.
 (folder/'runtime').symlink_to(out/'runtime',target_is_directory=True)
 write(folder/'manifest.json',qm);write(folder/'user_authorization.json',dict(manifest_sha256=sha(folder/'manifest.json'),scope='D095 pre-approved QA only after both identity gates'))
 print('QA_PREPARED_AFTER_BOTH_IDENTITY_GATES')


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--proposal',type=Path);p.add_argument('--action',choices=['prepare','verify','mechanism','prepare_qa','qa','score'],required=True);a=p.parse_args();out=a.output.resolve()
 if a.action=='prepare':prepare(out,a.proposal.resolve())
 else:
  m=verify(out)
  if a.action=='mechanism':raise SystemExit(run_mechanism(out,m))
  elif a.action=='prepare_qa':prepare_qa(out,m)
  elif a.action in ['qa','score']:
   from compare_observation_instances import verify as verify_qa,run as run_qa,score
   qm=verify_qa(out/'QA')
   if a.action=='qa':raise SystemExit(run_qa(out/'QA',qm))
   else:score(out/'QA',qm)
  else:print('D095_FROZEN_INPUTS_CODE_VERIFIED_NO_MODEL')
