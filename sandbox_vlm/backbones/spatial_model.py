"""Independent frozen Spatial-MLLM-135K runtime for auditable wiring checks.

Historical Qwen transports are untouched. This adapter does not run the D116
method or inherit a memory cache by itself; callers must label their inputs.
"""
from contextlib import contextmanager
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

MODEL_ID = 'Diankun/Spatial-MLLM-v1.1-Instruct-135K'
REVISION = '635b63733ed24f76b2f1a533dd5bc32836ea6c53'
OFFICIAL_COMMIT = '9fc47382c7bc5ab52951e6e2e64db08fca0948ee'


class Spatial135K:
    def __init__(self, official_root, *, device='cuda:0'):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Qwen2_5_VLProcessor
        self.official_root=Path(official_root).resolve()
        commit=subprocess.check_output(['git','-C',str(self.official_root),'rev-parse','HEAD'],text=True).strip()
        if commit != OFFICIAL_COMMIT:
            raise ValueError('official source revision changed')
        if subprocess.check_output(['git','-C',str(self.official_root),'diff','--name-only','HEAD'],text=True).strip():
            raise ValueError('official source has unsealed modifications')
        for package,required in [('torch','2.6.0'),('transformers','4.51.3'),('flash-attn','2.7.4.post1')]:
            if importlib.metadata.version(package).split('+')[0] != required:
                raise ValueError('independent official environment required: '+package)
        external=self.official_root/'src/qwenvl/external'
        existing=sys.modules.get('vggt')
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(external):
            raise RuntimeError('external VGGT namespace collision; use an independent worker')
        sys.path[:0]=[str(self.official_root),str(external)]
        from src.qwenvl.model.spatial_mllm import SpatialMLLMConfig,SpatialMLLMForConditionalGeneration
        checkpoint=snapshot_download(MODEL_ID,revision=REVISION,local_files_only=True)
        self.processor=Qwen2_5_VLProcessor.from_pretrained(checkpoint,local_files_only=True,use_fast=False)
        config=SpatialMLLMConfig.from_pretrained(checkpoint,local_files_only=True)
        self.device=device
        self.model,self.loading_info=SpatialMLLMForConditionalGeneration.from_pretrained(checkpoint,
            config=config,torch_dtype=torch.bfloat16,device_map={'':device},
            attn_implementation='flash_attention_2',local_files_only=True,output_loading_info=True)
        if any(self.loading_info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('checkpoint loading was not exact: '+str(self.loading_info))
        self.model.eval().requires_grad_(False)
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('backbone is not frozen')
        self.generation_attempts=0

    def prepare(self,messages,media,*,watermark,require_sandbox=False):
        from sandbox_vlm.backbones.spatial_inputs import prepare_inputs
        return prepare_inputs(self.processor,self.model.config,messages,media,
            watermark=watermark,require_sandbox=require_sandbox)

    def on_device(self,batch):
        # BatchFeature.to does not reliably move tensors nested in lists in
        # the pinned transformers release. Keep timing scalars unchanged.
        import torch
        def move(value):
            if isinstance(value,torch.Tensor):return value.to(self.device)
            if isinstance(value,list):return [move(v)for v in value]
            return value
        return {k:move(v)for k,v in batch.items()}

    @contextmanager
    def trace_media(self):
        """Observe actual model/encoder invocations, without replacing their outputs."""
        trace=[]
        def before_model(module,args,kwargs):
            trace.append(dict(component='MLLM_forward',
                image_pixel_values_present=kwargs.get('pixel_values')is not None,
                video_pixel_values_present=kwargs.get('pixel_values_videos')is not None,
                image_spatial_shapes=[list(t.shape)for t in (kwargs.get('image_tchw')or[])],
                video_spatial_shapes=[list(t.shape)for t in (kwargs.get('video_tchw')or[])]))
        def before_spatial(module,args,kwargs):
            values=args[0]if args else kwargs['video_tensor']
            trace.append(dict(component='actual_spatial_encoder',shapes=[list(v.shape)for v in values]))
        handles=[self.model.register_forward_pre_hook(before_model,with_kwargs=True),
                 self.model.spatial_encoder.register_forward_pre_hook(before_spatial,with_kwargs=True)]
        try:yield trace
        finally:
            for handle in handles:handle.remove()

    def generate(self,batch,*,generation_kwargs,note_boundary=None):
        import torch
        from transformers import LogitsProcessorList
        values=self.on_device(batch)
        width=int(values['input_ids'].shape[-1])
        kwargs=dict(generation_kwargs)
        boundary=None
        if note_boundary is not None:
            from sandbox_vlm.qa.note_boundary import NoteBoundary
            eos=self.model.generation_config.eos_token_id
            boundary=NoteBoundary(self.processor.tokenizer,width,kwargs['max_new_tokens'],
                eos if isinstance(eos,list)else[eos],**note_boundary)
            kwargs['logits_processor']=LogitsProcessorList([boundary])
        torch.cuda.synchronize();started=time.perf_counter();self.generation_attempts+=1
        with self.trace_media()as trace,torch.inference_mode():
            output=self.model.generate(**values,**kwargs)
        torch.cuda.synchronize()
        ids=output[0,width:];eos=self.model.generation_config.eos_token_id
        eos=eos if isinstance(eos,list)else[eos]
        capped=len(ids)>=generation_kwargs['max_new_tokens']and int(ids[-1])not in eos
        result=dict(status='answer_failed'if capped else'generated',
            raw_output=self.processor.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            output_token_ids=ids.tolist(),output_tokens=len(ids),reached_output_limit=capped,
            generation_seconds=time.perf_counter()-started,generation_kwargs=dict(generation_kwargs),
            actual_forward_trace=trace,model_calls=1,model_id=MODEL_ID,revision=REVISION)
        if boundary is not None:result['note_boundary']=boundary.report()
        return result

    def relevance(self,batch):
        """The original Yes/No next-token margin using this same frozen backbone."""
        import torch
        tokens=[self.processor.tokenizer.encode(t,add_special_tokens=False)for t in ('Yes','No')]
        if any(len(t)!=1 for t in tokens)or tokens[0]==tokens[1]:
            raise ValueError('Yes/No must be distinct single tokens')
        values=self.on_device(batch)
        if not bool(values['attention_mask'][:,-1].all()):raise ValueError('last token is padding')
        torch.cuda.synchronize();started=time.perf_counter()
        with self.trace_media()as trace,torch.inference_mode():
            # Official forward has no logits_to_keep argument; do not reuse
            # the Qwen3 scorer signature or silently substitute a new gate.
            output=self.model(**values,use_cache=False)
            pair=output.logits[:,-1,[tokens[0][0],tokens[1][0]]].float()
            margins=(pair[:,0]-pair[:,1]).cpu().tolist()
        torch.cuda.synchronize()
        return dict(margins=margins,yes_no_token_ids=[t[0]for t in tokens],
            model_seconds=time.perf_counter()-started,actual_forward_trace=trace,
            logical_requests=len(margins),forward_calls=1,generated_tokens=0,use_cache=False,
            model_id=MODEL_ID,revision=REVISION)

    def coverage(self,batch):
        """D043's same 4x4/64-bin descriptor, using Spatial's own 2D visual tower."""
        import torch
        import torch.nn.functional as F
        values=self.on_device(batch);grids=values['image_grid_thw'].tolist()
        sizes=[t*h*w//self.processor.image_processor.merge_size**2 for t,h,w in grids]
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            all_embeddings=self.model.visual(values['pixel_values'].to(self.model.visual.dtype),
                grid_thw=values['image_grid_thw'])
            embeddings=all_embeddings.split(sizes);descriptors=[]
            for embedding,(t,h,w)in zip(embeddings,grids):
                if t!=1 or embedding.shape[-1]%64:raise ValueError('independent coverage grid required')
                grid=embedding.float().reshape(h//2,w//2,-1).permute(2,0,1)
                pooled=F.adaptive_avg_pool2d(grid,(4,4)).permute(1,2,0).reshape(16,64,-1).mean(-1)
                descriptors.append(F.normalize(pooled,dim=-1).half())
            descriptors=torch.stack(descriptors).cpu()
        torch.cuda.synchronize()
        return descriptors,dict(vision_seconds=time.perf_counter()-started,vision_forward_calls=1,
            model_id=MODEL_ID,revision=REVISION,source='same_frozen_backbone_visual_tower',
            descriptor_shape=list(descriptors.shape),spatial_encoder_calls=0)
