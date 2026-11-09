"""Independent CPU replay of archived native Qwen inputs; zero model calls."""
import collections,gzip,hashlib,io,json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sandbox_vlm.qa.artifact import pixel_sha256


def read(p):return json.loads(Path(p).read_text())
def digest(item):return hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()


def run():
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    out=ROOT/'work/qwen_mindcube_viewfix_20261107_01';m=read(out/'manifest.json')
    config=m['config']['qa']['mllm'];processor=AutoProcessor.from_pretrained(config['model_id'],revision=config['revision'],local_files_only=True)
    pools=collections.defaultdict(list)
    for item in m['inputs']:
        p=out/('question_'+item['id']);r=read(p/'result.json')
        pools[(item['id'].split('_')[0],len(item['images']),r['sparse'])].append(item)
    selected=[min(v,key=digest) for _,v in sorted(pools.items())];receipts=[]
    base=ROOT/'work/mindcube_data_20261107/extracted/data'
    for item in selected:
        p=out/('question_'+item['id']);r=read(p/'result.json');side=m['config']['qa']['sparse_image_side'] if r['sparse'] else m['config']['qa']['image_side'];frames=[]
        for i,name in enumerate(item['images']):
            with Image.open(base/name) as im:
                im=im.convert('RGB');b=io.BytesIO();im.save(b,format='JPEG',quality=100)
            with Image.open(io.BytesIO(b.getvalue())) as im:
                image=im.convert('RGB');image.thumbnail((side,side),Image.Resampling.LANCZOS)
            with Image.open(p/f'qa_frame_{i:06d}.png') as old:assert image.size==old.size and image.tobytes()==old.convert('RGB').tobytes(),'source frame pixels differ'
            frames.append(image)
        with Image.open(p/'sandbox.png') as im:diagram=im.convert('RGB')
        stages=[]
        for saved in sorted(p.glob('*_messages.json')):
            stage=saved.name.removesuffix('_messages.json');messages=read(saved);images=frames if stage=='draft' else [diagram,*frames]
            for message in messages:
                for block in message['content']:
                    if block['type']=='image':
                        index=block.pop('evidence_image_index');assert block.pop('pixel_sha256')==pixel_sha256(images[index]);block['image']=images[index]
            native=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            paths=list((p/'actual_inputs').glob('*_'+stage+'/tensors.pt.gz'));assert len(paths)==1,(item['id'],stage)
            with gzip.open(paths[0],'rb') as f:actual=torch.load(f,map_location='cpu',weights_only=True)
            assert set(actual)==set(native),(item['id'],stage,'tensor keys differ')
            for k in actual:assert torch.equal(actual[k],native[k]),(item['id'],stage,k,'actual tensor differs')
            stages.append(dict(stage=stage,keys=list(actual),images=len(images),exact_tensor_replay=True))
        receipts.append(dict(id=item['id'],original_views=len(frames),sparse=r['sparse'],source_pixels_independently_verified=True,stages=stages))
        for im in [diagram,*frames]:im.close()
    target=ROOT/'work/audits/qwen_mindcube_viewfix_diagnosis_20261108_01/native_input_replay.json'
    target.write_text(json.dumps(dict(status='passed',selection='minimum input hash per setting/view-count/sparsity; no answer or score selection',questions=len(receipts),cases=receipts,model_calls=0),indent=2))
    print('native archived tensor/source-pixel replay passed',len(receipts),'questions',sum(len(r['stages']) for r in receipts),'model calls=0')

if __name__=='__main__':run()
