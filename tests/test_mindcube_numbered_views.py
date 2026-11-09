import ast,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sandbox_vlm.data.numbered_views import full_view_ids,bind_messages
from sandbox_vlm.qa.prompt import messages_for


@pytest.mark.parametrize('n',[2,3,4])
@pytest.mark.parametrize('sparse',[False,True])
def test_all_view_labels_question_image_identity_and_policy(n,sparse):
    images=[object() for _ in range(n)];diagram=object();question='In image 1, which object is left? A. one B. two'
    original=messages_for(question,images,list(range(n)),'brief_evidence' if sparse else 'question_last')
    original[0]['content'].insert(1,dict(type='image',image=diagram))
    oldtexts=[b.get('text') for b in original[0]['content']]
    adapted,contract=bind_messages(original,list(range(n)),has_sandbox=True)
    content=adapted[0]['content'];assert [b['image'] for b in content if b['type']=='image']==[diagram,*images]
    assert [b.get('text') for b in original[0]['content']]==oldtexts # no alias mutation
    assert any(b.get('text')=='Original question (verbatim):\n'+question for b in content)
    text='\n'.join(b['text'] for b in content if b['type']=='text')
    for i in range(n):assert f'Original benchmark image {i+1} (actual source frame ID {i}):' in text
    assert 'not occupy or shift any image number' in text and 'not video timestamps' in text
    if sparse:assert 'at most 100 words' in text and 'FINAL_ANSWER: <answer>' in text
    assert contract['original_view_media_indices']==list(range(1,n+1))


def test_missing_or_unordered_original_view_rejected():
    with pytest.raises(ValueError):bind_messages([], [0,2],has_sandbox=False)
    mapping=SimpleNamespace(frames=[SimpleNamespace(frame_id=0,source_index=0)])
    with pytest.raises(ValueError):full_view_ids(dict(dataset='mindcube',images=['a','b']),mapping)


def test_adapter_is_not_a_long_video_or_task_label_branch():
    mapping=SimpleNamespace(frames=[SimpleNamespace(frame_id=i,source_index=i) for i in range(3)])
    item=dict(dataset='mindcube',images=['a','b','c'],question='untouched')
    assert full_view_ids(item,mapping)==[0,1,2]
    with pytest.raises(ValueError):full_view_ids(dict(item,dataset='scannet'),mapping)


def test_exact_cold_runner_change_whitelist():
    source=(ROOT/'scripts/qwen_d116_cold.py').read_text();actual=(ROOT/'scripts/qwen_d116_mindcube_views.py').read_text()
    source=source.replace('from run_spatial_vsibench import read,write,sha','from run_spatial_vsibench import read,write,sha\nfrom sandbox_vlm.data.numbered_views import full_view_ids,bind_messages')
    source=source.replace('    chosen=select_base_frames(state,ledger,new_ids,sparse=sparse,config=cfg)','    chosen=full_view_ids(item,mapping) # Explicit original numbered views, all within original input budget.')
    source=source.replace('    def answer(stage,msg,imgs,lim,note):\n        archive.stage=stage',"    def answer(stage,msg,imgs,lim,note):\n        msg,contract=bind_messages(msg,chosen,has_sandbox=(len(imgs)==len(chosen)+1))\n        write(folder/(stage+'_view_contract.json'),contract)\n        archive.stage=stage")
    source=source.replace('full_qwen_perception=True,old_mllm_cache_used=False,',"full_qwen_perception=True,old_mllm_cache_used=False,dataset_adapter='explicit_numbered_views_v1',")
    assert ast.dump(ast.parse(source))==ast.dump(ast.parse(actual))
