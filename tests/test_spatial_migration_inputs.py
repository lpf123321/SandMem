"""CPU contracts for actual spatial streams and cache/causal boundaries."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from sandbox_vlm.backbones.cache_provenance import reuse_decision
from sandbox_vlm.backbones.spatial_inputs import input_record, spatial_tensors, validate_media


def test_images_are_independent_temporal_patches():
    a = Image.fromarray(np.full((56, 84, 3), 32, dtype=np.uint8))
    b = Image.fromarray(np.full((84, 56, 3), 192, dtype=np.uint8))
    images, videos = spatial_tensors([a,b], None, [[1,4,6],[1,6,4]], [],
                                    patch_size=14, temporal_patch_size=2)
    assert videos is None and [list(i.shape) for i in images] == [[2,3,56,84],[2,3,84,56]]
    assert all(torch.equal(i[0],i[1]) for i in images)
    assert images[0][0,0,0,0].item() == pytest.approx(32/255)
    assert images[1][0,0,0,0].item() == pytest.approx(192/255)


def test_video_spatial_pixels_match_exact_grid():
    video = torch.arange(4*3*56*84,dtype=torch.int64).reshape(4,3,56,84).remainder(256).byte()
    images, videos = spatial_tensors([], [video], [], [[2,4,6]], patch_size=14, temporal_patch_size=2)
    assert images is None and torch.equal(videos[0], video.float()/255)
    with pytest.raises(ValueError,match='padding'):
        spatial_tensors([], [video[:3]], [], [[2,4,6]], patch_size=14, temporal_patch_size=2)


def test_video_timing_metadata_is_kept_as_scalar_data():
    assert input_record([1.0,0.5])==[1.0,0.5]
    assert input_record([torch.zeros(2),0.5])[0]['shape']==[2]


def test_future_frames_and_full_movie_paths_fail_before_decode():
    message = [dict(content=[dict(type='video',video='whole.mp4')])]
    with pytest.raises(ValueError,match='exact explicit'):
        validate_media(message,[dict(kind='source_video',frame_ids=[0,1])],watermark=1)
    message[0]['content'][0]['video'] = ['0.png','2.png']
    with pytest.raises(ValueError,match='future'):
        validate_media(message,[dict(kind='source_video',frame_ids=[0,2])],watermark=1)


def test_joint_inputs_need_a_real_image_block_for_sandbox():
    message = [dict(content=[dict(type='image',image='actual.png')])]
    with pytest.raises(ValueError,match='actual sandbox'):
        validate_media(message,[dict(kind='source_frame',frame_ids=[0])],watermark=0,require_sandbox=True)
    assert validate_media(message,[dict(kind='sandbox',frame_ids=[0])],watermark=0,require_sandbox=True)


def test_cache_reuse_checks_prompts_and_backbone():
    old = dict(input_sha256='frame+box_A',model_revision='sam_rev',code_sha256='code',processor_sha256='proc')
    assert reuse_decision('sam_mask',old,old,mode='full_spatial')['reuse']
    assert not reuse_decision('sam_mask',old,dict(old,input_sha256='frame+box_B'),mode='full_spatial')['reuse']
    qwen = dict(old,mllm_backbone='Qwen/Qwen3-VL-2B-Instruct')
    assert not reuse_decision('instance_state',qwen,qwen,mode='full_spatial')['reuse']
    assert reuse_decision('instance_state',qwen,qwen,mode='fixed_qwen_memory_qa_swap')['reuse']
    assert not reuse_decision('vggt_geometry',{},old,mode='full_spatial')['reuse']


def test_official_static_image_helper_has_incompatible_rank():
    root = Path(__file__).resolve().parents[1]/'vendor/spatial-mllm-official'
    if not root.exists():
        pytest.skip('official source checkout unavailable')
    def load_function(path,name,namespace):
        tree=ast.parse(path.read_text())
        nodes = list(ast.walk(tree))
        node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        return namespace[name]
    helper=load_function(root/'src/inference.py','prepare_spatial_mllm_inputs',dict(torch=torch,np=np,Image=Image))
    values=helper({},None,[Image.new('RGB',(56,56))])
    assert values['image_tchw'][0].ndim==3
    from collections import defaultdict
    forward=load_function(root/'src/qwenvl/model/spatial_encoder.py','forward',dict(torch=torch,List=list,defaultdict=defaultdict))
    with pytest.raises(ValueError,match='unpack'):
        forward(SimpleNamespace(),values['image_tchw'])


def test_official_connector_accepts_independent_repeated_image_slots():
    path=Path(__file__).resolve().parents[1]/'vendor/spatial-mllm-official/src/qwenvl/model/connector/mlp_add_connector.py'
    if not path.exists():
        pytest.skip('official source checkout unavailable')
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='preprocess_spatial_embeds')
    namespace=dict(torch=torch,List=list)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    connector=SimpleNamespace(spatial_embeds_layer_idx=-1,visual_temporal_merge_size=2,visual_spatial_merge_size=2)
    spatial=[[torch.zeros(2,29,8)],[torch.ones(2,29,8)]]
    result=namespace['preprocess_spatial_embeds'](connector,spatial,[5,5],torch.tensor([[1,4,6],[1,6,4]]))
    assert list(result.shape)==[12,64]
    assert not result[:6].any() and result[6:].all()
