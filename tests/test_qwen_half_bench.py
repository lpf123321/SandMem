"""Sampling, frozen-method derivation and native-Qwen transport contracts."""
import ast,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_qwen_half_bench import stratify,mind_inputs,expected_failure


def test_sampling_ignores_answers_and_takes_half_each_type():
    rows=[dict(id=i,question_type='a' if i<5 else 'b',ground_truth=str(i)) for i in range(9)]
    inputs=[dict(id=i,question='q'+str(i)) for i in range(9)]
    before,counts=stratify(rows,inputs,lambda r:r['question_type'])
    for r in rows:r['ground_truth']='CHANGED'
    after,_=stratify(rows,list(reversed(inputs)),lambda r:r['question_type'])
    assert before==after and len(before)==5
    assert counts=={'a':{'full':5,'sample':3},'b':{'full':4,'sample':2}}


def test_mindcube_inference_allowlist_preserves_original_views_question():
    row=dict(id='among_group1',question='question A. one B. two',images=['one.jpg','two.jpg'],gt_answer='A',meta_info=['SECRET'],category=['SECRET'],type='0_frame')
    result=mind_inputs([row])[0]
    assert result['images']==row['images'] and result['question']==row['question']
    assert not {'gt_answer','meta_info','category','type'}&result.keys()
    assert 'SECRET' not in json.dumps(result)


def test_generated_cold_runner_only_translates_declared_model_checks():
    source=(ROOT/'scripts/run_spatial_vsibench.py').read_text()
    target=(ROOT/'scripts/qwen_d116_cold.py').read_text()
    names={'d116_identity_and_correspondences','discovery_with_failure_receipt','InputArchive','verification_with_original_images','fresh_question'}
    original={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    adapted={n.name:ast.get_source_segment(target,n) for n in ast.parse(target).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    for name,text in original.items():
        text=text.replace('from sandbox_vlm.runtime import VideoFrameReader','from sandbox_vlm.data.view_sequence import VideoFrameReader').replace('from sandbox_vlm.perception.parallel_decode import CoverageReader','from sandbox_vlm.data.view_sequence import CoverageReader')
        text=text.replace("                    assert any(t['component']=='actual_spatial_encoder' for t in trace)","                    # Native Qwen coverage.reuse validates exact cached pixels/grid.")
        text=text.replace("        assert any(t['component']=='actual_spatial_encoder' and len(t['shapes'])==len(imgs)\n            for t in result['actual_forward_trace']),'not all supplied images reached the spatial encoder'", "        assert len(result['image_grid_thw'])==len(imgs), 'Qwen image count mismatch'\n        assert result['budget']['status']=='ok', 'Qwen joint input budget failed'")
        text=text.replace('full_spatial_perception=True','full_qwen_perception=True')
        text=text.replace('    prior=(Path(resume_from)/f\'question_{item["id"]}\') if resume_from else None',"    if resume_from is not None: raise ValueError('fresh Qwen evaluation forbids semantic cache reuse')\n    prior=None")
        assert ast.dump(ast.parse(text))==ast.dump(ast.parse(adapted[name])),name


def test_unknown_engineering_errors_stop_instead_of_scoring():
    assert expected_failure(RuntimeError('QA contract failure: answer_failed'))
    assert not expected_failure(RuntimeError('CUDA out of memory'))
    assert not expected_failure(ValueError('source video changed'))


def test_native_view_reader_source_pixels_mapping_and_causality(tmp_path):
    import numpy as np
    import pytest
    from PIL import Image
    from sandbox_vlm.data.view_sequence import VideoFrameReader,CoverageReader,digest
    from sandbox_vlm.data.video import FrameMapping,FrameRef
    source=tmp_path/'odd.png'
    array=np.arange(17*19*3,dtype=np.uint8).reshape(17,19,3)
    Image.fromarray(array).save(source)
    index=tmp_path/'native.views.json';index.write_text(json.dumps(dict(source_images=[dict(path=str(source),sha256=digest(source))])))
    mapping=FrameMapping(video_id='views',fps=1.,sample_fps=1.,clip_len=16,overlap=4,frames=[FrameRef(frame_id=0,source_index=0,t_sec=0.)])
    reader=VideoFrameReader(index,mapping)
    with pytest.raises(ValueError):reader.encoded_source(0)
    reader.allowed_through=0
    assert np.array_equal(np.asarray(reader.original(0)),array)
    coverage=CoverageReader(index,mapping)
    assert np.array_equal(np.asarray(coverage.read_batch([0],allowed_through=0,max_side=64)[0]),array)
    with pytest.raises(ValueError):reader.encoded_source(1)
    Image.fromarray(np.zeros_like(array)).save(source)
    with pytest.raises(ValueError,match='source image changed'):reader.encoded_source(0)
