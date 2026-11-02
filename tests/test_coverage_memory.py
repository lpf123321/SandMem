import hashlib

import cv2
import numpy as np
import pytest

from sandbox_vlm.data.video import build_frame_mapping
from sandbox_vlm.memory.coverage import CoverageSandbox, ContinuitySettings, compact_memory, select_answer_frames
from sandbox_vlm.perception.coverage import NoveltyShortlist, pack_descriptor, unpack_descriptor
from sandbox_vlm.perception.coverage_budget import allocate_discovery
from sandbox_vlm.perception.parallel_decode import CoverageReader
from sandbox_vlm.qa.coverage_evidence import append_evidence,temporal_answer_frames


def descriptor(seed=0):
    a = np.random.default_rng(seed).normal(size=(16, 64)).astype(np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def actual(fid, *, name='tv', box=(1., 1., 20., 20.), context=0):
    return dict(video_id='video', name=name, description='visible object',
        evidence=dict(frame_id=fid, bbox_xyxy=list(box), source_hw=[100, 100], sam_mask_nonempty=True),
        appearance=dict(vector=[1., 0.], dtype='float32', dim=2),
        context_descriptor=descriptor(context), settings=ContinuitySettings(), allowed_through=fid)


def test_short_event_burst_survives_and_future_or_missing_frames_are_rejected():
    values = np.repeat(descriptor()[None], 16, axis=0)
    values[7] = descriptor(3)
    shortlist = NoveltyShortlist()
    selected, _ = shortlist.select(list(range(16)), values, query=15)
    assert 7 in selected and 0 in selected and len(selected) <= 7
    assert unpack_descriptor(pack_descriptor(values[0])).shape == (16,64)
    with pytest.raises(ValueError):
        shortlist.select([17], values[:1], query=17)
    with pytest.raises(ValueError):
        shortlist.select([16], values[:1], query=15)


def test_far_appearance_is_not_identity_and_local_actual_continuity_can_reuse():
    state = CoverageSandbox('video', 'original question')
    assert state.ingest(**actual(0))['status']=='created_visual_hypothesis'
    assert state.ingest(**actual(2))['status']=='matched_continuous_visual_hypothesis'
    assert state.ingest(**actual(100))['status']=='created_visual_hypothesis'
    assert len(state.instances)==2 and state.ambiguities[-1]['candidate_visual_instance_ids']==[0]
    assert state.ingest(**actual(101, context=9))['status']=='created_visual_hypothesis'
    assert CoverageSandbox.from_dict(state.to_dict()).to_dict()==state.to_dict()
    state.validate()


def test_covisible_objects_and_duplicate_actual_observations_remain_consistent():
    state = CoverageSandbox('video', 'question')
    state.ingest(**actual(0))
    second = state.ingest(**actual(0, box=(30.,30.,50.,50.)))
    assert second['visual_instance_id']==1
    assert not state.ambiguities
    assert state.ingest(**actual(0))['status']=='duplicate_actual_observation'
    assert len(state.viewpoints['0'])==2
    bad=actual(1);bad['allowed_through']=0
    with pytest.raises(ValueError):state.ingest(**bad)


def test_question_provenance_keeps_aliases_and_more_than_64_hypotheses():
    state = CoverageSandbox('video', 'How many tv(s)?')
    ledger={}
    for i in range(70):
        fid=i*20
        state.ingest(**actual(fid,name='tv' if i%2 else 'television'))
        ledger[fid]=dict(margin=1.,descriptor=pack_descriptor(descriptor(i)))
    selected=select_answer_frames(state,ledger,maximum=32)
    result=compact_memory(state,'How many tv(s)?',selected)
    assert len(selected)==32 and set(selected)<=set(ledger)
    assert sum(len(ids) for ids in result['name_to_visual_hypothesis_ids'].values())==70
    assert set(result['name_to_visual_hypothesis_ids'])=={'tv','television'}
    assert result['limitations']['omitted_hypotheses']==0
    with pytest.raises(ValueError):compact_memory(state,'A different question?',selected)


def test_lifetime_discovery_can_grow_without_expanding_answer_images():
    ledger={fid:dict(margin=1.,descriptor=pack_descriptor(descriptor(fid))) for fid in range(0,7200,20)}
    frames,stats=allocate_discovery(ledger,query=7199,total_frames=7200,selected=[],previous_query=-1)
    assert len(frames)==stats['lifetime_cap']==240 and len(frames)>32
    assert max(frames)<=7199 and len(frames)==len(set(frames))
    with pytest.raises(ValueError):
        allocate_discovery(ledger,query=7000,total_frames=7200,selected=[],previous_query=-1)


def test_parallel_contiguous_decode_preserves_exact_source_pixels_and_query_boundary(tmp_path):
    path=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24.,(48,32))
    assert writer.isOpened()
    for i in range(145):writer.write(np.full((32,48,3),i,dtype=np.uint8))
    writer.release()
    mapping=build_frame_mapping('video',145,fps=24.,sample_fps=1.,clip_len=16,overlap=4)
    outputs=[]
    for workers in [0,4]:
        reader=CoverageReader(path,mapping,workers=workers)
        images=reader.read_batch(list(range(6)),allowed_through=5,max_side=48)
        outputs.append([hashlib.sha256(i.tobytes()).hexdigest() for i in images])
        with pytest.raises(ValueError):reader.read_batch([6],allowed_through=5,max_side=48)
        reader.close()
    assert outputs[0]==outputs[1]


def test_dense_discovery_does_not_spend_slots_on_explicit_negative_hits():
    ledger={fid:dict(margin=2. if fid%2 else -2.,descriptor=pack_descriptor(descriptor(fid)))for fid in range(100)}
    selected,stats=allocate_discovery(ledger,query=99,total_frames=100,selected=[],previous_query=-1,positive_only=True)
    assert selected and all(ledger[f]['margin']>0 for f in selected)
    assert stats['negative_excluded']==50


def test_lossy_compact_ledger_does_not_change_dimensions_or_context_direction():
    a=descriptor()
    compact=unpack_descriptor(pack_descriptor(a,quantized=True))
    assert compact.shape==a.shape and np.max(np.abs(a-compact))<.005
    assert len(pack_descriptor(a,quantized=True))<len(pack_descriptor(a))


def test_evidence_serialization_preserves_aliases_and_actual_links_under_compression():
    state=CoverageSandbox('video','How many tv(s)?')
    for i in range(70):state.ingest(**actual(i*20,name='tv'if i%2 else'television'))
    class Tokenizer:
        def encode(self,text):return list(text)
    messages=[dict(role='user',content=[dict(type='text',text='original question')])]
    payload,audit=append_evidence(messages,state,'How many tv(s)?',Tokenizer(),
        max_text_tokens=1800,reserved_output_tokens=64,images=32)
    table=payload['hypotheses']
    assert len(table['rows'])==70 and audit['all_hypotheses_included']
    assert audit['format']=='columnar_records'
    assert audit['estimated_memory_tokens']>audit['text_allowance']  # fail explicitly; never drop IDs to pass
    assert {table['name_table'][r[0]]for r in table['rows']}=={'tv','television'}
    row=table['rows'][-1]
    assert table['missing_last_frame_equals_first'] and [row[1],row[2]if len(row)>2 else row[1]]==[1380,1380]


def test_weak_semantic_mode_avoids_repeating_the_shortlist_probe():
    from contextlib import contextmanager
    from sandbox_vlm.perception.coverage import score_visual_batch
    class Vision:
        @contextmanager
        def reuse_pixels(self,*args):yield
        @contextmanager
        def reuse(self,*args):yield [1]
    class Scorer:
        def prepare(self,question,images):return images
        def forward(self,images):return [0.]*len(images),{}
    ids,values,stats=score_visual_batch(Vision(),Scorer(),{},list(range(16)),'question',[0,4,8,12])
    assert ids==list(range(16))and stats['logical_language_image_requests']==20
    assert stats['next_batch_prefer_full']
    _,_,next_stats=score_visual_batch(Vision(),Scorer(),{},list(range(16)),'question',[0,4,8,12],force_full=True)
    assert next_stats['logical_language_image_requests']==16 and len(next_stats['language_calls'])==1
