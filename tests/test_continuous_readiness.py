import json
import tempfile
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from test_continuous_model import configuration, ingest, detection
from sandbox_vlm.continuous_model import ContinuousSandMem
from sandbox_vlm.continuous_preflight import sample_plan, validate_start, apply_input_profile
from sandbox_vlm.qa.budget import QABudget
from sandbox_vlm.perception.continuous_backend import ContinuousPerception, ContinuousPerceptionFailure


class ReadinessTests(unittest.TestCase):
    def answer(self, outputs, *, build=True):
        model=ContinuousSandMem('video',configuration())
        ingest(model,0,[detection(0)] if build else [detection(0,box=False)])
        if not build:ingest(model,1,[detection(1,name='desk')])
        calls=[]
        class Answerer:
            def answer(self,*a,**kw):
                calls.append(a)
                return dict(status='generated',raw_output=outputs[len(calls)-1])
        result=model.answer('object',maximum=1,available_through=model.last_frame_id,sparse=True,
            load_image=lambda _:Image.new('RGB',(16,16)),answerer=Answerer(),
            limits=QABudget(2,1280,8192,32768,256),audit=lambda *args:None,
            answer_validator=lambda value:value in ['A','B','C','D'])
        return result,calls

    def test_invalid_draft_cannot_veto_valid_uncited_verification(self):
        result,calls=self.answer(['FINAL_ANSWER: None of these options.', 'FINAL_ANSWER: B'])
        self.assertEqual(result['prediction'],'B')
        self.assertEqual(result['reason'],'draft_failed_use_verification')
        self.assertEqual(len(calls),2)

    def test_invalid_verification_cannot_replace_valid_draft_even_with_frame_citation(self):
        result,_=self.answer(['FINAL_ANSWER: A', 'F0 supports this.\nFINAL_ANSWER: None of the options.'])
        self.assertEqual(result['prediction'],'A')
        self.assertEqual(result['reason'],'verification_failed')

    def test_unrelated_provided_region_and_box_do_not_enable_two_pass(self):
        result,calls=self.answer(['FINAL_ANSWER: A'],build=False)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['prediction'],'A')

    def test_options_do_not_retrieve_unasked_entities(self):
        model=ContinuousSandMem('video',configuration())
        ingest(model,0,[detection(0)])
        ingest(model,1,[detection(1,name='desk')])
        prepared=model.prepare('Where is the chair?\nOptions:\nA. Desk\nB. Floor',maximum=2,available_through=1)
        self.assertEqual(prepared['anchor_ids'],[0])
        self.assertEqual(prepared['question'],'Where is the chair?\nOptions:\nA. Desk\nB. Floor')

    def test_streaming_wrong_entry_is_rejected_before_model_calls(self):
        result=validate_start(configuration(),plan=sample_plan(1.,600,1.),purpose='wiring',
                              protocol='streaming',answer_format='number')
        self.assertEqual(result['status'],'blocked')
        self.assertTrue(any('future' in reason for reason in result['problems']))

    def test_truncated_prefix_and_wiring_configuration_cannot_be_evaluation(self):
        result=validate_start(configuration(),plan=sample_plan(1.,600,1.,64),purpose='evaluation',
                              protocol='offline',answer_format='choice')
        self.assertEqual(result['status'],'blocked')
        self.assertTrue(any('truncated_prefix' in reason for reason in result['problems']))
        self.assertTrue(any('8_discovery' in reason for reason in result['problems']))

    def test_non_integer_fps_sampling_has_exact_full_vs_prefix_flag(self):
        plan=sample_plan(29.97,18001,1.)
        self.assertTrue(plan['full_video'])
        self.assertEqual(plan['frame_ids'],sorted(set(plan['frame_ids'])))
        self.assertEqual(sample_plan(29.97,18001,1.,64)['frame_ids'],plan['frame_ids'][:64])
        self.assertFalse(sample_plan(29.97,18001,1.,64)['full_video'])
        self.assertTrue(sample_plan(29.97,18001,1.,99999)['full_video'])

    def test_inherit_dense_32_frames_and_sparse_8_without_task_labels(self):
        profile=dict(sparse=False,real_frame_budget=32,real_image_side=384,max_output_tokens=64,
                     origin='input density, not task label')
        config=apply_input_profile(configuration(),profile)
        self.assertEqual(config['qa']['budget']['max_images'],33)
        self.assertEqual(config['qa']['real_frame_budget'],32)
        profile.update(sparse=True,real_frame_budget=8,real_image_side=640,max_output_tokens=256)
        config=apply_input_profile(configuration(),profile)
        self.assertEqual(config['qa']['budget']['max_images'],9)
        self.assertEqual(config['qa']['real_image_side'],640)

    def test_input_profile_rejects_answers_and_inconsistent_limits(self):
        profile=dict(sparse=False,real_frame_budget=32,real_image_side=384,max_output_tokens=256,origin='input')
        with self.assertRaisesRegex(ValueError,'density'):apply_input_profile(configuration(),profile)
        profile['answer']='A'
        with self.assertRaisesRegex(ValueError,'whitelisted'):apply_input_profile(configuration(),profile)

    def test_streamed_snapshot_preserves_complete_state_and_geometry_provenance(self):
        model=ContinuousSandMem('video',configuration())
        for i in range(20):ingest(model,i,[detection(i)] if i%3==0 else [])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'memory.json';model.write_snapshot(path)
            actual=json.loads(path.read_text());expected=model.export()
            actual['active_voxels'].sort();expected['active_voxels'].sort()
            self.assertEqual(actual,expected)
            with self.assertRaises(FileExistsError):model.write_snapshot(path)

    def test_discovery_failure_stops_before_sam_and_preserves_original_receipt(self):
        engine=ContinuousPerception.__new__(ContinuousPerception)
        engine.native_loader=lambda *a,**kw:torch.zeros(1,3,2,2)
        engine.stream=SimpleNamespace(step=lambda *a,**kw:dict(camera_pose=torch.zeros(1,9),
            pts3d_in_other_view=torch.zeros(1,2,2,3),conf=torch.ones(1,2,2),cost=dict(model_frame_steps=1)))
        engine.pose_decoder=lambda *a:(torch.eye(4)[None,None,:3,:],torch.eye(3)[None,None])
        engine.config=dict(reference='ref')
        engine.discovery=SimpleNamespace(discover=lambda *a,**kw:dict(status='perception_failed',raw_output='truncated'))
        attempts=[];engine.attempt=attempts.append
        with patch.object(torch.Tensor,'cuda',lambda self:self),patch('torch.autocast',return_value=nullcontext()):
            with self.assertRaises(ContinuousPerceptionFailure) as failure:
                engine.frame('not_opened.png',0,0,discover=True,available_through=0)
        self.assertEqual(attempts,['geometry','discovery'])
        self.assertEqual(failure.exception.receipt['discovery']['raw_output'],'truncated')


if __name__=='__main__':unittest.main()
