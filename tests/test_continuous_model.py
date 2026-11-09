import json
from pathlib import Path
from copy import deepcopy
import unittest

import numpy as np
from PIL import Image

from sandbox_vlm.continuous_model import ContinuousSandMem
from sandbox_vlm.memory.geometry import CameraParams, BBox3D
from sandbox_vlm.qa.budget import QABudget


def configuration():
    c = json.loads((Path(__file__).resolve().parents[1] / 'configs/continuous_model_20261109.json').read_text())
    c['points'] = dict(confidence_quantile=.5, grid_stride=1, max_points=4, min_points=2)
    return c


def detection(fid, feature=(1, 0), name='chair', box=True, x=0):
    return dict(name=name, description='complete stored description', appearance=np.array(feature),
                evidence=dict(frame_id=fid,bbox_xyxy=[x, 0, x+1, 1],source_hw=[10, 10],sam_mask_nonempty=True),
                bbox_3d=BBox3D([1, 1, 1], [1, 1, 1], np.eye(3), 'ref').to_dict() if box else None)


def ingest(m, fid, rows=()):
    pose = np.eye(4);pose[0, 3] = fid
    return m.ingest(camera=CameraParams(fid, pose, np.eye(3), (2, 2), 'ref'),
                    descriptor=[np.cos(fid),np.sin(fid)],
                    world_points=np.array([[[0,0,1],[1,0,1]],[[0,1,1],[1,1,1]]]),
                    confidence=np.ones((2,2)),detections=list(rows),available_through=fid)


class ModelTests(unittest.TestCase):
    def model(self):
        return ContinuousSandMem('video', configuration())

    def test_joint_assignment_avoids_greedy_collision_and_uses_no_cross_frame_iou(self):
        m=self.model()
        unit=lambda degrees:(np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees)))
        ingest(m,0,[detection(0,unit(0)),detection(0,unit(40))])
        report=ingest(m,1,[detection(1,unit(20),x=8),detection(1,unit(0),x=7)])
        self.assertEqual(report['assigned_instance_ids'],[1,0])
        self.assertEqual(len(m.instances),2)

    def test_different_caption_unmatched_and_ids_not_reused(self):
        m=self.model();ingest(m,0,[detection(0)])
        report=ingest(m,1,[detection(1,name='desk')])
        self.assertEqual(report['assigned_instance_ids'],[1])
        m._prune({'F1_O0'})
        report=ingest(m,2,[detection(2)])
        self.assertEqual(report['assigned_instance_ids'],[2])

    def test_same_frame_one_to_one_allows_unmatched_birth(self):
        m=self.model();ingest(m,0,[detection(0)])
        report=ingest(m,1,[detection(1),detection(1)])
        self.assertEqual(len(set(report['assigned_instance_ids'])),2)
        self.assertEqual(len(m.instances),2)

    def test_compaction_deletes_middle_geometry_anchor_without_archival_restore(self):
        m=self.model()
        ingest(m,0,[detection(0,box=False)]);ingest(m,1,[detection(1)]);ingest(m,2,[detection(2)])
        after=deepcopy(m);after._prune({'F0_O0','F2_O0'})
        self.assertLess(after.structure_bytes(),m.structure_bytes())
        m.config['memory_limit_bytes']=(after.structure_bytes()+m.structure_bytes())//2
        m.enforce_capacity()
        self.assertEqual(set(m.observations),{'F0_O0','F2_O0'})
        self.assertEqual(m.instances[0]['boxes'],{})
        self.assertEqual(len(m.frames),3)
        self.assertEqual([o.observation_id for s in m.scenes.scenes for o in s.observations],['F0_O0','F2_O0'])

    def test_oldest_last_observed_evicted_first(self):
        m=self.model();ingest(m,0,[detection(0)]);ingest(m,1,[detection(1,name='desk')])
        after=deepcopy(m);after._prune({'F1_O0'})
        m.config['memory_limit_bytes']=(after.structure_bytes()+m.structure_bytes())//2
        result=m.enforce_capacity()
        self.assertEqual(result['evicted_instance_ids'],[0])
        self.assertEqual(set(m.instances),{1})
        self.assertEqual(m.frames[0]['observation_ids'],[])

    def test_irreducible_frames_fail_visibly(self):
        m=self.model();ingest(m,0)
        m.config['memory_limit_bytes']=1
        with self.assertRaisesRegex(MemoryError,'sampled-frame'):
            m.enforce_capacity()

    def test_original_complete_fields_and_non_target_middle_selected(self):
        m=self.model()
        for fid in range(9):ingest(m,fid,[detection(fid)] if fid in [0,8] else [])
        prepared=m.prepare('Where is the chair?',maximum=6,available_through=8)
        self.assertEqual(prepared['payload']['records'][0],
                         [0,'chair','complete stored description',[0,8]])
        self.assertIn(4,prepared['selection']['frame_ids'])
        self.assertFalse(any(r[1]==4 for r in prepared['payload']['provided_frame_regions']))
        self.assertIn([4,1],prepared['payload']['frame_scene_links'])

    def test_real_render_and_messages_reach_answerer_and_do_not_mutate_memory(self):
        m=self.model()
        for fid in range(9):ingest(m,fid,[detection(fid)] if fid in [0,8] else [])
        before=json.dumps(m.export(),sort_keys=True)
        calls=[];audits={}
        class Answerer:
            def answer(self,messages,images,limits,**kwargs):
                calls.append((messages, len(images)))
                if len(calls)==2:
                    self_image=messages[0]['content'][2]['image']
                    assert self_image is images[0] and self_image.width==1280
                return dict(status='generated',raw_output='F0 observed.\nFINAL_ANSWER: A')
        result=m.answer('Where is the chair?',maximum=6,available_through=8,sparse=True,
            load_image=lambda fid:Image.new('RGB',(32,24),(fid,0,0)),answerer=Answerer(),
            limits=QABudget(7,1280,8192,32768,256),audit=lambda name,value:audits.update({name:value}))
        self.assertEqual([n for _,n in calls],[6,7])
        self.assertEqual(result['prediction'],'A')
        self.assertEqual(audits['prepared']['rendering']['panels'][0]['trajectory_frame_ids'],list(range(9)))
        self.assertEqual(before,json.dumps(m.export(),sort_keys=True))

    def test_no_geometry_links_still_inputs_map_once(self):
        m=self.model()
        for i in range(3):ingest(m,i)
        calls=[]
        class Answerer:
            def answer(self,messages,images,limits,**kwargs):
                calls.append(len(images));return dict(status='generated',raw_output='FINAL_ANSWER: A')
        result=m.answer('Original question',maximum=2,available_through=2,sparse=True,
            load_image=lambda fid:Image.new('RGB',(32,32)),answerer=Answerer(),
            limits=QABudget(3,1280,8192,32768,256),audit=lambda *args:None)
        self.assertEqual(calls,[3])
        self.assertEqual(result['prediction'],'A')

    def test_future_memory_or_frames_rejected(self):
        m=self.model();ingest(m,0)
        with self.assertRaisesRegex(ValueError,'causal'):
            m.prepare('chair',maximum=1,available_through=-1)

    def test_invalid_requested_answer_format_is_preserved_as_failure_without_retry(self):
        m=self.model();ingest(m,0)
        calls=[]
        class Answerer:
            def answer(self,*args,**kwargs):
                calls.append(1)
                return dict(status='generated',raw_output='FINAL_ANSWER: None of the options are correct.')
        result=m.answer('Question',maximum=1,available_through=0,sparse=True,
            load_image=lambda _:Image.new('RGB',(32,32)),answerer=Answerer(),
            limits=QABudget(2,1280,8192,32768,256),audit=lambda *args:None,
            answer_validator=lambda value:value in ['A','B','C','D'])
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error'],'answer_contract_violation')
        self.assertEqual(len(calls),1)
        self.assertEqual(result['prediction'],'None of the options are correct.')


if __name__=='__main__':unittest.main()
