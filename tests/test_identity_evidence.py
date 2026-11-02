"""Evidence boundaries for the pre-approved D095 candidate; no model mocks as results."""
from copy import deepcopy
import json
import numpy as np
from PIL import Image
import pytest
from sandbox_vlm.memory.identity_evidence import (parse_unit,unit_images,region_key,mask_iou,
 unique_candidates,verify_pair,supported_components,pack_mask,unpack_mask,UNIT_PATTERN)


def region():return dict(frame_id=7,source_hw=[12,16],bbox_xyxy=[2.2,3.2,8.8,9.8])


def test_exact_crop_no_padding_and_native_source_not_modified():
 source=Image.fromarray(np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3))
 before=source.tobytes();marked,crop=unit_images(source,region())
 assert source.tobytes()==before and crop.size==(7,7)
 assert crop.tobytes()==source.crop((2,3,9,10)).tobytes()
 assert marked.tobytes()!=before
 with pytest.raises(ValueError,match='shape'):
  unit_images(Image.new('RGB',(17,12)),region())


@pytest.mark.parametrize('status',['one_object','part','multiple_objects','unclear'])
def test_strict_unit_schema(status):
 assert parse_unit(json.dumps(dict(unit_status=status)))==status
 with pytest.raises(ValueError):parse_unit(json.dumps(dict(unit_status=status,extra=1)))
 with pytest.raises(ValueError):parse_unit('{"unit_status":"same_object"}')


def test_masks_and_frame_support_must_be_real_and_complete():
 a=np.ones((4,4),dtype=bool);b=a.copy();b[:3]=False
 assert mask_iou(a,b)==.25
 with pytest.raises(ValueError):mask_iou(a,a.astype(float))
 with pytest.raises(ValueError):verify_pair({0:a},{0:a},[0,1],['b'],['a'],'a','b')
 assert np.array_equal(unpack_mask(pack_mask(b)),b)


def test_alternative_region_cannot_be_removed_by_name_and_prevents_unique_match():
 mask=np.ones((3,3),dtype=bool)
 assert unique_candidates(mask,[('target',mask),('other_name',mask)])==['other_name','target']
 a={0:mask,1:mask}
 result=verify_pair(a,a,[0,1],['other_name','target'],['source'],'source','target')
 assert result['status']=='identity_unresolved'


def test_all_shared_frames_and_both_directions_are_necessary():
 one=np.ones((3,3),dtype=bool);zero=np.zeros((3,3),dtype=bool)
 a={0:one,1:one};b={0:one,1:zero}
 assert verify_pair(a,b,[0,1],['b'],['a'],'a','b')['status']=='identity_unresolved'
 assert verify_pair(a,a,[0,1],['b'],[],'a','b')['status']=='identity_unresolved'
 assert verify_pair(a,a,[0,1],['b'],['a'],'a','b')['status']=='supported_local_relation'


def test_conflicts_reject_whole_component_not_one_inconvenient_edge():
 edges=[('a','b'),('b','c'),('a','c')]
 result=supported_components(edges,[('a','c')])
 assert not result['accepted'] and result['rejected'][0]['members']==['a','b','c']
 assert result['rejected'][0]['reason']=='co_visible_conflict'


def test_transitivity_without_direct_evidence_does_not_confirm_component():
 result=supported_components([('a','b'),('b','c')],[])
 assert not result['accepted'] and result['rejected'][0]['missing_direct_edges']==[('a','c')]
 assert supported_components([('a','b')],[])['accepted']==[['a','b']]


def test_actual_region_identity_does_not_union_whole_parent_track():
 a=region();b=deepcopy(a);b['frame_id']=8
 assert region_key(a)!=region_key(b)
 result=supported_components([(region_key(a),region_key(b))],[])
 assert result['identity_scope']=='actual_regions_only_not_parent_tracks'
 assert 'parent' not in str(result['accepted'])


def test_unit_regex_accepts_only_frozen_enumeration():
 from lmformatenforcer import RegexParser
 def accepts(raw):
  parser=RegexParser(UNIT_PATTERN)
  for c in raw:
   if c not in parser.get_allowed_characters():return False
   parser=parser.add_character(c)
  return parser.can_end()
 assert accepts('{"unit_status":"one_object"}')
 assert accepts('{"unit_status":"multiple_objects"}')
 assert not accepts('{"unit_status":"same_object"}')
 assert not accepts('{"unit_status":"one_object","extra":1}')


def test_processor_grid_upsampling_preserves_supplied_image_and_token_contract():
 from sandbox_vlm.qa.budget import QABudget
 from sandbox_vlm.qa.unit_budget import check_unit_budget
 limits=QABudget(2,384,2048,1024,64)
 kwargs=dict(image_sizes_hw=[[288,384],[256,110]],processed_image_sizes_hw=[[288,384],[416,192]],input_tokens=385,visual_tokens=186,context_limit=262144)
 assert check_unit_budget(limits,**kwargs)['status']=='ok'
 assert check_unit_budget(limits,**dict(kwargs,visual_tokens=2049,input_tokens=2248))['status']=='budget_exceeded'
 assert check_unit_budget(limits,**dict(kwargs,image_sizes_hw=[[288,384],[300,110]]))['status']=='budget_exceeded'


def test_unit_transport_checks_approved_budget_before_any_model_forward():
 import torch
 from types import SimpleNamespace
 from scripts.run_identity_first import unit_answer
 from sandbox_vlm.qa.budget import QABudget
 class Processor:
  image_processor=SimpleNamespace(merge_size=2,patch_size=16)
  def apply_chat_template(self,*args,**kwargs):
   return dict(input_ids=torch.tensor([[7,7,5]]),attention_mask=torch.ones((1,3),dtype=torch.long),image_grid_thw=torch.tensor([[1,2,2],[1,2,2]]),pixel_values=torch.zeros((1,1)))
 model=SimpleNamespace(processor=Processor(),model=SimpleNamespace(config=SimpleNamespace(image_token_id=7,text_config=SimpleNamespace(max_position_embeddings=262144))))
 report=dict(unit_attempts=0)
 result=unit_answer(model,[],[Image.new('RGB',(400,400)),Image.new('RGB',(20,20))],QABudget(2,384,2048,1024,64),report)
 assert result['status']=='budget_exceeded' and result['model_calls']==0 and report['unit_attempts']==0
