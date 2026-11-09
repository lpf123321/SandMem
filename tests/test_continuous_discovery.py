import json
import unittest
import torch
from sandbox_vlm.perception.record_boundary import RecordBoundary, complete_prefix
from sandbox_vlm.perception.discovery import parse_discovery

class Tokenizer:
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join('},' if i==300 else chr(i) for i in ids if i!=0)

class RecordBoundaryTests(unittest.TestCase):
    def record(self,name='chair',box=None):
        return dict(name=name,description='blue chair with braces } and [',bbox_2d=box or [1,2,3,4])

    def simulate(self,raw,cap=1536):
        tokenizer=Tokenizer();boundary=RecordBoundary(tokenizer,1,cap,165,[0]);ids=[]
        original=tokenizer.encode(raw)+[0];position=0
        for _ in range(cap):
            scores=torch.full((1,301),-100.)
            token=original[min(position,len(original)-1)]
            scores[0,token]=1.
            changed=boundary(torch.tensor([[1]+ids]),scores)
            chosen=int(changed.argmax())
            ids.append(chosen);position+=1
            if chosen==0:break
        return tokenizer.decode(ids),boundary

    def test_duplicate_finishes_valid_original_records_and_keeps_raw_duplicate(self):
        records=[self.record('table'),self.record(),self.record(),self.record()]
        raw=json.dumps(dict(frame_id=165,objects=records))
        text,boundary=self.simulate(raw)
        payload=json.loads(text)
        self.assertEqual(payload['objects'],records[:3])
        self.assertEqual(len(parse_discovery(text,165,(1000,1000))),3)
        self.assertEqual(boundary.events[0]['reason'],'complete_exact_record_repeated')
        self.assertFalse(boundary.events[0]['field_values_changed'])

    def test_original_markdown_fence_closed_as_syntax(self):
        raw='```json\n'+json.dumps(dict(frame_id=165,objects=[self.record(),self.record(),self.record()]))+'\n```'
        text,boundary=self.simulate(raw)
        self.assertTrue(text.endswith('\n```'))
        self.assertEqual(len(parse_discovery(text,165,(1000,1000))),2)

    def test_same_name_different_boxes_all_preserved(self):
        records=[self.record(box=[1,2,3,4]),self.record(box=[4,5,6,7])]
        raw=json.dumps(dict(frame_id=165,objects=records))
        text,boundary=self.simulate(raw)
        self.assertEqual(text,raw)
        self.assertEqual(boundary.events,[])

    def test_joint_brace_comma_token_cannot_leave_a_trailing_comma(self):
        record=json.dumps(self.record(),separators=(',',':'))
        prefix='{"frame_id":165,"objects":['+record+','+record[:-1]
        tokenizer=Tokenizer();boundary=RecordBoundary(tokenizer,1,1536,165,[0])
        ids=tokenizer.encode(prefix)
        for _ in range(5):
            scores=torch.full((1,301),-100.);scores[0,300]=1.
            chosen=int(boundary(torch.tensor([[1]+ids]),scores).argmax());ids.append(chosen)
            if chosen==0:break
        text=tokenizer.decode(ids)
        self.assertEqual(len(parse_discovery(text,165,(1000,1000))),2)
        self.assertEqual(json.loads(text)['objects'],[self.record(),self.record()])

    def test_partial_second_record_never_recovered_or_filled(self):
        prefix='{"frame_id":165,"objects":['+json.dumps(self.record())+',{"name":"unfinished'
        boundary=RecordBoundary(Tokenizer(),1,len(prefix)+3,165,[0])
        scores=torch.full((1,301),-100.);scores[0,ord('x')]=1.
        self.assertIs(boundary(torch.tensor([[1]+Tokenizer().encode(prefix)]),scores),scores)
        self.assertEqual(boundary.events,[])
        with self.assertRaises(ValueError):parse_discovery(prefix,165,(1000,1000))

    def test_wrong_frame_identity_is_not_rewritten(self):
        raw=json.dumps(dict(frame_id=166,objects=[self.record(),self.record()]))
        self.assertIsNone(complete_prefix(raw,165))

    def test_budget_closes_only_after_all_model_field_values_exist(self):
        record=json.dumps(self.record(),separators=(',',':'))
        prefix='{"frame_id":165,"objects":['+record[:-1]
        boundary=RecordBoundary(Tokenizer(),1,len(prefix)+5,165,[0]);ids=Tokenizer().encode(prefix)
        for _ in range(5):
            scores=torch.full((1,301),-100.);scores[0,ord('}')]=1.
            token=int(boundary(torch.tensor([[1]+ids]),scores).argmax());ids.append(token)
            if token==0:break
        self.assertEqual(json.loads(Tokenizer().decode(ids))['objects'],[self.record()])
        self.assertEqual(boundary.events[0]['reason'],'reserve_complete_record_and_JSON_closure')

    def test_empty_identified_list_is_valid_without_made_up_objects(self):
        raw='{"frame_id":165,"objects":[]}'
        text,boundary=self.simulate(raw)
        self.assertEqual(text,raw)
        self.assertEqual(parse_discovery(text,165,(1000,1000)),[])

if __name__=='__main__':unittest.main()
