"""Read-only post-score audit. No model calls, prompt changes or rescoring repairs."""
import argparse,collections,hashlib,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/mindcube_official_20261107'))
from src.evaluation.core.extractors import extract_answer


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(run,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():raise FileExistsError('preserve prior audit')
    manifest=read(run/'manifest.json');refs=read(run/'mind_references.json');scored=read(run/'mindcube_scores.json')
    inputs={r['id']:r for r in manifest['inputs'] if r['dataset']=='mindcube'}
    assert len(refs)==len(inputs)==527 and {r['id'] for r in refs}==set(inputs)
    video={(v['dataset'],v['scene_name']):v for v in manifest['videos']}
    counts=collections.Counter();by=collections.defaultdict(collections.Counter);examples=[];rows=[];invalid=[];hist=collections.Counter();random_sum=0
    sources={};different=[];numbering_counts=collections.Counter();native=0
    for r in refs:
        item=inputs[r['id']];folder=run/('question_'+r['id']);result=read(folder/'result.json')
        assert read(folder/'input.json')==item and item['question']==r['question'] and item['images']==r['images']
        assert result['question_id']==r['id']
        parsed=extract_answer(result['raw_output']);correct=parsed==r['gt_answer'];counts['correct']+=correct
        group=r['id'].split('_')[0];by[group]['n']+=1;by[group]['correct']+=correct
        choices=re.findall(r'\b([A-E])\.\s+',r['question']);assert choices==list('ABCDE')[:len(choices)]
        random_sum+=1/len(choices);hist[len(choices)]+=1;by[group]['chance_sum_million']+=round(1/len(choices)*1000000)
        expected=list(range(len(r['images'])));actual=result['actual_frame_ids'];missing=sorted(set(expected)-set(actual))
        if missing:counts['missing_views']+=1;by[group]['missing_views']+=1
        if actual!=expected and not missing:counts['reordered_views']+=1
        counts['sparse' if result['sparse'] else 'dense']+=1
        if missing:counts['missing_sparse' if result['sparse'] else 'missing_dense']+=1
        mapping=read(folder/'mapping.json')['frames'];assert [x['source_index'] for x in mapping]==expected
        v=video[('mindcube',item['scene_name'])];source=read(v['source_video'])['source_images']
        assert len(source)==len(expected)
        for name,s in zip(r['images'],source):
            assert Path(s['path']).as_posix().endswith('/'+name)
            sources[s['path']]=s['sha256']
        stage='verification' if result['two_pass'] else 'joint'
        messages=read(folder/(stage+'_messages.json'));content=messages[0]['content']
        image_blocks=[b for b in content if b['type']=='image']
        assert len(image_blocks)==len(actual)+1,'map or source image missing from actual message'
        assert [b['evidence_image_index'] for b in image_blocks]==list(range(len(actual)+1))
        texts='\n'.join(b['text'] for b in content if b['type']=='text')
        assert item['formatted_question'] in texts
        generation=read(folder/(stage+'_generation.json'))
        traces=generation.get('actual_forward_trace',[])
        assert len(traces)==1 and traces[0]['component']=='actual_qwen_visual_encoder'
        assert len(traces[0]['image_grid_thw'])==len(image_blocks);native+=1
        # Keep exact message labels. Absence of an explicit original-image map
        # is a compatibility finding, not proof of which indexing the VLM used.
        labels=[b['text'] for b in content if b['type']=='text' and re.search(r'frame\s*(?:ID)?\s*\d+',b['text'],re.I) and len(b['text'])<250]
        if any('Actual source frame ID 0:' in label for label in labels):numbering_counts['uses_zero_based_frame_zero']+=1
        if re.search(r'(?:original|benchmark)\s+(?:image|view)\s*1',texts,re.I):numbering_counts['explicit_original_image_one_map']+=1
        row=dict(id=r['id'],group=group,original_views=len(expected),actual_frame_ids=actual,missing_frame_ids=missing,
            actual_answer=parsed,correct=correct,random_chance=1/len(choices),sparse=result['sparse'],two_pass=result['two_pass'])
        rows.append(row)
        if parsed is None:
            evidence={}
            for name in ['draft','verification','joint']:
                p=folder/(name+'_generation.json')
                if p.exists():
                    g=read(p);evidence[name]=dict(raw_output=g.get('raw_output'),status=g['status'])
            invalid.append(dict(id=r['id'],raw_prediction=result['raw_output'],generations=evidence))
        if group=='rotation' and missing and len(examples)<3:
            examples.append(dict(**row,question=r['question'],message_frame_labels=labels,
                image_blocks=image_blocks,first_text=content[0]['text']))
    for path,h in sources.items():
        if sha(path)!=h:different.append(path)
    assert not different
    assert counts['correct']==scored['official']['results']['gen_cogmap_correct']==168
    assert len(invalid)==len(scored['official']['error_cases']['gen_cogmap_error'])==15
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    details=dict(status='audit_completed_no_model_calls',questions=527,correct=168,accuracy=168/527,
        official_score_confirmed=True,full_selected_denominator=True,invalid_answers_wrong=len(invalid),
        groups={k:dict(v) for k,v in by.items()},findings=dict(counts),question_option_count_histogram=dict(hist),
        exact_uniform_random_expectation=random_sum/527,expected_random_correct=random_sum,
        maximum_if_only_15_invalid_fixed=(168+15)/527,numbering=dict(numbering_counts),
        source_images_verified=len(sources),source_image_hash_mismatches=different,
        question_manifest_and_original_image_list_verified=527,actual_Qwen_QA_encoder_image_count_verified=native,
        source_freeze_verified=True,zero_GPU_calls=True,
        interpretation='Missing designated views and absent original image-number mapping are confirmed compatibility gaps; effect on accuracy is not causally measured. Low score itself does not prove a scorer bug. No fixed-output guarantee above random.',
        examples=examples,reference_sha256=sha(run/'mind_references.json'),prior_scores_sha256=sha(run/'mindcube_scores.json'))
    (out/'audit.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    (out/'per_question.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'invalid_output_evidence.json').write_text(json.dumps(invalid,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in details.items() if k not in ('examples','groups')},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run.resolve(),a.output.resolve())
