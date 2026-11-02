"""D044: bounded, frozen two-observation visual cue; never geometry truth."""
import hashlib
import io
import math
import time

from PIL import Image

from sandbox_vlm.perception.evidence import PromptEvidence


def observation_crop(reader, evidence, *, max_side=256):
    actual = PromptEvidence.from_dict(evidence)
    content, ref, size = reader.encoded_source(actual.frame_id)
    if size != list(actual.source_hw):
        raise ValueError('actual observation source dimensions changed')
    x0, y0, x1, y1 = actual.bbox_xyxy
    # Fixed 10% local padding supplies boundaries without another image input.
    dx, dy = (x1-x0)*.1, (y1-y0)*.1
    box = [max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
           min(size[1], math.ceil(x1+dx)), min(size[0], math.ceil(y1+dy))]
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert('RGB').crop(box)
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image, dict(evidence=actual.to_dict(), source_index=ref.source_index,
        source_sha256=hashlib.sha256(content).hexdigest(), crop_xyxy=box,
        input_image_size_hw=[image.height, image.width],
        input_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())


def pair_messages(images, names, descriptions):
    text = ('These are two actual local observations from one video, in chronological order. '
        'Do they show the same individual physical object, rather than merely the same kind of object? '
        'Use distinctive appearance and visible surrounding details. Similar ordinary objects '
        'can be different individuals. If identity is uncertain, answer No. '
        'Reply with exactly Yes or No.\n'
        f'Observation A: {names[0]}; {descriptions[0]}\n'
        f'Observation B: {names[1]}; {descriptions[1]}')
    return [dict(role='user', content=[dict(type='image', image=image) for image in images]
                 + [dict(type='text', text=text)])]


class VisualPairVerifier:
    def __init__(self, answerer, reader, *, maximum_pairs=16, margin_min=2., image_side=256, audit_sink=None):
        if (type(maximum_pairs) is not int or not 1 <= maximum_pairs <= 16 or
                image_side != 256 or not math.isfinite(margin_min) or margin_min < 2.):
            raise ValueError('D044 requires <=16 pairs, 256px, margin >=2')
        from sandbox_vlm.qa.relevance import FrozenRelevanceScorer
        self.answerer, self.reader = answerer, reader
        self.token_ids = FrozenRelevanceScorer(answerer).token_ids
        self.maximum_pairs, self.margin_min, self.image_side = maximum_pairs, margin_min, image_side
        self.records, self.audit_sink = [], audit_sink
        self.forward_calls = 0

    def to_dict(self):
        return dict(representation='D044_bounded_visual_pair_cues', maximum_pairs=self.maximum_pairs,
                    margin_min=self.margin_min, image_side=self.image_side, records=self.records)

    def verify(self, previous, current, *, names, descriptions, candidate_id, allowed_through):
        import torch
        a, b = PromptEvidence.from_dict(previous), PromptEvidence.from_dict(current)
        if max(a.frame_id, b.frame_id) > allowed_through:
            raise ValueError('future pair observation')
        if a.frame_id == b.frame_id:
            return dict(accepted=False, status='same_frame_forbidden', forward_calls=0)
        if len(self.records) >= self.maximum_pairs:
            return dict(accepted=False, status='pair_budget_exhausted', forward_calls=0)
        # Count attempts, including I/O or model failures, without retry.
        record = dict(pair_index=len(self.records), candidate_id=candidate_id,
            previous=previous, current=current, names=list(names), descriptions=list(descriptions),
            allowed_through=allowed_through, accepted=False, status='started', forward_calls=0,
            generated_tokens=0, geometry_confirmation=False)
        self.records.append(record)
        started = time.perf_counter()
        forwards_before = self.forward_calls
        images = []
        try:
            order = sorted(range(2), key=lambda i: [a, b][i].frame_id)
            inputs = [observation_crop(self.reader, [previous, current][i], max_side=self.image_side) for i in order]
            images = [p[0] for p in inputs]
            messages = pair_messages(images, [names[i] for i in order], [descriptions[i] for i in order])
            record['images'] = [p[1] for p in inputs]
            record['prompt'] = messages[0]['content'][-1]['text']
            record.update(self.score_images(images, [names[i] for i in order], [descriptions[i] for i in order]))
        except Exception as exc:
            record.update(status='verification_failed', accepted=False, error=str(exc))
        finally:
            record['wall_seconds'] = time.perf_counter()-started
            record['forward_calls'] = self.forward_calls-forwards_before
            if self.audit_sink is not None:
                self.audit_sink(record, images)
        return dict(accepted=record['accepted'], status=record['status'], pair_index=record['pair_index'],
                    margin=record.get('margin'), forward_calls=record['forward_calls'])

    def score_images(self, images, names, descriptions):
        """One forward; also used by the bounded independent positive/negative diagnostic."""
        import torch
        if len(images) != 2 or any(max(i.size) > self.image_side for i in images):
            raise ValueError('two bounded actual crops required')
        messages = pair_messages(images, names, descriptions)
        processor_started = time.perf_counter()
        processor = self.answerer.processor
        prepared = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        result = dict(processor_seconds=time.perf_counter()-processor_started,
            image_grid_thw=prepared['image_grid_thw'].tolist(),
            input_tokens=int(prepared['attention_mask'].sum()),
            visual_tokens=int((prepared['input_ids']==self.answerer.model.config.image_token_id).sum()),
            prompt=messages[0]['content'][-1]['text'], forward_calls=1, generated_tokens=0)
        result['processed_image_sizes_hw'] = [[r[1]*processor.image_processor.patch_size,
            r[2]*processor.image_processor.patch_size] for r in result['image_grid_thw']]
        expected = sum(t*h*w//processor.image_processor.merge_size**2 for t,h,w in result['image_grid_thw'])
        if len(result['image_grid_thw']) != 2 or result['input_tokens'] > 4096 or result['visual_tokens'] != expected:
            raise ValueError('pair input contract exceeded')
        prepared = prepared.to(self.answerer.device)
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        self.forward_calls += 1
        with torch.inference_mode():
            output = self.answerer.model(**prepared, use_cache=False, logits_to_keep=1)
            values = output.logits[0, -1, self.token_ids].float().cpu().tolist()
        torch.cuda.synchronize()
        margin = float(values[0]-values[1])
        if not math.isfinite(margin):
            raise ValueError('nonfinite pair logits')
        result.update(status='verified', yes_no_logits=values, yes_no_token_ids=self.token_ids,
            margin=margin, accepted=margin >= self.margin_min, model_seconds=time.perf_counter()-model_started)
        return result
