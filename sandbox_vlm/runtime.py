"""Video source and streaming driver; inputs contain no benchmark answers."""
import io
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image

from sandbox_vlm.alignment.association import MatchSettings
from sandbox_vlm.data.video import build_frame_mapping, build_clips, read_source_frame, select_discovery_frames
from sandbox_vlm.data.vsi_super import plan_streaming_clips
from sandbox_vlm.memory.backend import BoundedSandboxBackend
from sandbox_vlm.pipeline import build_video
from sandbox_vlm.qa.budget import QABudget
from sandbox_vlm.qa.pipeline import run_question
from sandbox_vlm.qa.artifact import save_joint_input, state_sha256


class VideoFrameReader:
    def __init__(self, path, mapping):
        self.path, self.mapping = Path(path), mapping
        stat = self.path.stat()
        self.identity = (stat.st_size, stat.st_mtime_ns)
        self.allowed_through = -1

    def available_frame_ids(self):
        return [f.frame_id for f in self.mapping.frames if f.frame_id <= self.allowed_through]

    def encoded_source(self, fid):
        import cv2
        if type(fid) is not int or fid < 0 or fid > self.allowed_through:
            raise ValueError('future or unavailable frame requested')
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.identity:
            raise ValueError('source video changed')
        ref = self.mapping.by_id(fid)
        pixels = cv2.cvtColor(read_source_frame(self.path, ref.source_index), cv2.COLOR_BGR2RGB)
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format='JPEG', quality=100)
        return encoded.getvalue(), ref, list(pixels.shape[:2])

    def read(self, fid, *, max_side):
        content, ref, size = self.encoded_source(fid)
        with Image.open(io.BytesIO(content)) as opened:
            image = opened.convert('RGB')
            image.thumbnail((max_side,max_side), Image.Resampling.LANCZOS)
        return image, dict(frame_id=fid, source_index=ref.source_index,
            source_sha256=hashlib.sha256(content).hexdigest(), source_image_size_hw=size,
            input_image_size_hw=[image.height,image.width])

    def encoded_sequence(self, ids):
        """Yield at most 16 chronological source JPEGs with one decoder.

        No persistent pixels or cross-query cache. The caller must consume or
        close the iterator before leaving its bounded clip.
        """
        import cv2
        if (not ids or len(ids) > 16 or any(type(fid) is not int for fid in ids)
                or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] > self.allowed_through):
            raise ValueError('invalid, oversized or future source sequence')
        refs = [self.mapping.by_id(fid) for fid in ids]
        if any(a.source_index >= b.source_index for a, b in zip(refs, refs[1:])):
            raise ValueError('source sequence must be strictly chronological')
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.identity:
            raise ValueError('source video changed')
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise IOError('cannot open source sequence')
            capture.set(cv2.CAP_PROP_POS_FRAMES, refs[0].source_index)
            position = refs[0].source_index - 1
            for fid, ref in zip(ids, refs):
                stat = self.path.stat()
                if (stat.st_size, stat.st_mtime_ns) != self.identity or fid > self.allowed_through:
                    raise ValueError('source changed or query watermark moved')
                while position < ref.source_index:
                    if not capture.grab():
                        raise IOError('cannot decode source sequence')
                    position += 1
                ok, pixels = capture.retrieve()
                if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != ref.source_index + 1:
                    raise IOError('source sequence frame identity mismatch')
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, format='JPEG', quality=100)
                yield encoded.getvalue(), ref, list(pixels.shape[:2])
        finally:
            capture.release()


def run_video_example(example, video_path, config, *, discovery, tracker, geometry, answerer, output):
    """No scoring/GT access. This entry must only be invoked for authorized runs."""
    import cv2
    if config['perception']['mllm'].get('question_focus', False):
        raise ValueError('question-focused perception config cannot run the Proposal mapping path')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError('cannot open source video')
    count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    import math
    if count < 1 or not math.isfinite(fps) or fps <= 0 or config['video']['sample_fps'] != 1.:
        raise ValueError('invalid video metadata or official 1 FPS sampling mismatch')
    mapping = build_frame_mapping(example.video_path, count, fps=fps, sample_fps=1.,
        clip_len=config['video']['clip_len'], overlap=config['video']['overlap'])
    mapping.save(output/'frame_mapping.json')
    reader = VideoFrameReader(video_path, mapping)
    backend = BoundedSandboxBackend(config['memory']['backend_max_bytes'])
    if example.streaming:
        schedule = plan_streaming_clips(mapping.frames, example.query_frame_ids,
            clip_len=mapping.clip_len, overlap=mapping.overlap)
    else:
        schedule = [(len(mapping.frames)-1,build_clips(mapping.frames,mapping.clip_len,mapping.overlap))]
    first, answers, query_records, mapping_status = True, [], [], 'ok'
    for query, clips in schedule:
        reader.allowed_through = query
        for clip in clips:
            with tempfile.TemporaryDirectory(prefix='clip-',dir=output) as temporary:
                root = Path(temporary)
                frames=[]
                for ref in clip.frames:
                    data, _, size = reader.encoded_source(ref.frame_id)
                    path = root/f'{ref.frame_id:06d}.jpg'; path.write_bytes(data)
                    frames.append(dict(**ref.to_dict(), path=path.name, image_size_hw=size,
                                       sha256=hashlib.sha256(data).hexdigest()))
                selected=select_discovery_frames(clip,config['perception']['discovery']['frames_per_clip'],
                    config['perception']['discovery']['strategy'])
                manifest=dict(video_id=example.video_path, clips=[dict(clip_id=clip.clip_id, frames=frames,
                    discovery_frame_ids=[f.frame_id for f in selected])])
                def audit(cid,row,observations,cameras):
                    folder=output/f'clip_{cid:05d}';folder.mkdir()
                    for name,value in [('report',row),('observations',observations),('cameras',cameras)]:
                        (folder/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
                base, result=build_video(root,manifest,config,backend=backend,discovery=discovery,
                    tracker=tracker,geometry=geometry,audit_sink=audit,resume=not first)
                first=False
                if result['status']!='ok':
                    mapping_status=result['status']
                if mapping_status=='capacity_limited':
                    return dict(status='capacity_limited',sample_id=example.sample_id,query_records=query_records,
                                prediction=None, stopped_at_frame=clip.frames[-1].frame_id)
        base=backend.read_working_copy()
        # Even if a future selection bug is introduced, the source reader still
        # prevents every access beyond this query's watermark.
        for view in base.all_viewpoints():
            if view.frame_id > query:
                raise ValueError('future viewpoint in streaming snapshot')
        folder=output/f'query_{query:06d}';folder.mkdir()
        base_sha = state_sha256(base)
        (folder/'base_sandbox.json').write_text(json.dumps(base.to_dict(),ensure_ascii=False,indent=2))
        def evidence(images,metadata,messages):
            save_joint_input(folder, images, metadata, messages)
        qa, working=run_question(base,example.model_question,reader,QABudget(**config['qa']['budget']),
            answerer,MatchSettings(**config['alignment']['match']),evidence_sink=evidence,backend=backend,
            retrieval_question=example.original_question,
            max_targets=config['qa'].get('selection', {}).get('max_targets'),
            citation_policy=config['qa']['mllm'].get('citation_policy','implicit'),
            output_constraint=config['qa']['mllm'].get('output_constraint','none'))
        qa.update(original_dataset_question=example.original_question,query_frame_id=query)
        if state_sha256(base) != base_sha:
            raise ValueError('question revisions changed the reusable video base sandbox')
        qa['mapping_contract'] = dict(path='proposal_four_module', question_conditioned_mapping=False,
            base_sandbox_sha256=base_sha, base_state_unchanged=True,
            scope='execution_path_only_not_long_video_mechanism_acceptance')
        (folder/'result.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
        answer=qa.get('parsed_answer',{}).get('answer') if qa['status']=='ok' else None
        if example.streaming and answer is not None:
            import math
            try:
                answer=float(answer)  # Only exact numeric formatting; no regex/solver/model retry.
                if not math.isfinite(answer):
                    answer=None
            except (ValueError,TypeError):
                answer=None
        answers.append(answer)
        query_records.append(dict(query_frame_id=query,status=qa['status'],mapping_status=mapping_status,
            input_contract=qa['evidence']['input_contract'], base_sandbox_sha256=base_sha))
        del working
    prediction=json.dumps(answers) if example.streaming else answers[0]
    return dict(status='ok' if mapping_status=='ok' and all(a is not None for a in answers) else 'incomplete',
        sample_id=example.sample_id,prediction=prediction,query_records=query_records,
        answer_count=len(answers),mapping_status=mapping_status)
