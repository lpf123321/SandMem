"""Vgent inference logic with a separately audited Spatial transport.

The original class is compiled from the pinned upstream source. Changes are
limited to media provenance at transport boundaries and returning raw final
text for the common VSI-Bench scorer (including numerical questions).
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import types


def load_port_class(repo):
    import numpy as np
    import torch
    from collections import defaultdict
    repo = Path(repo)
    # Imports from the upstream model utilities do not instantiate any model.
    spec = importlib.util.spec_from_file_location('vgent_native_video_utils', repo / 'models/utils.py')
    video_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video_utils)
    namespace = dict(np=np, torch=torch, json=json, re=re,
                     defaultdict=defaultdict, resize_video=video_utils.resize_video)
    prompts = repo / 'utils/prompts.py'
    exec(compile(prompts.read_text(), str(prompts), 'exec'), namespace)
    # Avoid importing an unrelated model namespace or instantiating old weights.
    helpers = repo / 'utils/retrieval.py'
    tree = ast.parse(helpers.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(helpers), 'exec'), namespace)
    source = repo / 'utils/vgent.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Vgent')
    changes = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for call in ast.walk(fn):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'mllm_response' and len(call.args) >= 6):
                continue
            media = call.args[5]
            if not isinstance(media, ast.Name):
                continue
            expression = None
            if fn.name == 'refine_nodes' and media.id == 'video_input':
                expression = 'self.source_frame_ids[node*self.args.chunk_size:node*self.args.chunk_size+len(video_input)]'
            elif fn.name == 'aggregate_nodes' and media.id == 'video_segments':
                expression = '[self.source_frame_ids[int(i)] for i in indices]'
            if expression:
                call.keywords.append(ast.keyword(arg='source_ids', value=ast.parse(expression, mode='eval').body))
                changes.append(fn.name + ': explicit source frame IDs at transport boundary')
        if fn.name == 'aggregate_nodes':
            # Remove only the post-generation forced MCQ/C fallback. Raw final
            # output is passed to the same official parser used by all baselines.
            found = False
            for block in ast.walk(fn):
                for field in ('body', 'orelse'):
                    statements = getattr(block, field, None)
                    if not isinstance(statements, list):
                        continue
                    for index, stmt in enumerate(statements):
                        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                                and isinstance(stmt.targets[0], ast.Name)
                                and stmt.targets[0].id == 'pred_answer'):
                            statements[index:] = [ast.Return(value=ast.Name(id='output_text', ctx=ast.Load()))]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                raise ValueError('upstream final parser changed')
            changes.append('aggregate_nodes: raw final response replaces MCQ-only fallback parser')
    if len(changes) != 3:
        raise ValueError('unexpected transport boundaries')
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['Vgent'], namespace, video_utils, dict(
        source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ast_modified=True, changes=changes,
        official_commit='c35c4b7731f22b962de24f31f842f6d84e183f8f')


class SpatialVgent:
    def __init__(self, runtime, repo, output):
        cls, self.namespace, self.video_utils, self.source_receipt = load_port_class(repo)
        self.runtime = runtime
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.agent = cls.__new__(cls)
        self.agent.args = types.SimpleNamespace(model_name='qwenvl_spatial135k',
            task='vsibench', chunk_size=64, n_retrieval=20, n_refine=5,
            uniform_frame=450, fps=1.0, total_pixels=16384)
        self.agent.mllm_response = self.response
        self.agent.video_llm = self.agent.processor = self.agent.image_processor = None
        # Official graph-free paths never call BGE. An unexpected graph path
        # must fail rather than silently swap or omit the required embedding.
        self.agent.embedding_model = self.agent.embedding_tokenizer = None
        self.call_count = 0

    def set_video(self, frames, source_ids, *, source_path, source_fps, watermark):
        import torch
        if len(frames) >= self.agent.args.chunk_size*self.agent.args.n_retrieval:
            raise ValueError('graph-active path requires full BGE/graph integration')
        self.source_ids = list(source_ids)
        self.agent.source_frame_ids = list(source_ids)
        self.source_path, self.source_fps, self.watermark = str(source_path), source_fps, watermark
        raw = torch.from_numpy(frames).permute(0, 3, 1, 2)
        resized, self.sample_fps = self.video_utils.resize_video(raw, 1.0,
            total_pixels=16384*max(1, (len(frames)+63)//64)*28*28)
        self.raw, self.processed = [raw], [resized]
        self.pngs = []
        from PIL import Image
        for frame, source_id in zip(frames, source_ids):
            path = self.output / ('source_%08d.png' % source_id)
            Image.fromarray(frame).save(path)
            self.pngs.append(str(path.resolve()))

    def response(self, unused_model, unused_tokenizer, unused_processor, text,
                 image_inputs, video, max_new_tokens=512, size_list=None, fps=None,
                 *, source_ids=None):
        if image_inputs is not None:
            raise ValueError('unexpected upstream image branch')
        content, provenance = [], []
        if video is not None:
            if source_ids is None or len(source_ids) != len(video):
                raise ValueError('video call has no exact source mapping')
            positions = {source_id: index for index, source_id in enumerate(self.source_ids)}
            paths = [self.pngs[positions[i]] for i in source_ids]
            # Upstream resizing selects its pixel budget; the frozen Spatial
            # processor owns tensor preparation and must receive both streams.
            content.append(dict(type='video', video=paths,
                                resized_height=int(video.shape[-2]),
                                resized_width=int(video.shape[-1])))
            provenance.append(dict(kind='source_video', frame_ids=list(source_ids),
                                   source_video=self.source_path))
        content.append(dict(type='text', text=text))
        messages = [dict(role='user', content=content)]
        batch, receipt = self.runtime.prepare(messages, provenance, watermark=self.watermark)
        batch['second_per_grid_ts'] = [2.0/(fps or self.sample_fps)] if video is not None else None
        receipt['actual_second_per_grid_ts'] = batch['second_per_grid_ts']
        self.call_count += 1
        out = self.output / ('call_%06d' % self.call_count)
        out.mkdir(exist_ok=False)
        import torch
        torch.save(batch, out / 'actual_inputs.pt')
        (out / 'messages.json').write_text(json.dumps(messages, indent=2))
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        result = self.runtime.generate(batch, generation_kwargs=dict(
            max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0))
        (out / 'output.json').write_text(json.dumps(result, indent=2))
        return result['raw_output']

    def answer(self, item):
        agent = self.agent
        candidates = item.get('options') or []
        # Only answer-format instructions change for the new dataset. Native
        # keyword extraction, JSON attempts, retrieval, refinement and evidence
        # aggregation retain their original instructions and control flow.
        self.namespace['PRED_PROMPT'] = ('Respond with only the letter of the correct option.\n'
            if candidates else 'Respond with only the requested numerical value.\n')
        before = self.call_count
        keywords, info = agent.extract_keywords(item['question'], candidates, self.processed)
        retrieved = agent.retrieve_nodes(item['question'], keywords, self.processed, candidates,
                                        None, None, None, info)
        refined, sql_check, check_result = agent.refine_nodes(retrieved, item['question'], info,
            candidates, self.processed, None)
        query = dict(question=item['question'], candidates=candidates,
                     letters=[chr(65+i) for i in range(len(candidates))])
        raw_output = agent.aggregate_nodes(refined, info, self.processed, self.raw, None, None,
            item['formatted_question'], query, None, sql_check, check_result, self.sample_fps)
        return dict(raw_output=raw_output, model_calls=self.call_count-before,
                    graph_active=False, source_receipt=self.source_receipt)
