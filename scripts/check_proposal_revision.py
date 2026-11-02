"""CPU-only restoration check using an existing spatial snapshot and images.

No weights, source-video decoding, discovery, SAM, VGGT, generation or scoring.
The optional processor check loads only the pinned processor/config metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox_vlm.alignment.association import MatchSettings
from sandbox_vlm.config import load_config
from sandbox_vlm.memory import Sandbox
from sandbox_vlm.memory.backend import BoundedSandboxBackend
from sandbox_vlm.qa.artifact import save_joint_input, state_sha256
from sandbox_vlm.qa.budget import QABudget, check_budget
from sandbox_vlm.qa.frames import PreparedFrameReader
from sandbox_vlm.qa.pipeline import run_question


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class ProcessorOnly:
    """An explicit non-generation transport, never reported as an MLLM result."""
    def __init__(self, config, measure):
        self.processor = None
        if measure:
            from transformers import AutoConfig, AutoProcessor
            settings = config['qa']['mllm']
            kwargs = dict(revision=settings['revision'], local_files_only=True)
            self.processor = AutoProcessor.from_pretrained(settings['model_id'], **kwargs)
            self.model_config = AutoConfig.from_pretrained(settings['model_id'], **kwargs)

    def answer(self, messages, images, limits):
        result = dict(status='cpu_input_verified_no_generation', model_calls=0,
                      model_weights_loaded=False)
        if self.processor is None:
            return result
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt')
        grid = inputs['image_grid_thw'].tolist()
        visual = int((inputs['input_ids'] == self.model_config.image_token_id).sum().item())
        expected = sum(t*h*w // self.processor.image_processor.merge_size**2 for t,h,w in grid)
        if len(grid) != len(images) or visual != expected:
            raise ValueError('processor image count/token accounting differs')
        patch = self.processor.image_processor.patch_size
        result.update(budget=check_budget(limits,
            image_sizes_hw=[[i.height, i.width] for i in images],
            processed_image_sizes_hw=[[h*patch, w*patch] for _,h,w in grid],
            input_tokens=int(inputs['input_ids'].shape[-1]), visual_tokens=visual,
            context_limit=int(self.model_config.text_config.max_position_embeddings)),
            image_grid_thw=grid, processor_tensor_sha256={
                name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in inputs.items()})
        if result['budget']['status'] != 'ok':
            result['status'] = 'budget_exceeded'
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/default.yaml')
    parser.add_argument('--d065-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure-processor', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    if cfg['qa']['mllm']['model_id'] != 'Qwen/Qwen3-VL-2B-Instruct':
        raise ValueError('only the approved frozen 2B processor is allowed')
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [args.sandbox, args.questions, args.sample/'manifest.json', args.config,
                 args.d065_root/'manifest.json']
    inputs = {str(p.resolve()): sha(p) for p in protected}
    report = dict(purpose='proposal_restoration_CPU_input_contract_only', status='running',
        model_calls=0, model_weights_loaded=False, source_video_decodes=0,
        benchmark_scoring_started=False, new_QA_predictions=0, questions=[],
        warning='Historical short diagnostic geometry is reused, not a new long-video result.')
    write(args.output/'report.json', report)
    try:
        base = Sandbox.from_dict(read(args.sandbox))
        before = state_sha256(base)
        manifest = read(args.sample/'manifest.json')
        reader = PreparedFrameReader(args.sample, manifest, video_id=base.video_id)
        questions = read(args.questions)
        if (not questions or any(set(q) != {'id', 'question'} for q in questions)
                or len({q['id'] for q in questions}) != len(questions)):
            raise ValueError('existing diagnostic IDs and original questions only')
        limits = QABudget(**cfg['qa']['budget'])
        backend = BoundedSandboxBackend(cfg['memory']['backend_max_bytes'], device='cpu')
        transport = ProcessorOnly(cfg, args.measure_processor)
        for index, question in enumerate(questions):
            folder = args.output/f'question_{index:03d}'
            result, working = run_question(base, question['question'], reader, limits, transport,
                MatchSettings(**cfg['alignment']['match']), backend=backend,
                evidence_sink=lambda images, metadata, messages: save_joint_input(folder, images, metadata, messages))
            for record in result['evidence']['image_records'][1:]:
                source = args.sample/reader.frames[record['frame_id']]['path']
                inputs[str(source.resolve())] = sha(source)
            if state_sha256(base) != before or state_sha256(working) != before:
                raise ValueError('CPU input check changed the base sandbox')
            write(folder/'input_check.json', result)
            report['questions'].append(dict(id=question['id'], status=result['status'],
                input_contract=result['evidence']['input_contract'], processor=result['generation']))
        # D065 remains a different representation. Do not fabricate missing 3D,
        # silently discard its observations, or promote D081's successful subset.
        frozen = read(args.d065_root/'manifest.json')
        report['approved_offline_state_readiness'] = []
        for index in (346, 372, 424, 459):
            row = frozen['records'][index]
            query = row['frozen_queries'][-1]['query_frame_id']
            path = args.d065_root/f'sample_{index:02d}'/f'query_{query:06d}'/'visual_sandbox.json'
            inputs[str(path.resolve())] = sha(path)
            state = read(path)
            try:
                Sandbox.from_dict(state)
            except ValueError as exc:
                report['approved_offline_state_readiness'].append(dict(index=index,
                    status='needs_valid_spatial_memory', representation=state.get('representation'),
                    visual_hypotheses=len(state.get('instances', [])), reason=str(exc)))
            else:
                raise ValueError('D065 visual state was silently accepted as a spatial sandbox')
        report.update(status='ok' if all(q['status']=='cpu_input_verified_no_generation' for q in report['questions']) else 'incomplete',
            base_state_unchanged=state_sha256(base)==before,
            long_video_QA_ready=False, VSC_improvement_established=False,
            base_counts=base.capacity_report(), backend=backend.last_report,
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
                *sorted((ROOT/'sandbox_vlm').rglob('*.py')), Path(__file__)]})
        if any(sha(p) != digest for p,digest in inputs.items()):
            raise ValueError('existing source artifact changed during CPU check')
        write(args.output/'input_sha256.json', inputs)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report.update(CPU_seconds=time.perf_counter()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(args.output/'report.json', report)
    print(json.dumps({key: report[key] for key in ('status','model_calls','CPU_seconds','long_video_QA_ready')}))


if __name__ == '__main__':
    main()
