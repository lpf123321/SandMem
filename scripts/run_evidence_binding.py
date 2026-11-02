"""D108: fixed evidence-factor diagnosis, followed by one observation-binding repair."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIMIT = 64 * 2**20
SOURCE = ROOT / 'work/bench/d105_complete_sandbox_20261031_03'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def freeze(out, manifest):
    paths = [*sorted((ROOT / 'sandbox_vlm').rglob('*.py')), Path(__file__),
             ROOT / 'scripts/run_evidence_binding.sbatch']
    paths += [ROOT / rel for rel in manifest.get('additional_sources', [])]
    manifest['source_sha256'] = {}
    for p in paths:
        rel = p.relative_to(ROOT)
        dest = out / 'runtime' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        manifest['source_sha256'][str(rel)] = sha(p)
    (out / 'runtime/venv').symlink_to(ROOT / 'venv', target_is_directory=True)
    write(out / 'manifest.json', manifest)


def prepare(out, mode):
    from transformers import AutoProcessor
    from sandbox_vlm.qa.evidence_binding import diagnostic_messages, bound_messages, linked_selection
    from sandbox_vlm.qa.complete_sandbox_repair import render_repaired
    from sandbox_vlm.qa.continuous_sandbox import open_rgb_images
    from sandbox_vlm.qa.map_only import indexed_messages, text_tokens
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.qa.budget import check_budget, QABudget
    from sandbox_vlm.memory.storage import cpu_storage_report
    from sandbox_vlm.data.vsi_super import load_super_inputs
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    old = read(SOURCE / 'manifest.json')
    processor = AutoProcessor.from_pretrained(old['config']['qa']['mllm']['model_id'],
        revision=old['config']['qa']['mllm']['revision'], local_files_only=True)
    m = dict(method='D108_' + mode, source=str(SOURCE), config=old['config'],
        records=deepcopy(old['records']), jobs=[], inherited_sha256={}, input_sha256={},
        D073_exclusions_sha256=old['D073_exclusions_sha256'], new_perception_calls=0,
        maximum_runtime_seconds=3300, maximum_QA_attempts=117 if mode == 'diagnostic' else 294,
        structure_limit_bytes=LIMIT, labels_used_for_preparation=False, all_previously_exposed=True)
    peak = 0
    inputs = {}
    for rec in m['records']:
        if mode == 'diagnostic' and rec['phase'] != 'diagnosis':
            continue
        queries = [rec['queries'][-1]] if mode == 'diagnostic' else rec['queries']
        if mode == 'repair':
            key = rec['variant'], rec['duration']
            if key not in inputs:
                inputs[key] = load_super_inputs(ROOT / 'work/protocol/vsi_super' / key[0] /
                    f'test_{key[1]}.parquet', variant=key[0])
            example = inputs[key][int(rec['sample_id'].split(':')[1])]
            assert example.sample_id == rec['sample_id']
        for query in queries:
            src = SOURCE / query['folder']
            snap = read(src / 'input.json')
            assert snap['direct_answer'] is True
            selection = linked_selection(example.original_question, snap['payload']) if mode == 'repair' else None
            changed = mode == 'diagnostic' or set(selection['relevant_ids']) != {r[0] for r in snap['payload']['records']}
            arms = ['frames_only', 'catalog_only', 'diagram_only'] if mode == 'diagnostic' else ['repair']
            for arm in arms:
                folder = out / arm / query['folder']
                job = dict(record=rec['record_index'], sample_id=rec['sample_id'], phase=rec['phase'],
                    task=rec['task'], duration=rec['duration'], variant=rec['variant'],
                    query_frame_id=query['query_frame_id'], arm=arm, source_folder=str(src),
                    folder=str(folder.relative_to(out)), new_QA_required=changed)
                m['jobs'].append(job)
                if not changed:
                    continue
                folder.mkdir(parents=True)
                diagram = None
                if mode == 'repair':
                    geometry_path = ROOT / 'work/bench/d107_d105_geometry_repair_20261031_01' / query['folder'] / 'geometry.json'
                    geometry = read(geometry_path)
                    if 'original_D105' not in geometry:
                        geometry = dict(original_D105=geometry, added_panels=[])
                    relevant = set(selection['relevant_ids'])
                    for extent in geometry['original_D105']['sandbox']['spatial_extents']:
                        extent['instances'] = [r for r in extent['instances'] if r['global_instance_id'] in relevant]
                    for panel in geometry['added_panels']:
                        panel['observations'] = [r for r in panel['observations'] if r['instance_id'] in relevant]
                    geometry['added_panels'] = [p for p in geometry['added_panels'] if p['observations']]
                    plotted = {r['global_instance_id'] for e in geometry['original_D105']['sandbox']['spatial_extents']
                               for r in e['instances']}
                    plotted.update(r['instance_id'] for e in geometry['added_panels'] for r in e['observations'])
                    snap['payload']['plotted_ids'] = sorted(plotted)
                    snap['payload']['unplotted_ids'] = sorted({r[0] for r in snap['payload']['records']} - plotted)
                    snap['payload']['box_observation_sources']['rows'] = [r for r in
                        snap['payload']['box_observation_sources']['rows'] if r[0] in plotted]
                    added_sources = [[r['instance_id'], r['frame_id'], r['observation_id']]
                                     for p in geometry['added_panels'] for r in p['observations']]
                    seen_sources = {r[0] for r in snap['payload']['box_observation_sources']['rows']}
                    snap['payload']['box_observation_sources']['rows'] += [r for r in added_sources if r[0] not in seen_sources]
                    snap['payload']['local_references'] = [e['coordinate_reference'] for e in
                        geometry['original_D105']['sandbox']['spatial_extents']] + [e['coordinate_reference'] for e in geometry['added_panels']]
                    diagram = render_repaired(geometry)
                    diagram.save(folder / 'sandbox.png')
                    snap['limits']['max_image_side'] = 1280
                    write(folder / 'selection.json', selection)
                    write(folder / 'geometry.json', geometry)
                    m['inherited_sha256'][str(geometry_path)] = sha(geometry_path)
                elif arm == 'diagram_only':
                    diagram = open_rgb_images([snap['map_path']])[0]
                images = [object() for _ in snap['frame_ids']]
                if mode == 'repair':
                    messages, joint = bound_messages(snap, images, diagram, selection)
                else:
                    messages, joint = diagnostic_messages(snap, images, diagram, arm)
                has_map = len(joint) == len(images) + 1
                paths = ([str(folder / 'sandbox.png') if mode == 'repair' else snap['map_path']] if has_map else []) + snap['frame_paths']
                hashes = ([pixel_sha256(diagram)] if has_map else []) + snap['image_pixel_sha256'][1:]
                grids = deepcopy(snap['original_image_grid_thw'])
                sizes = deepcopy(snap['original_budget']['input_image_sizes_hw'])
                if has_map:
                    grid = processor.image_processor(images=[diagram], return_tensors='pt')['image_grid_thw'].tolist()[0]
                    grids.insert(0, grid)
                    sizes.insert(0, [diagram.height, diagram.width])
                visual = sum(g[0]*g[1]*g[2] // processor.image_processor.merge_size**2 for g in grids)
                tokens = text_tokens(processor.tokenizer, messages, len(joint), processor.chat_template)
                patch = processor.image_processor.patch_size
                budget = check_budget(QABudget(**snap['limits']), image_sizes_hw=sizes,
                    processed_image_sizes_hw=[[g[1]*patch, g[2]*patch] for g in grids], input_tokens=tokens+visual,
                    visual_tokens=visual, context_limit=snap['original_budget']['model_context_limit'])
                if budget['status'] != 'ok':
                    raise ValueError(budget)
                serialized = indexed_messages(messages, joint, hashes)
                prepared = dict(paths=paths, pixel_sha256=hashes, limits=snap['limits'],
                    budget=budget, grids=grids, generation_kwargs=snap['generation_kwargs'], has_map=has_map)
                write(folder / 'input.json', prepared)
                write(folder / 'messages.json', serialized)
                for p in [src / 'input.json', src / 'messages.json', *map(Path, paths)]:
                    if not p.is_relative_to(out):
                        m['inherited_sha256'][str(p)] = sha(p)
                for name in ['input.json', 'messages.json', 'sandbox.png', 'geometry.json', 'selection.json']:
                    p = folder / name
                    if p.exists():
                        m['input_sha256'][str(p)] = sha(p)
                size = cpu_storage_report([m, old, inputs, snap, messages, serialized, prepared])['cpu_structural_bytes']
                size += sum(h*w*3 for h,w in sizes) + 4*len(json.dumps(serialized).encode())
                peak = max(peak, size)
                if size > LIMIT:
                    raise MemoryError('64MiB preparation bound')
                if diagram is not None:
                    diagram.close()
    assert len(m['jobs']) == (117 if mode == 'diagnostic' else 294)
    m['changed_queries'] = sum(j['new_QA_required'] for j in m['jobs'])
    m['preparation_seconds'] = time.perf_counter() - start
    m['preparation_structure_upper_bytes'] = peak
    m['preparation_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    freeze(out, m)
    print(json.dumps({k:m[k] for k in ['method','changed_queries','preparation_seconds','preparation_structure_upper_bytes']}))


def execute(out):
    import torch
    from sandbox_vlm.qa.model import QwenAnswerer
    from sandbox_vlm.qa.budget import QABudget
    from sandbox_vlm.qa.continuous_sandbox import open_rgb_images
    from sandbox_vlm.qa.artifact import pixel_sha256
    from sandbox_vlm.memory.storage import cpu_storage_report
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required')
    if (out / 'run_report.json').exists():
        raise FileExistsError('preserve first execution')
    start = time.perf_counter()
    m = read(out / 'manifest.json')
    if m.get('confirmation_required'):
        authorization=read(out/'authorization.json')
        if (authorization.get('QA_authorized')is not True
                or authorization.get('sealed_manifest_sha256')!=sha(out/'manifest.json')
                or authorization.get('maximum_QA_attempts')!=m['maximum_QA_attempts']):
            raise RuntimeError('confirmed scope must match sealed input before QA')
    for group in ['source_sha256', 'inherited_sha256', 'input_sha256']:
        for p, digest in m[group].items():
            if sha(ROOT / p if group == 'source_sha256' else p) != digest:
                raise ValueError('frozen input changed: ' + p)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    reuse_source=Path(m.get('reuse_prediction_source',m['source']))
    original = {(r['record'], r['query_frame_id']):r for r in read(reuse_source / 'run_report.json')['queries']}
    report = dict(status='running', queries=[], QA_attempts=0, new_perception_calls=0, retries=0,
                  slurm_job_id=os.environ['SLURM_JOB_ID'], maximum_structure_upper_bytes=0)
    report['exact_prediction_reuses']=0
    report['reuse_breakdown']={}
    report['prediction_reuse_source']=str(reuse_source)
    device=torch.cuda.get_device_properties(0)
    report['hardware']=dict(device_name=device.name,total_memory_bytes=device.total_memory,
        visible_CUDA_devices=torch.cuda.device_count(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    model = QwenAnswerer(m['config']['qa']['mllm'])
    try:
        for job in m['jobs']:
            if time.perf_counter() - start > 3300:
                raise TimeoutError('fixed runtime limit')
            row = deepcopy(job)
            if not job['new_QA_required']:
                row.update(original[(job['record'], job['query_frame_id'])], reused_D105_prediction=True)
                row['new_QA_required']=False
                row['arm']=job['arm']
                kind=job.get('prediction_reuse_kind','identical_input_and_generation')
                row['prediction_reuse_kind']=kind
                report['reuse_breakdown'][kind]=report['reuse_breakdown'].get(kind,0)+1
                report['exact_prediction_reuses']+=1
            else:
                folder = out / job['folder']
                snap = read(folder / 'input.json')
                serialized = read(folder / 'messages.json')
                images = open_rgb_images(snap['paths'])
                try:
                    hashes = [pixel_sha256(i) for i in images]
                    if hashes != snap['pixel_sha256']:
                        raise ValueError('actual source pixels differ')
                    messages = deepcopy(serialized)
                    for message in messages:
                        for item in message['content']:
                            if item['type'] == 'image':
                                index = item.pop('evidence_image_index')
                                if item.pop('pixel_sha256') != hashes[index]:
                                    raise ValueError('message image hash mismatch')
                                item['image'] = images[index]
                    size = cpu_storage_report([m, report, original, snap, serialized, messages])['cpu_structural_bytes']
                    size += sum(i.height*i.width*3 for i in images) + 4*(folder/'messages.json').stat().st_size
                    size += m.get('note_generation_reserve_bytes', 0)
                    report['maximum_structure_upper_bytes'] = max(report['maximum_structure_upper_bytes'], size)
                    if size > LIMIT:
                        raise MemoryError('64MiB QA bound')
                    if model.generation_attempts >= m['maximum_QA_attempts']:
                        raise RuntimeError('fixed QA call cap')
                    generation = model.answer(messages, images, QABudget(**snap['limits']),
                                              note_boundary=snap.get('note_boundary'))
                    write(folder / 'generation.json', generation)
                    if generation['budget'] != snap['budget'] or generation['image_grid_thw'] != snap['grids']:
                        raise ValueError('actual processor differs from sealed input')
                    if generation['generation_kwargs'] != snap['generation_kwargs']:
                        raise ValueError('D105 generation settings changed')
                    prediction = None
                    status = generation['status']
                    if status == 'generated':
                        answer = generation['raw_output'].strip()
                        try:
                            if snap.get('response_style') == 'D065_strict':
                                from sandbox_vlm.qa.d065_prompt_alignment import extract_d065_answer
                                answer = extract_d065_answer(answer, sparse=snap['sparse'])
                            prediction = float(answer) if job['variant'] == 'count' else answer
                            status = 'ok' if answer else 'empty_answer'
                        except ValueError as exc:
                            status = 'answer_failed' if snap.get('response_style') == 'D065_strict' else 'invalid_numeric_answer'
                            row['answer_error'] = str(exc)
                    row.update(status=status, prediction=prediction, reused_D105_prediction=False)
                finally:
                    for image in images:
                        image.close()
            report['queries'].append(row)
            report['QA_attempts'] = model.generation_attempts
            write(out / 'run_report.json', report)
            print(row['arm'], row['record'], row['status'], flush=True)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    report.update(wall_seconds=time.perf_counter()-start, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out / 'run_report.json', report)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', required=True, choices=['prepare-diagnostic','prepare-repair','execute'])
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.action.startswith('prepare-'):
        prepare(args.output.resolve(), args.action.removeprefix('prepare-'))
    else:
        raise SystemExit(execute(args.output.resolve()))
