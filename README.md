> 2026-11-09：重复发现故障已修复，613帧完整离线开发验收通过；可恢复小规模离线测试。正式全任务/流式评测仍有独立门槛。见 [修复与真实验收](docs/CONTINUOUS_DISCOVERY_REPAIR_20261109.md)。

> **2026-11-09 current model correction:** The continuous candidate is now in `sandbox_vlm.continuous_model`, with executable entry `scripts/run_continuous_model.py` and API `create_continuous_model`. It processes all sampled frames with StreamVGGT FIFO, maintains Scenes, renders actual 3D evidence, and selects QA frames by temporal initialization, instance coverage and visual diversity. See [implementation and usage](docs/CONTINUOUS_MODEL_20261109.md). The D116 aliases described below remain the protected historical reference; old scores do not belong to this new candidate.

# SandMem

Training-free long-video spatial question answering with **observation-linked instance memory and a rendered 3D sandbox**. The project name is provisional; the stable Python namespace is `sandbox_vlm`.

This repository contains the frozen **D116 method plus the approved memory-capacity policy**. The inference/selection/rendering core is copied byte-for-byte from the accepted snapshot, rather than reconstructed from an earlier proposal. See [the method](docs/METHOD.md) and [the verification report](verification/README.md).

## Fixed method

- Instance records retain names, descriptions, appearance and actual source-frame/region links.
- Question-linked lexical retrieval selects complete relevant records and actual same-frame context.
- Frozen SAM and VGGT supply local observation and geometry evidence. The rendered **3D sandbox image, complete related records, real frames and original question enter the MLLM together**. Independent local references are not treated as a global map.
- Sparse inputs with actual observation links use an independent original-frame draft followed by joint evidence verification. Other inputs use a joint answer directly. Routing does not read task labels or answers.
- On capacity overflow, keep each instance's earliest and latest actual observations; if necessary, evict instances in order of oldest last observation. Surviving IDs remain stable, deleted IDs are not reused, and archived records cannot silently return.

## Environment and CPU verification

The tested Qwen reference environment is Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, transformers 4.57.6 and NumPy 2.2.6. Model weights, datasets and environments are not stored in Git.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# Install the preserved dependency source without changing the tested pins.
SAM2_BUILD_CUDA=0 .venv/bin/pip install --no-deps --no-build-isolation -e vendor/sam2 -e vendor/vggt
.venv/bin/python tools/verify_release.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests
```

The CPU tests do not load model weights or generate benchmark answers. CUDA inference additionally needs the pinned checkpoints and the appropriate SAM build. Third-party source and license notices are retained under `vendor/`; the Spatial files there are interface-test fixtures only, not a full Spatial installation.

## Method API

```python
from sandbox_vlm.api import create_memory, BoundedVisualBackend
from sandbox_vlm.api import compose, draw, joint_messages
from sandbox_vlm.api import eligible, verification_messages, decide

memory = create_memory(video_id, original_question, limit_bytes=64 * 1024 * 1024)
# memory.ingest(...) enforces first/last compaction and eviction after each write.
backend = BoundedVisualBackend(64 * 1024 * 1024, device="cuda")
# Include all persistent auxiliary state when committing.
# backend.commit(memory, auxiliary_state=auxiliary)
```

`compose`, `draw`, `joint_messages` and the two-pass helpers are direct aliases of the frozen implementation. Real frames, masks, geometry and complete auxiliary state must be supplied by their perception owners. This API is not a new solver or a replacement perception pipeline.

## Historical reproduction and current scope

The scripts in `scripts/` preserve the historical frozen-manifest workflow. They require separately supplied perception artifacts, official data and pinned checkpoints; a fresh clone does not contain those artifacts. The old generic `mainline.py`/default configuration is historical compatibility code, not the final D116 entry point.

For an existing reference artifact workspace, replay all 294 development inputs on CPU with the new checkout's code:

```bash
python tools/replay_d116_inputs.py \
  --reference-workspace /path/to/reference-workspace \
  --output work/verification/d116_inputs.json
```

The recorded **57.3333% Recall / 25.55% offline Count / 26.945% streaming Count** belong to the earlier Qwen D116/D118 full evaluation, before the capacity-policy extension. They are not a new capacity experiment or Spatial-MLLM result. Spatial-MLLM migration and its checkpoint/budget contract remain a separate task; isolated adapter source is not evidence that the full migration is complete.

## Renaming

Change `display_name` in `project.json` and the presentation title when the final name is chosen. Repository/folder names may change independently; no algorithm import or internal package name depends on `SandMem`.
