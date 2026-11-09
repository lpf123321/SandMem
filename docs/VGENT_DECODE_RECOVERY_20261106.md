# Vgent exact-input decode recovery — 2026-11-06

Job 17362 exited 1:0 after 04:29:26. The formal evaluator raised Decord
`Unable to handle EOF` while fetching the 1 FPS RGB batch for
`arkitscenes/42446056.mp4`. This call was before per-question exception handling,
so it stopped the full job. It was not a reported CUDA OOM. The exact internal
Decord indexed-seek failure has not been isolated; the media SHA256 is unchanged
from successful HERMES, CausalMem and SimpleStream reads:
`1e51627a880604d3c8c93df11cc6ad2207803182f4139297ee5b57d4106066c0`.

## CPU-only acceptance

- Smaller 16-frame indexed batches reproduced the EOF failure; that failed
  probe and original traceback remain under `recovery_20261106_01/`.
- Sequential Decord decoding recovered all 373 frozen source IDs, ending at
  source index 11160 (source declares 11165 frames, 30 FPS).
- Last four recovered RGB frames are byte-identical to original SimpleStream
  lossless PNGs. A healthy control video's 169 RGB observations are all
  byte-identical to original Vgent lossless PNGs.
- All 24 original native-processor input blocks of the failed video are
  byte-identical to pre-existing HERMES pixel tensor hashes, including their
  source IDs and temporal/spatial grids. No weights, model forward or answers
  were involved in this check.
- The actual recovery helper passed both direct known-file sequential reading
  and an injected EOF fallback check; healthy files keep original get_batch.
  No missing frame is padded, replaced, dropped, or silently repeated by the
  recovery helper. Native processor temporal padding remains unchanged.

## Isolated continuation

`work/baselines_candidates_20261105/vgent_formal_resume_02/` is a new frozen run.
Original `vgent_formal_01/`, its empty failed-video directory, all 237 outputs,
all model calls, failed probes, frozen source and failure records are preserved.
All 26 completed video checkpoints (237 unique questions) are referenced through
links with their predictions pinned by SHA256. They are reused irrespective of
answer correctness; none are regenerated. The failed video had zero model calls.
Remaining new questions: 4893, original full combined denominator: 5130.

`config.json` and `inputs.jsonl` are byte-identical to the original formal run.
The complete Spatial transport, prompts, native JSON attempts, graph threshold,
retrieval/refinement/aggregation, source FPS/indices, full frame count, native
visual/text/output budgets and official scorer are unchanged. Only CPU decode
handling, job-boundary timing and completion bookkeeping differ in the driver.
This is not a new method or a score-guided candidate. No Proposal, AGENTS.md,
PLAN.md or SandMem method files were modified.

Job **17414**, one A100, was confirmed RUNNING on gpu1. It is scheduled through
Slurm directly, independent of this chat's background processes. Each allocation
is capped at 12 hours. The runner checkpoints after 10 hours at a video boundary
(leaving two hours for a started video's processing and IO); this is within the
original 42000-second upper bound. A successful checkpoint with progress requests
requeue of the same job ID. Each restart verifies the frozen manifest and skips
all completed video checkpoints. Interruptions, new decoding failures, GPU
capacity failures and incomplete video folders stop for review rather than
repeating failed model calls. The cluster has no RequeueExit rule, so an explicit
fatal process exit does not trigger an error-retry loop.

## Reproduction and cost accounting

Launcher: `vgent_formal_resume_02/launcher.sbatch`.
Immutable inputs/dependencies/reuse receipts: `manifest.json`, `recovery.json`,
`reused_predictions.json`, `pixel_equivalence.json`.
Current state: `controller_status.json`; per-allocation receipts and costs:
`attempts/job_<id>_restart_<n>/` plus uniquely named completion files. Requeue
intent/response is persisted before the next allocation. Slurm output uses
append mode to preserve prior logs.

Cost is additive: original wiring 19 model calls; failed formal job 2310 model
calls; each resumed allocation records its new model calls, time and peak GPU
memory. CPU decode/probe/processor verification is additional CPU-only work
(0 learned model forwards), not free model perception. No Qwen-produced memory
or semantic cache enters this continuation. Newly decoded source RGB is ordinary
model-independent preprocessing. Final combined results go to this new run's
`scores.json` and `reports/vgent_recovered_results.json`; the old incomplete run
must never be relabeled as a completed full evaluation.

## Second failure and correction: consume the validated data, not the decoder

Job 17414 FAILED 1:0 after 00:09:28. It made **zero new MLLM calls** and stopped
again on the same video during `fresh.next()` inside the model process. All
237 previous predictions remain unchanged. CPU standalone sequential decoding
and its pixel-equivalence checks were successful, but did not establish decoder
stability after model loading in the actual GPU job. The exact decoder/process
interaction is not diagnosed. No evidence supports blaming CUDA memory or a
changed media file.

The previous repair was incomplete: it re-ran sequential decoding instead of
consuming the actual already-validated CPU RGB array. This avoidable deployment
gap is corrected in a new independent directory `vgent_formal_resume_03/`; the
failed original and resume_02 runs and their manifests remain unchanged.

- The known 373-frame NPY data is reused directly, pinned by both file SHA256
  and actual RGB SHA256, source video SHA256, exact IDs, FPS and dimensions.
  This is the same data already accepted by all 24 native processor input hashes.
- The model process reads cached RGB only. It does not invoke Decord. A cache
  miss for another source video invokes a fresh CPU-only interpreter with
  CUDA_VISIBLE_DEVICES empty; the worker rejects unexpected Torch imports and
  validates all original source IDs. CPU decode/IO time is recorded and counts
  toward the allocated GPU wall time when performed during an allocation.
- Cache integrity/provenance failure or an isolated decoder failure stops for
  review. No missing frame is replaced, dropped, repeated or inferred. CPU RGB
  caching contains no model-produced observations, memory, embeddings or labels.
- End-to-end cache IO tests passed: all 373 known frames have the accepted RGB
  checksum; all 169 control-video frames from the isolated worker match the
  original lossless Vgent PNGs. The parent test process imported neither Decord
  nor Torch; the worker imported no Torch and had no visible CUDA device.
- The original config, questions, Vgent/Spatial transport, prompts, sample IDs,
  native budgets, scorer, original 237 predictions and checkpoint-requeue policy
  are unchanged. Existing artifact protection and the EGAgent pause still apply.

Job **17432** was confirmed RUNNING on A100 gpu1. At submission no additional
Vgent answer or completion is claimed. Actual progress must be checked in
resume_03/videos/arkitscenes_42446056/decode_receipt.json and query output files,
then in its controller/completion files. A RUNNING scheduler state alone is not
proof that source ingestion or inference passed. The extra 17414 allocation
(9m28, zero new model calls) is retained in cost accounting.

### Actual GPU continuation acceptance

Job 17432 has now passed the prior failure point in the actual model process. Its decode receipt confirms direct loading of the accepted 373-frame RGB checksum and exact source IDs, with no model-process video decoding. 5 new per-question outputs have been saved for this formerly failing video. Every checked completed question includes at least one recorded actual Spatial encoder invocation. This is evidence of actual new inference, not merely scheduler RUNNING. Original 237 predictions remain preserved. Full evaluation is still in progress; no full Vgent score is claimed. Snapshot: reports/vgent_cached_resume_actual_progress_17432.json.
