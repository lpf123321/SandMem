# Additional training-free baseline audit — 2026-11-05

This task concerns Vgent (2510.14032), EGAgent (2601.18157), zero-shot S-Agent
(2606.20515), SimpleStream (2604.02317), and R4DSG (2608.11017).
No SandMem method, Proposal, AGENTS.md, PLAN.md, or historical results changed.

## Source and release findings

| Method | Long-video / training status | Public implementation | Verified VSI-Bench score | Action |
|---|---|---|---|---|
| Vgent | Explicit training-free long-video graph RAG | [official](https://github.com/xiaoqian-shen/Vgent) | None found in paper or targeted third-party searches | Independent Spatial port required |
| EGAgent | Very-long-video inference with pretrained planners, graph generation and retrieval; no new task training in method | [official](https://github.com/facebookresearch/egagent) | None found | Independent Spatial port required |
| S-Agent zero-shot | Multi-view/video spatial tool reasoning; zero-shot variant distinct from trained S-Agent-8B; includes 3D tools | [landing repository](https://github.com/Ropedia/S-Agent) explicitly says inference code not released | None verified; paper 58.8 is ReVSI, not VSI-Bench | Exclude under user no-code rule |
| SimpleStream | Training-free streaming baseline, only recent N frames; no historical memory | [official](https://github.com/EvolvingLMMs-Lab/SimpleStream) | None found | First independent Spatial port submitted |
| R4DSG | Long egocentric RGB relative 3D/time memory; complete front-end training/inference provenance cannot be verified from released code | [project site](https://github.com/dualtransparency/R4DSG), [partial reference code](https://github.com/dualtransparency/R4DSG-code) | None found; reported gain is on 255 EgoLifeQA questions | Not eligible for faithful end-to-end reproduction with current release |

Search absence is not proof that no paper anywhere contains a score. Sources
and exact commits are archived under `work/baselines_candidates_20261105/`.
The separate R4DSG-code release explicitly disclaims reproducing original
experiments: cached observations only, fixed windows/BM25, absent SAM3/SAM3D
front-end, absent original episode/prompt/sampling/archival settings.
Do not invent a front-end and label it the official R4DSG baseline.

Paper sources: https://arxiv.org/html/2510.14032,
https://arxiv.org/html/2601.18157,
https://arxiv.org/html/2606.20515,
https://arxiv.org/html/2604.02317,
https://arxiv.org/html/2608.11017.

## Shared measurement definition

- Frozen Spatial-MLLM 135K, revision
  `635b63733ed24f76b2f1a533dd5bc32836ea6c53`, BF16, no training/quantization.
- Reuse already verified original full VSI-Bench data: 5,130 questions / 288 videos,
  HF revision `3a6a1cf9ac54c403509b15260897cc6ddc41b408`.
- Data root `work/baselines_vsibench_20261105/data/`; no duplicate download.
- All methods retain their inference logic and necessary auxiliary models;
  native-backbone paper results are distinct from independently adapted results.
- Same original question/options, official task answer instructions and official
  VSI-Bench answer parsing/scoring as previous Spatial HERMES/CausalMem runs.
  Eight-task macro average; numerical MRA; no dropped failures/invalid answers.
- Report actual frame count, visual tokens, text tokens, output budget, all
  planner/graph/perception calls and auxiliary-model costs. No equal-budget claim.
- Current user has already seen the benchmark; input-fixed wiring is not blind
  validation and is not used for prompt/threshold selection.

## SimpleStream initial frozen port

Official recent-4 / 1 FPS setting. Use last four of the existing common 1 FPS
source pool, queried at source-video end; never replace this with uniform
whole-video coverage. Native Qwen upstream samples with a linspace grid; the
common time-arange pool is an explicit adaptation difference. Frames are
independent images as in the official Qwen2.5 implementation. Native Spatial
image preprocessing and its dual image encoder streams replace manual image-only
feature injection and the single manually constructed vision prefix.
Original dataset-specific OVO/StreamingBench instructions are replaced by the
same official VSI-Bench format used by the other local Spatial baselines.

Native original frames are 480x640 or 640x480 for all 288 videos. No extra pixel
cap, input truncation, quantization, hidden memory, retrieval, or auxiliary model.
Four actual source observations per question; each static image receives native
Spatial temporal padding, which is disclosed separately from observations.
No visual-feature cache: every question is processed by both Spatial streams.
Native visual inputs are saved once per video and hash-checked identical across
questions; exact text tensors, serialized messages, all tensor hashes, forward
traces, original lossless PNGs, source mappings and raw outputs are saved.
Greedy, repetition penalty 1.0, max 16 output tokens. No failed-input retries.

Two wiring questions chosen by minimum input-only SHA256 separately among
numerical/choice inputs, without answers: IDs 69, 1077. Job **17319**, one A100,
first performs those two calls and asserts source/image mapping, visual tensor
consistency and actual spatial encoder invocation. Only if both pass, automatically
freezes a separate full manifest and runs 5,130 questions with official scoring.
No pilot score is computed. GPU allocation capped at 12 hours.
This is a submitted job, not a completed result. The run folders are
`simple_pilot_01/` and `simple_formal_01/` beneath the candidate root.

Entrypoints: `scripts/baselines_vsibench/simple_recent.sbatch`,
`simple_recent.py`, `freeze_simple_recent.py`. Frozen runtime copies, configs and
manifests live inside each run. Do not modify submitted runtime files.

## Vgent compatibility issues found before implementation

The official entrypoint bypasses graph construction whenever the 1 FPS video
contains fewer than `chunk_size * n_retrieval = 64 * 20 = 1280` frames.
Every VSI-Bench video has at most 434 such frames; **all 288 therefore take the
native graph-free path with these defaults**. Do not silently reduce the threshold
and claim it is the unchanged official configuration.

Its final parser assumes multiple-choice letters and replaces unmatched output
with the third option; VSI-Bench numerical answers require a separately documented
output-interface adaptation and the common official scorer. Native keyword and
JSON parsing steps can make up to five calls; failures and all attempts must be
retained and billed, never counted as free formatting work. Keep BGE if an active
path calls it; absence of actual graph retrieval must be reported rather than
advertising graph-memory evaluation.

## EGAgent compatibility issues found before implementation

Requires SigLIP2 giant patch16-384 for visual retrieval, graph extraction and
planning/analysis/grading/final-answer MLLM workers; all MLLM workers must use the
confirmed Spatial checkpoint. Original default captioners/planners include GPT4.1,
LLaVA-Video and Gemini, so a QA-only swap would be invalid. LangChain structured
output and SQL search are not supported by changing one model path. Preserve
planner/tool execution and maximum five plan steps, all actual calls and schemas.
VSI-Bench RGB input supplies no transcript, so absence must be explicit rather
than fabricated from annotation labels or a different source.

Auxiliary checkpoint metadata pinned for preparation:
BGE `BAAI/bge-large-en-v1.5` revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`;
SigLIP2 `google/siglip2-giant-opt-patch16-384` revision
`a713301b217d38485fb2204c808367d10bc3cc40`.
Download receipts record exact bytes and HF LFS SHA256; downloading is not proof
that model loading or integration has passed.

## Subsequent execution / user decision

- SimpleStream job 17319 wiring passed on A100-PCIE-40GB: two calls, four
  real images each, 1,564 visual tokens each; peak allocated 12,559,552,512
  bytes, reserved 12,991,856,640 bytes. Formal 5,130-input manifest was then
  sealed and formal inference started; no interim scores used.
- Vgent CPU checks passed: raw numerical answer survives, exact frame IDs reach
  transport, original five invalid-JSON attempts retained. Three mock-fixture
  setup failures are saved separately; none used a model. Two-input GPU
  acceptance frozen separately in `vgent_wire_frozen_02`.
- Vgent submission was rejected by QOSMaxSubmitJobPerUserLimit, zero GPU calls.
  A serial submitter waits for 17319 to leave the queue and submits only this
  one-hour A100 wiring check. It does not automatically authorize a modified
  method or start formal scoring without acceptance review.
- Both auxiliary model downloads completed with official HF LFS SHA256 checks
  (`reports/aux_download_receipts.json`). This is download provenance, not model
  integration. Original BGE is unreachable in Vgent's default graph-free path.
- User explicitly chose **pause EGAgent, complete the other baselines first**
  when asked about absent original raw-caption generation prompt/code. No new
  caption prompt was adopted, no EGAgent GPU inference started, and no score
  is available. The proposed 64-frame/512-output caption protocol is unapproved.

## Precommitted serial continuation

The formal Vgent source, native defaults and output interfaces are sealed in
`vgent_formal_prepared_01/manifest.json` before receiving any Vgent GPU outcome.
The independent coordinator `runtime/finish_candidates.py` waits for the wiring
job, requires successful completion and actual dual-stream video consumption,
then freezes the full original 5,130-input configuration. It submits one A100 job
at a time, at most 12 hours each. A normal video-boundary time checkpoint is
resumed with the exact same manifest; an incomplete video, wiring failure,
submission rejection, OOM or other fatal failure stops for review. No automatic
frame reduction, prompt editing, checkpoint substitution or failed-input retry.

Status is in `reports/serial_completion.json`; final per-task scores, formal
model calls, generated tokens, visual/text tokens and actual observation counts
are summarized automatically in `reports/results.json` and `reports/results.md`,
with Slurm allocation records in `reports/slurm_accounting.txt`. If Vgent cannot
pass wiring or memory acceptance its score remains unavailable and the reason
is retained. S-Agent and R4DSG exclusions and the explicit EGAgent pause remain
separate from scored results. Output caps and potential native JSON retries for
Vgent are disclosed in its configuration, never hidden as free preprocessing.

SimpleStream pilot output-format audit also passed without scoring: both
numerical and letter responses are parseable under the unchanged official
first-token parser, both use two generated tokens and neither hits the cap.
`reports/simple_pilot_format_check.json` records this additional audit; no input,
configuration, prompt or model output was modified.

## Status audit and durable continuation correction

SimpleStream job 17319 completed 0:0 on A100 gpu1 in 01:47:54. All 5,130
predictions and 288 video checkpoints exist; overall official eight-task macro
score is 39.71745778142598%. Formal peak allocated GPU memory is
12,602,502,144 bytes. Full task scores are in simple_formal_01/scores.json.
This is the independent recent-4 Spatial135K port, not the native-backbone paper
score or a strictly equal-budget comparison. No score-driven changes were made.

The previous external background submitter/coordinator no longer existed, and
no Vgent submission receipt was produced. Root cause was not established. The
stale orchestration status and interruption receipt are preserved. No Vgent GPU
calls had occurred. The replacement job **17362** runs orchestration inside its
Slurm allocation, independently of chat processes. Existing wiring and prepared
formal manifests were verified unchanged before submission. On wiring success
it freezes and runs the already-prepared formal implementation; failures stop
without altering inputs. If too little allocation time remains for the fixed
formal budget, it leaves the accepted frozen run ready for a fresh allocation.
A normal formal video-boundary time checkpoint likewise requires resubmission;
this replacement does not promise an external background auto-submit process.
No Proposal, AGENTS.md, PLAN.md or SandMem method code was changed.

## 2026-11-06 exact-input continuation

Vgent job 17362 passed wiring but stopped after 237 questions / 26 videos on an indexed Decord EOF failure. CPU sequential recovery passed all 373 source IDs and all 24 native processor hashes. Independent frozen continuation job 17414 is RUNNING on A100 gpu1, reusing those 237 predictions and evaluating only the remaining 4893. Method/configuration/scoring are unchanged. Details and all cost/provenance limits: [VGENT_DECODE_RECOVERY_20261106.md](VGENT_DECODE_RECOVERY_20261106.md). The original run and its failure remain preserved.

### Follow-up correction

Job 17414 failed during sequential decoding in the model process, with zero new MLLM calls. Its standalone CPU validation had passed, but deployment still re-decoded the file. New job 17432 uses the actual checksummed, previously validated 373-frame CPU array and isolates future decoder workers. The 237 original answers and all failed runs are preserved. See the second-failure section of VGENT_DECODE_RECOVERY_20261106.md. Scheduler RUNNING is not proof of source ingestion; check actual decode and query outputs.
