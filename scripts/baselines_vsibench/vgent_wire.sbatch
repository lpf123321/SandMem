#!/bin/bash
#SBATCH --job-name=vsi-Vgent-Spatial-wire
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export SANDMEM_ROOT="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
BASELINE_LIBRARY_DIRS="$(work/envs/spatial135k/bin/python -B -c 'import sys; from pathlib import Path; paths=[d for p in sys.path if p for d in [*Path(p).glob("nvidia/*/lib"),Path(p)/"cusparselt/lib"] if d.is_dir()]; print(":".join(dict.fromkeys(str(d) for d in paths)))')"
export LD_LIBRARY_PATH="$BASELINE_LIBRARY_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec work/envs/spatial135k/bin/python -u work/baselines_candidates_20261105/vgent_wire_frozen_02/runtime/vgent_wire.py \
  --inputs work/baselines_candidates_20261105/vgent_wire_frozen_02/inputs.jsonl \
  --output work/baselines_candidates_20261105/vgent_wire_01 \
  --data-root work/baselines_vsibench_20261105 \
  --repo work/baselines_candidates_20261105/repos/vgent
