#!/bin/bash
#SBATCH --job-name=vsi-Vgent-Spatial-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export SANDMEM_ROOT="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
BASELINE_LIBRARY_DIRS="$(work/envs/spatial135k/bin/python -B -c 'import sys; from pathlib import Path; paths=[d for p in sys.path if p for d in [*Path(p).glob("nvidia/*/lib"),Path(p)/"cusparselt/lib"] if d.is_dir()]; print(":".join(dict.fromkeys(str(d) for d in paths)))')"
export LD_LIBRARY_PATH="$BASELINE_LIBRARY_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec work/envs/spatial135k/bin/python -u work/baselines_candidates_20261105/vgent_formal_01/runtime/vgent_full.py \
  --data-root work/baselines_vsibench_20261105 \
  --run work/baselines_candidates_20261105/vgent_formal_01
