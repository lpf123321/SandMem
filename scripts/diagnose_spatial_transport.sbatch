#!/bin/bash
#SBATCH --job-name=spatial-api-diagnose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
SPATIAL_LIBRARY_DIRS="$(work/envs/spatial135k/bin/python -B -c 'import sys; from pathlib import Path; paths=[d for p in sys.path if p for d in [*Path(p).glob("nvidia/*/lib"),Path(p)/"cusparselt/lib"] if d.is_dir()]; print(":".join(dict.fromkeys(str(d) for d in paths)))')"
export LD_LIBRARY_PATH="$SPATIAL_LIBRARY_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec work/envs/spatial135k/bin/python -u scripts/diagnose_spatial_transport.py \
  --reference "$PWD/work/migration_135k_20261103" \
  --official-root "$PWD/work/spatial-mllm-official" --output "${1:?new diagnostic output}"
