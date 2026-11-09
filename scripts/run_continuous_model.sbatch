#!/bin/bash
#SBATCH --job-name=sandmem-continuous
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${SANDMEM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
exec "${SANDMEM_PYTHON:-python}" -u scripts/run_continuous_model.py "$@"
