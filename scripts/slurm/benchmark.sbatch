#!/bin/bash
# Reference-ROI benchmark as a SLURM array (one shard per task).
#
#   DATA_ROOT=/path/to/data/test PROMPTS=/path/to/prompts.json OUT=runs/reference \
#     sbatch --array=0-2 scripts/slurm/benchmark.sbatch
#
# Environment: ARMS (default "whole_erp editpano"), CONFIG (YAML), EXTRA (extra CLI args),
#              CONDA_ENV (activated when set), HF_HUB_OFFLINE=1 recommended on compute nodes.
#SBATCH --job-name=editpano-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT}" "${PROMPTS:?set PROMPTS}" "${OUT:?set OUT}"
[ -n "${CONDA_ENV:-}" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
editpano -v bench run --data-root "$DATA_ROOT" --prompts "$PROMPTS" --out-dir "$OUT" \
  --arms ${ARMS:-whole_erp editpano} ${CONFIG:+--config "$CONFIG"} \
  --num-shards "$NUM_SHARDS" --shard-index "$SHARD" ${EXTRA:-}
