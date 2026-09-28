#!/bin/bash
# Longitude-origin (seam) equivariance study.  Build + validate the benchmark once on a
# CPU node first:
#   editpano seam build --data-root DATA --prompts PROMPTS --out-dir seam/dataset
#   editpano seam validate --manifest seam/dataset/manifest.json
# then:
#   MANIFEST=seam/dataset/manifest.json OUT=seam/runs sbatch --array=0-2 scripts/slurm/seam_study.sbatch
#
# Environment: STRATEGIES, SEEDS (default 0), CONFIG, EXTRA (e.g. "--tiled"), CONDA_ENV.
#SBATCH --job-name=editpano-seam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
: "${MANIFEST:?set MANIFEST}" "${OUT:?set OUT}"
[ -n "${CONDA_ENV:-}" ] && { source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
editpano -v seam run --manifest "$MANIFEST" --out-dir "$OUT" \
  --strategies ${STRATEGIES:-identity current split direct_circular canonical} \
  --seeds ${SEEDS:-0} ${CONFIG:+--config "$CONFIG"} \
  --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard-index "${SLURM_ARRAY_TASK_ID:-0}" ${EXTRA:-}
