#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of cwd
cd "$(dirname "$0")/.."

DATASET_PATH="./datasets/synthetic/scene_003"
NUM_PSEUDO=100
SEED=42
PADDING_RATIO=0.05          # expand pseudo XY bounds by this fraction of observed span
MIN_DISTANCE_RATIO=1.0

python pseudo_gen.py \
  --dataset_path "${DATASET_PATH}" \
  --num_pseudo "${NUM_PSEUDO}" \
  --seed "${SEED}" \
  --padding_ratio "${PADDING_RATIO}" \
  --min_distance_ratio "${MIN_DISTANCE_RATIO}"
