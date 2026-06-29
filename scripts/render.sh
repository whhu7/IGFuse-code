#!/usr/bin/env bash
set -euo pipefail

# Run from repo root regardless of cwd
cd "$(dirname "$0")/.."

DATASET_PATH="./datasets/synthetic/scene_001"
MODEL_PATH="./output/synthetic/scene_001"
ITERATION=20000
TEST_STATE_INDEX=0

# Optional: pick a GPU
# export CUDA_VISIBLE_DEVICES=0

python render.py \
  --dataset_path "${DATASET_PATH}" \
  --model_path "${MODEL_PATH}" \
  --iteration "${ITERATION}" \
  --test_state_index "${TEST_STATE_INDEX}"
