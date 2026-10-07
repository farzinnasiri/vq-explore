#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?MODEL is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${GPU:?GPU is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$ROOT/outputs/$MODEL/$DATASET_NAME"
MANIFEST_PATH="${MANIFEST_PATH:-$ROOT/manifest.csv}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/.cache}"
IMAGE="nasiri/flextok-prefix-pilot:latest"
BATCH_SIZE=1
if [[ "$MODEL" == "one_d_piece" ]]; then
  IMAGE="nasiri/one-d-piece-prefix-pilot:latest"
  BATCH_SIZE=8
fi

mkdir -p "$OUTPUT" "$CACHE_ROOT/home" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"
docker run --rm \
  --gpus "device=$GPU" \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache/home \
  -e HF_HOME=/cache/huggingface \
  -e TORCH_HOME=/cache/torch \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$ROOT:/experiment" \
  -v "$CACHE_ROOT:/cache" \
  -v "$DATASET_ROOT:/dataset:ro" \
  -w /app \
  "$IMAGE" \
  python3 /experiment/run_prefix_experiment.py \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-root /dataset \
    --manifest "/experiment/$(basename "$MANIFEST_PATH")" \
    --output "/experiment/outputs/$MODEL/$DATASET_NAME" \
    --batch-size "$BATCH_SIZE" \
    --checkpoint-every 25
