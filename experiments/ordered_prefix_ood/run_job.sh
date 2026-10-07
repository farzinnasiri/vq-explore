#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?MODEL is required: flextok or one_d_piece}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${GPU:?GPU is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MANIFEST_PATH:?MANIFEST_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required: writable results outside the repository}"
: "${CACHE_ROOT:?CACHE_ROOT is required: writable model cache outside the repository}"
case "$MODEL" in
  flextok) DEFAULT_IMAGE="nasiri/flextok-prefix-pilot:latest"; DEFAULT_BATCH=1 ;;
  one_d_piece) DEFAULT_IMAGE="nasiri/one-d-piece-prefix-pilot:latest"; DEFAULT_BATCH=8 ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
RUNNER="${RUNNER:-run_prefix_experiment.py}"
[[ "$RUNNER" == "run_prefix_experiment.py" || "$RUNNER" == "run_prefix_experiment_batched.py" ]] || exit 2
[[ -f "$MANIFEST_PATH" && -d "$DATASET_ROOT" ]] || { echo "Missing manifest or dataset" >&2; exit 2; }
MANIFEST_PATH="$(cd "$(dirname "$MANIFEST_PATH")" && pwd)/$(basename "$MANIFEST_PATH")"
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"
OUTPUT="$OUTPUT_ROOT/$MODEL/$DATASET_NAME"

mkdir -p "$OUTPUT" "$CACHE_ROOT/home" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"
OUTPUT="$(cd "$OUTPUT" && pwd)"
CACHE_ROOT="$(cd "$CACHE_ROOT" && pwd)"
docker run --rm \
  --gpus "device=$GPU" \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache/home \
  -e HF_HOME=/cache/huggingface \
  -e TORCH_HOME=/cache/torch \
  -e TORCHDYNAMO_DISABLE=1 \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v "$ROOT:/experiment:ro" \
  -v "$MANIFEST_PATH:/manifest.csv:ro" \
  -v "$OUTPUT:/output" \
  -v "$CACHE_ROOT:/cache" \
  -v "$DATASET_ROOT:/dataset:ro" \
  -w /app \
  "${IMAGE_NAME:-$DEFAULT_IMAGE}" \
  python3 "/experiment/$RUNNER" \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --dataset-root /dataset \
    --manifest /manifest.csv \
    --output /output \
    --batch-size "${BATCH_SIZE:-$DEFAULT_BATCH}" \
    --seed "${SEED:-20260812}" --timesteps "${TIMESTEPS:-25}" \
    --guidance-scale "${GUIDANCE_SCALE:-15.0}" \
    --checkpoint-every 25
