#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU SHARD_NAME BATCH_SIZE" >&2
  exit 2
fi

GPU="$1"
SHARD="$2"
BATCH_SIZE="$3"
ROOT="${ROOT:-/megaverse/storage/nasiri/ordered-prefix-ood-full}"
MANIFEST="$ROOT/manifests/part2_shards/$SHARD.csv"
IMAGE="nasiri/flextok-prefix-pilot:latest"

run_dataset() {
  local dataset="$1"
  local dataset_root="$2"
  local output="$ROOT/outputs/flextok_matched10k_part2_shards/$SHARD/$dataset"

  mkdir -p "$output"
  docker run --rm \
    --name "opfx_ft_${SHARD}_${dataset}" \
    --gpus "device=$GPU" \
    --shm-size=16g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/cache/home \
    -e HF_HOME=/cache/huggingface \
    -e TORCH_HOME=/cache/torch \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v "$ROOT:/experiment" \
    -v "$ROOT/cache-nasiri:/cache" \
    -v "$dataset_root:/dataset:ro" \
    -w /app \
    "$IMAGE" \
    python3 /experiment/run_prefix_experiment_batched.py \
      --model flextok \
      --dataset-name "$dataset" \
      --dataset-root /dataset \
      --manifest "/experiment/manifests/part2_shards/$SHARD.csv" \
      --output "/experiment/outputs/flextok_matched10k_part2_shards/$SHARD/$dataset" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-every 240 \
      --preview-count 5 \
      --seed 20260812 \
      --timesteps 25 \
      --guidance-scale 15.0
}

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }

run_dataset imagenet /home/nasiri/datasets/shared/imagenet/val
run_dataset imagenet_r /megaverse/datasets/projects/vqtok/imagenet-r
