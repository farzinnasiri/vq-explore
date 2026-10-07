import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms


PREFIXES = [8, 16, 32, 64, 96, 128, 160, 208, 256]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["flextok", "one_d_piece"], required=True)
    parser.add_argument(
        "--dataset-name",
        choices=[
            "imagenet",
            "imagenet_r",
            "imagenet_sketch",
            "imagenet_objectnet",
            "objectnet",
        ],
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    return parser.parse_args()


def load_manifest(path, dataset_name):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset_name]
    if not rows:
        raise ValueError(f"Manifest has no rows for {dataset_name}")
    return rows


def preprocess():
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )


def metric_values(original, reconstruction, lpips_model):
    original = original.float().clamp(0, 1)
    reconstruction = reconstruction.float().clamp(0, 1)
    mse = (original - reconstruction).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    lpips_values = lpips_model(original * 2 - 1, reconstruction * 2 - 1).flatten()
    ssim_values = []
    for reference, result in zip(original, reconstruction):
        reference = reference.cpu().permute(1, 2, 0).numpy()
        result = result.cpu().permute(1, 2, 0).numpy()
        ssim_values.append(
            structural_similarity(reference, result, data_range=1.0, channel_axis=2)
        )
    return psnr.cpu().numpy(), np.asarray(ssim_values), lpips_values.cpu().numpy()


def save_image(path, tensor):
    image = (tensor.float().cpu().clamp(0, 1) * 255).round().byte()
    Image.fromarray(image.permute(1, 2, 0).numpy()).save(path)


def image_seed(base_seed, relative_path):
    digest = hashlib.sha256(f"{base_seed}:{relative_path}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def enable_turing_fallback(model):
    from flextok.model.layers.attention import FlexAttention
    from flextok.model.preprocessors.flex_seq_packing import BlockWiseSequencePacker

    for module in model.modules():
        if isinstance(module, FlexAttention):
            module.use_flex_attention = False
        elif isinstance(module, BlockWiseSequencePacker):
            module.compile_block_mask = False
            module.return_materialized_mask = True


class FlexTokRunner:
    name = "EPFL-VILAB/flextok_d12_d12_in1k"

    def __init__(self, device, timesteps, guidance_scale):
        from flextok.flextok_wrapper import FlexTokFromHub

        self.device = device
        self.timesteps = timesteps
        self.guidance_scale = guidance_scale
        self.model = FlexTokFromHub.from_pretrained(self.name).to(device).eval()
        enable_turing_fallback(self.model)

    def prepare(self, images):
        return images * 2 - 1

    def encode(self, images):
        tokens = self.model.tokenize(images)
        assert all(sequence.shape == (1, 256) for sequence in tokens)
        return tokens

    def decode(self, tokens, prefix, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        reconstruction = self.model.detokenize(
            [sequence[:, :prefix] for sequence in tokens],
            timesteps=self.timesteps,
            guidance_scale=self.guidance_scale,
            perform_norm_guidance=True,
            generator=generator,
            verbose=False,
        )
        return (reconstruction.float().clamp(-1, 1) + 1) / 2


class OneDPieceRunner:
    name = "turing-motors/One-D-Piece-S-256"

    def __init__(self, device):
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        sys.path.append("/app")
        from modeling.one_d_piece import OneDPiece

        config = OmegaConf.load("/app/configs/one-d-piece_s256.yaml")
        config.model.vq_model.finetune_decoder = True
        config.model.vq_model.strict_length_assertion = False
        checkpoint = hf_hub_download(self.name, "pytorch_model.bin")
        model = OneDPiece(config)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = model.to(device).eval().requires_grad_(False)

    def prepare(self, images):
        return images

    def encode(self, images):
        _, result = self.model.encode(images)
        tokens = result["min_encoding_indices"]
        assert tokens.shape[1:] == (1, 256)
        return tokens

    def decode(self, tokens, prefix, seed):
        del seed
        return self.model.decode_tokens(tokens[:, :, :prefix]).float().clamp(0, 1)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(args, runner, count, manifest_sha256):
    metadata = {
        "model": runner.name,
        "dataset": args.dataset_name,
        "n_images": count,
        "prefixes": PREFIXES,
        "preprocessing": "RGB, resize shorter edge to 256, center crop 256",
        "lpips_backbone": "alex",
        "seed": args.seed,
        "dataset_root": str(args.dataset_root),
        "manifest_sha256": manifest_sha256,
    }
    if args.model == "flextok":
        metadata.update(
            {
                "decoder_timesteps": args.timesteps,
                "guidance_scale": args.guidance_scale,
                "perform_norm_guidance": True,
                "attention_backend": "PyTorch SDPA",
                "decoder_noise": "one deterministic generator stream per batch, reset for every prefix",
            }
        )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


FIELDNAMES = [
    "model",
    "dataset",
    "wnid",
    "image",
    "original_width",
    "original_height",
    "prefix_tokens",
    "seed",
    "decode_seconds",
    "lpips",
    "psnr",
    "ssim",
]


def load_existing_metrics(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["image"], int(row["prefix_tokens"])): row for row in rows}


def write_metrics(path, rows):
    temporary = path.with_suffix(".tmp")
    ordered = sorted(rows.values(), key=lambda row: (row["image"], int(row["prefix_tokens"])))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)
    rows = load_manifest(args.manifest, args.dataset_name)
    manifest_sha256 = file_sha256(args.manifest)
    metrics_path = args.output / "per_image_metrics.csv"
    metadata_path = args.output / "metadata.json"
    if metrics_path.exists():
        if not metadata_path.exists():
            raise ValueError(f"Cannot resume {metrics_path} without metadata.json")
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata["manifest_sha256"] != manifest_sha256:
            raise ValueError("Existing metrics were produced from a different manifest")
    transform = preprocess()
    device = "cuda"
    torch.manual_seed(args.seed)
    runner = (
        FlexTokRunner(device, args.timesteps, args.guidance_scale)
        if args.model == "flextok"
        else OneDPieceRunner(device)
    )
    lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    metric_rows = load_existing_metrics(metrics_path)
    completed_images = {
        row["image"]
        for row in rows
        if all((row["image"], prefix) in metric_rows for prefix in PREFIXES)
    }
    rows = [row for row in rows if row["image"] not in completed_images]
    write_metadata(args, runner, len(rows) + len(completed_images), manifest_sha256)

    processed_since_checkpoint = 0
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        source_images = []
        original_sizes = []
        for row in batch_rows:
            with Image.open(args.dataset_root / row["image"]) as image:
                original_sizes.append(image.size)
                source_images.append(transform(image))
        source = torch.stack(source_images).to(device)
        model_input = runner.prepare(source)
        batch_seed = image_seed(args.seed, "|".join(row["image"] for row in batch_rows))
        with torch.inference_mode():
            tokens = runner.encode(model_input)
            for prefix in PREFIXES:
                if all((row["image"], prefix) in metric_rows for row in batch_rows):
                    continue
                torch.cuda.synchronize()
                decode_started = time.perf_counter()
                reconstruction = runner.decode(tokens, prefix, batch_seed)
                torch.cuda.synchronize()
                decode_seconds = time.perf_counter() - decode_started
                if not torch.isfinite(reconstruction).all():
                    raise FloatingPointError(f"Non-finite reconstruction at prefix {prefix}")
                psnr, ssim, lpips_values = metric_values(source, reconstruction, lpips_model)
                for index, row in enumerate(batch_rows):
                    metric_rows[(row["image"], prefix)] = {
                            "model": runner.name,
                            "dataset": args.dataset_name,
                            "wnid": row["wnid"],
                            "image": row["image"],
                            "original_width": original_sizes[index][0],
                            "original_height": original_sizes[index][1],
                            "prefix_tokens": prefix,
                            "seed": batch_seed,
                            "decode_seconds": decode_seconds / len(batch_rows),
                            "lpips": float(lpips_values[index]),
                            "psnr": float(psnr[index]),
                            "ssim": float(ssim[index]),
                        }
                    if start + index < args.preview_count:
                        stem = f"{start + index:03d}_{row['wnid']}"
                        if prefix == PREFIXES[0]:
                            save_image(preview_dir / f"{stem}_original.png", source[index])
                        save_image(
                            preview_dir / f"{stem}_{prefix:03d}_tokens.png",
                            reconstruction[index],
                        )
        processed_since_checkpoint += len(batch_rows)
        if processed_since_checkpoint >= args.checkpoint_every:
            write_metrics(metrics_path, metric_rows)
            processed_since_checkpoint = 0
        print(f"Processed {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    if processed_since_checkpoint or not metrics_path.exists():
        write_metrics(metrics_path, metric_rows)


if __name__ == "__main__":
    main()
