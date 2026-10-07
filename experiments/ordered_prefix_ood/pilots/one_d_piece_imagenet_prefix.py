import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import lpips
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from torchvision import transforms

from modeling.one_d_piece import OneDPiece


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def select_images(dataset_path, max_images):
    class_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    selected = []
    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            selected.append(images[0])
        if len(selected) == max_images:
            break
    return selected


def tensor_to_pil(image):
    image = (image.float().cpu().clamp(0, 1) * 255).byte()
    return Image.fromarray(image.permute(1, 2, 0).numpy())


@torch.inference_mode()
def image_metrics(original, reconstruction, lpips_model):
    reconstruction = reconstruction.float().clamp(0, 1)
    mse = (original.float() - reconstruction).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    lpips_values = lpips_model(original.float() * 2 - 1, reconstruction * 2 - 1).flatten()

    ssim_values = []
    for reference, result in zip(original, reconstruction):
        reference = reference.float().cpu().permute(1, 2, 0).numpy()
        result = result.cpu().permute(1, 2, 0).numpy()
        ssim_values.append(
            structural_similarity(reference, result, data_range=1.0, channel_axis=2)
        )
    return psnr.cpu().numpy(), np.asarray(ssim_values), lpips_values.cpu().numpy()


def save_preview(output_path, image_name, original, reconstructions, prefixes):
    cells = [tensor_to_pil(original)] + [tensor_to_pil(image) for image in reconstructions]
    labels = ["Original"] + [f"{prefix} tokens" for prefix in prefixes]
    canvas = Image.new("RGB", (256 * len(cells), 286), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (cell, label) in enumerate(zip(cells, labels)):
        canvas.paste(cell, (index * 256, 30))
        draw.text((index * 256 + 8, 8), label, fill="black")
    canvas.save(output_path / f"{image_name}.png")


def summarize(rows, prefixes):
    summary = {}
    for prefix in prefixes:
        prefix_rows = [row for row in rows if row["prefix_tokens"] == prefix]
        summary[str(prefix)] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in prefix_rows])),
                "std": float(np.std([row[metric] for row in prefix_rows])),
            }
            for metric in ("lpips", "psnr", "ssim")
        }
    return summary


def load_model(config_path, repository, device):
    config = OmegaConf.load(config_path)
    config.model.vq_model.finetune_decoder = True
    config.model.vq_model.strict_length_assertion = False
    checkpoint = hf_hub_download(repository, "pytorch_model.bin")
    model = OneDPiece(config)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval().requires_grad_(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/datasets/imagenet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot"))
    parser.add_argument("--config", type=Path, default=Path("configs/one-d-piece_s256.yaml"))
    parser.add_argument("--model", default="turing-motors/One-D-Piece-S-256")
    parser.add_argument("--prefixes", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir(exist_ok=True)

    device = "cuda"
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    paths = select_images(args.dataset, args.max_images)
    print(f"Selected {len(paths)} images from {args.dataset}")

    model = load_model(args.config, args.model, device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    rows = []

    for batch_start in range(0, len(paths), args.batch_size):
        batch_paths = paths[batch_start : batch_start + args.batch_size]
        originals = torch.stack([transform(Image.open(path)) for path in batch_paths]).to(device)
        with torch.inference_mode():
            _, result = model.encode(originals)
            tokens = result["min_encoding_indices"]
        assert tokens.shape[-1] == 256

        batch_reconstructions = []
        for prefix in args.prefixes:
            with torch.inference_mode():
                reconstructions = model.decode_tokens(tokens[:, :, :prefix]).float().clamp(0, 1)
            psnr, ssim, lpips_values = image_metrics(originals, reconstructions, lpips_model)
            for index, path in enumerate(batch_paths):
                rows.append(
                    {
                        "image": str(path.relative_to(args.dataset)),
                        "prefix_tokens": prefix,
                        "lpips": float(lpips_values[index]),
                        "psnr": float(psnr[index]),
                        "ssim": float(ssim[index]),
                    }
                )
            batch_reconstructions.append(reconstructions.cpu())

        if batch_start < 4:
            for index, path in enumerate(batch_paths):
                save_preview(
                    preview_dir,
                    path.stem,
                    originals[index].cpu(),
                    [batch[index] for batch in batch_reconstructions],
                    args.prefixes,
                )
        print(f"Processed {min(batch_start + len(batch_paths), len(paths))}/{len(paths)} images")

    with (args.output / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model,
        "dataset": str(args.dataset),
        "n_images": len(paths),
        "prefixes": args.prefixes,
        "metrics": summarize(rows, args.prefixes),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
