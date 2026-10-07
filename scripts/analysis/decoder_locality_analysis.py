from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised in non-docker environments
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

try:
    import torch
except ImportError:  # pragma: no cover - exercised in non-docker environments
    torch = None

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - exercised in non-docker environments
    structural_similarity = None


EPS = 1e-8
INSIDE_MIN = 0.02

DEFAULT_METRICS = ("pixel_locality", "psnr_ssim", "lpips", "merged_fidelity")
OPTIONAL_METRICS = ("token_sanity",)
ALL_METRICS = DEFAULT_METRICS + OPTIONAL_METRICS
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_ARG = "all"
DEFAULT_EXPERIMENT_NAME = ""
DEFAULT_NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
DEFAULT_GPU_BATCH_SIZE = 64
DEFAULT_LPIPS_BATCH_SIZE = 64
DEFAULT_MAX_IMAGES = 0
DEFAULT_SEED = 0

CANONICAL_KEY_COLUMNS = [
    "experiment",
    "experiment_dirname",
    "image_id",
    "fraction_label",
    "mode",
]

CANONICAL_VARIANT_COLUMNS = [
    "experiment",
    "experiment_dirname",
    "image_id",
    "mode",
    "fraction_label",
    "target_fraction",
    "actual_fraction",
    "clean_path",
    "edit_path",
    "x0",
    "y0",
    "x1",
    "y1",
    "j0",
    "i0",
    "j1",
    "i1",
    "token_grid_h",
    "token_grid_w",
]


def script_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def analysis_cache_root(output_root: Path) -> Path:
    return output_root / "analysis_cache"


def experiment_bundle_dir(output_root: Path, experiment_dirname: str) -> Path:
    return analysis_cache_root(output_root) / experiment_dirname


def sorted_metadata_files(input_root: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"metadata_part_(\d+)\.jsonl$", path.name)
        if match is None:
            return (10**9, path.name)
        return (int(match.group(1)), path.name)

    return sorted(input_root.glob("metadata_part_*.jsonl"), key=key)


def load_img_float01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def parse_metrics(raw: str) -> list[str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one metric group must be selected")
    if "all" in parts:
        parts = list(DEFAULT_METRICS)
    unknown = [part for part in parts if part not in ALL_METRICS and part != "merged_fidelity"]
    if unknown:
        raise ValueError(f"Unknown metric group(s): {unknown}")

    selected = list(dict.fromkeys(parts))
    if "merged_fidelity" in selected:
        if "psnr_ssim" not in selected:
            selected.append("psnr_ssim")
        if "lpips" not in selected:
            selected.append("lpips")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export decoder locality analysis tables from one robustness dataset root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to one robustness dataset directory containing images/ and metadata_part_*.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root where analysis_cache/ will be created. Defaults to the vq-explore repo root.",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS_ARG,
        help="Comma-separated metric groups: pixel_locality,psnr_ssim,lpips,merged_fidelity,token_sanity,all",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Optional experiment label. Defaults to the input directory basename.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="CPU worker count for process-pool metrics.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=DEFAULT_GPU_BATCH_SIZE,
        help="Reserved for future GPU metric groups. Kept for CLI parity.",
    )
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=DEFAULT_LPIPS_BATCH_SIZE,
        help="Number of row pairs evaluated per LPIPS forward pass.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="Optional cap on source images before row expansion. 0 means no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in the experiment bundle directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic ordering and future extensions.",
    )
    return parser


def normalize_modes(record: dict) -> list[str]:
    modes = record.get("token_edit_modes")
    if modes is not None:
        return [str(mode) for mode in modes]

    legacy_mode = record.get("token_edit_mode")
    if legacy_mode is not None:
        return [str(legacy_mode)]

    legacy_by_mode = record.get("indices_edit_by_mode")
    if isinstance(legacy_by_mode, dict):
        return [str(mode) for mode in legacy_by_mode.keys()]

    new_by_fraction = record.get("indices_edit_by_fraction_and_mode")
    if isinstance(new_by_fraction, dict):
        discovered = []
        for value in new_by_fraction.values():
            if isinstance(value, dict):
                discovered.extend(str(mode) for mode in value.keys())
        return sorted(set(discovered))

    return []


def fraction_labels_for_record(record: dict) -> list[str]:
    patches = record.get("patches_by_fraction")
    if isinstance(patches, dict) and patches:
        return sorted(patches.keys(), key=lambda item: int(item))

    sweep = record.get("patch_fraction_sweep")
    if sweep:
        return [str(int(round(float(fraction) * 100.0))) for fraction in sweep]

    legacy_bbox = record.get("patch_bbox_px")
    if legacy_bbox is not None:
        return ["legacy"]

    return []


def find_single_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[0]


def resolve_clean_path(sample_dir: Path) -> Path | None:
    direct = sample_dir / "1_recon_clean.png"
    if direct.exists():
        return direct
    return find_single_file(sample_dir, "*_recon_clean.png")


def resolve_edit_path(sample_dir: Path, fraction_label: str, mode: str) -> Path | None:
    if fraction_label == "legacy":
        direct_mode = sample_dir / f"2_recon_token_edit_{mode}.png"
        if direct_mode.exists():
            return direct_mode
        direct = sample_dir / "2_recon_token_edit.png"
        if direct.exists():
            return direct
        return find_single_file(sample_dir, f"*_recon_token_edit_{mode}.png")

    direct = sample_dir / f"2_recon_token_edit_patch{fraction_label}_{mode}.png"
    if direct.exists():
        return direct
    return find_single_file(sample_dir, f"*_recon_token_edit_patch{fraction_label}_{mode}.png")


def ensure_bbox(value, fallback=None):
    bbox = value if value is not None else fallback
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    return [int(coord) for coord in bbox]


def build_variant_rows_from_record(
    record: dict,
    *,
    images_root: Path,
    experiment: str,
    experiment_dirname: str,
) -> list[dict]:
    image_id = record.get("image_id")
    if not image_id:
        return []

    sample_dir = images_root / image_id
    clean_path = resolve_clean_path(sample_dir)
    if clean_path is None:
        return []

    token_grid_hw = record.get("token_grid_hw")
    if not token_grid_hw or len(token_grid_hw) != 2:
        return []

    token_grid_h, token_grid_w = int(token_grid_hw[0]), int(token_grid_hw[1])
    modes = normalize_modes(record)
    fraction_labels = fraction_labels_for_record(record)
    if not modes or not fraction_labels:
        return []

    patches_by_fraction = record.get("patches_by_fraction") or {}
    rows = []
    for fraction_label in fraction_labels:
        if fraction_label == "legacy":
            bbox_px = ensure_bbox(record.get("patch_bbox_px"))
            bbox_tok = ensure_bbox(record.get("patch_bbox_tok"))
            if bbox_px is None or bbox_tok is None:
                continue
            patch_area = max(0, bbox_tok[2] - bbox_tok[0]) * max(0, bbox_tok[3] - bbox_tok[1])
            actual_fraction = patch_area / float(max(1, token_grid_h * token_grid_w))
            patch_meta = {
                "target_fraction": actual_fraction,
                "actual_fraction": actual_fraction,
                "patch_bbox_px": bbox_px,
                "patch_bbox_tok": bbox_tok,
            }
        else:
            patch_meta = patches_by_fraction.get(fraction_label)
            if not isinstance(patch_meta, dict):
                continue

        bbox_px = ensure_bbox(patch_meta.get("patch_bbox_px"), record.get("patch_bbox_px"))
        bbox_tok = ensure_bbox(patch_meta.get("patch_bbox_tok"), record.get("patch_bbox_tok"))
        if bbox_px is None or bbox_tok is None:
            continue

        target_fraction = float(patch_meta.get("target_fraction", patch_meta.get("actual_fraction", 0.0)))
        actual_fraction = float(patch_meta.get("actual_fraction", target_fraction))

        for mode in modes:
            edit_path = resolve_edit_path(sample_dir, fraction_label, mode)
            if edit_path is None:
                continue

            rows.append(
                {
                    "experiment": experiment,
                    "experiment_dirname": experiment_dirname,
                    "image_id": image_id,
                    "mode": str(mode),
                    "fraction_label": str(fraction_label),
                    "target_fraction": target_fraction,
                    "actual_fraction": actual_fraction,
                    "clean_path": str(clean_path),
                    "edit_path": str(edit_path),
                    "x0": int(bbox_px[0]),
                    "y0": int(bbox_px[1]),
                    "x1": int(bbox_px[2]),
                    "y1": int(bbox_px[3]),
                    "j0": int(bbox_tok[0]),
                    "i0": int(bbox_tok[1]),
                    "j1": int(bbox_tok[2]),
                    "i1": int(bbox_tok[3]),
                    "token_grid_h": token_grid_h,
                    "token_grid_w": token_grid_w,
                }
            )
    return rows


def iter_metadata_records(input_root: Path) -> Iterable[dict]:
    for metadata_path in sorted_metadata_files(input_root):
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)


def build_variants_dataframe(
    input_root: Path,
    *,
    experiment: str,
    experiment_dirname: str,
    max_images: int | None,
) -> pd.DataFrame:
    images_root = input_root / "images"
    rows = []
    for image_index, record in enumerate(iter_metadata_records(input_root)):
        if max_images is not None and image_index >= max_images:
            break
        rows.extend(
            build_variant_rows_from_record(
                record,
                images_root=images_root,
                experiment=experiment,
                experiment_dirname=experiment_dirname,
            )
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_VARIANT_COLUMNS)

    frame = frame[CANONICAL_VARIANT_COLUMNS].copy()
    frame = frame.sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)
    return frame


def _record_edit_indices(record: dict, fraction_label: str, mode: str):
    if fraction_label == "legacy":
        if "indices_edit" in record and record.get("token_edit_mode") == mode:
            return record.get("indices_edit")
        by_mode = record.get("indices_edit_by_mode") or {}
        return by_mode.get(mode)

    by_fraction = record.get("indices_edit_by_fraction_and_mode") or {}
    return (by_fraction.get(fraction_label) or {}).get(mode)


def build_token_sanity_dataframe(
    input_root: Path,
    *,
    experiment: str,
    experiment_dirname: str,
    max_images: int | None,
) -> pd.DataFrame:
    rows = []
    for image_index, record in enumerate(iter_metadata_records(input_root)):
        if max_images is not None and image_index >= max_images:
            break

        image_id = record.get("image_id")
        token_grid_hw = record.get("token_grid_hw")
        indices_clean = record.get("indices_clean")
        if not image_id or indices_clean is None or not token_grid_hw or len(token_grid_hw) != 2:
            continue

        h_tok, w_tok = int(token_grid_hw[0]), int(token_grid_hw[1])
        total_tokens = h_tok * w_tok
        z_clean = np.asarray(indices_clean, dtype=np.int64)
        if z_clean.size != total_tokens:
            continue

        for fraction_label in fraction_labels_for_record(record):
            if fraction_label == "legacy":
                patch_bbox_tok = ensure_bbox(record.get("patch_bbox_tok"))
                target_fraction = None
                actual_fraction = None
            else:
                patch_meta = (record.get("patches_by_fraction") or {}).get(fraction_label) or {}
                patch_bbox_tok = ensure_bbox(patch_meta.get("patch_bbox_tok"))
                target_fraction = patch_meta.get("target_fraction")
                actual_fraction = patch_meta.get("actual_fraction")

            if patch_bbox_tok is None:
                continue

            j0, i0, j1, i1 = map(int, patch_bbox_tok)
            patch_mask = np.zeros((h_tok, w_tok), dtype=bool)
            patch_mask[i0:i1, j0:j1] = True
            patch_mask_flat = patch_mask.reshape(-1)
            patch_tokens = int(patch_mask_flat.sum())
            if patch_tokens == 0:
                continue

            if actual_fraction is None:
                actual_fraction = patch_tokens / float(max(1, total_tokens))
            if target_fraction is None:
                target_fraction = actual_fraction

            for mode in normalize_modes(record):
                indices_edit = _record_edit_indices(record, fraction_label, mode)
                if indices_edit is None:
                    continue

                z_edit = np.asarray(indices_edit, dtype=np.int64)
                if z_edit.size != total_tokens:
                    continue

                changed = z_edit != z_clean
                inside = float(changed[patch_mask_flat].mean()) if patch_tokens else np.nan
                outside = float(changed[~patch_mask_flat].mean()) if patch_tokens < total_tokens else np.nan
                overall = float(changed.mean())

                rows.append(
                    {
                        "experiment": experiment,
                        "experiment_dirname": experiment_dirname,
                        "image_id": image_id,
                        "mode": str(mode),
                        "fraction_label": str(fraction_label),
                        "target_fraction": float(target_fraction),
                        "actual_fraction": float(actual_fraction),
                        "j0": j0,
                        "i0": i0,
                        "j1": j1,
                        "i1": i1,
                        "token_grid_h": h_tok,
                        "token_grid_w": w_tok,
                        "edit_frac_overall": overall,
                        "edit_frac_inside": inside,
                        "edit_frac_outside": outside,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)


def group_payloads_from_variants(variants: pd.DataFrame) -> list[dict]:
    payloads = []
    group_cols = ["experiment", "experiment_dirname", "image_id", "clean_path"]
    for keys, group in variants.groupby(group_cols, sort=False):
        experiment, experiment_dirname, image_id, clean_path = keys
        payloads.append(
            {
                "experiment": experiment,
                "experiment_dirname": experiment_dirname,
                "image_id": image_id,
                "clean_path": clean_path,
                "rows": group.to_dict("records"),
            }
        )
    return payloads


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, height: int, width: int):
    x0 = max(0, min(width, int(x0)))
    x1 = max(0, min(width, int(x1)))
    y0 = max(0, min(height, int(y0)))
    y1 = max(0, min(height, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def delta_map(x_clean: np.ndarray, x_edit: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x_edit - x_clean), axis=-1)


def inside_outside_stats(delta: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> dict:
    inside = delta[y0:y1, x0:x1]
    inside_sum = float(inside.sum())
    inside_count = int(inside.size)

    total_sum = float(delta.sum())
    total_count = int(delta.size)
    outside_sum = total_sum - inside_sum
    outside_count = total_count - inside_count

    inside_mean = inside_sum / (inside_count + EPS)
    outside_mean = outside_sum / (outside_count + EPS)

    outside_parts = []
    if y0 > 0:
        outside_parts.append(delta[:y0, :].ravel())
    if y1 < delta.shape[0]:
        outside_parts.append(delta[y1:, :].ravel())
    if x0 > 0:
        outside_parts.append(delta[y0:y1, :x0].ravel())
    if x1 < delta.shape[1]:
        outside_parts.append(delta[y0:y1, x1:].ravel())
    outside_flat = np.concatenate(outside_parts) if outside_parts else np.array([], dtype=delta.dtype)

    leakage_ratio = outside_mean / (inside_mean + EPS)
    leakage_ratio_safe = outside_mean / (inside_mean + EPS)
    outside_p99 = float(np.percentile(outside_flat, 99)) if outside_flat.size else 0.0
    outside_p999 = float(np.percentile(outside_flat, 99.9)) if outside_flat.size else 0.0

    return {
        "inside_change": inside_mean,
        "outside_change": outside_mean,
        "leakage_ratio": leakage_ratio,
        "leakage_ratio_safe": leakage_ratio_safe,
        "outside_p99": outside_p99,
        "outside_p999": outside_p999,
    }


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if structural_similarity is None:
        raise RuntimeError("PSNR/SSIM metric requested, but scikit-image is not installed")
    height, width = a.shape[:2]
    win = min(7, height, width)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return float("nan")
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0, win_size=win))


def compute_psnr(a: np.ndarray, b: np.ndarray, *, data_range: float) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def _pixel_locality_worker(payload: dict) -> list[dict]:
    clean = load_img_float01(Path(payload["clean_path"]))
    height, width = clean.shape[:2]
    out_rows = []

    for row in payload["rows"]:
        edit_path = Path(row["edit_path"])
        if not edit_path.exists():
            continue

        edit = load_img_float01(edit_path)
        if edit.shape != clean.shape:
            continue

        bbox = clamp_bbox(row["x0"], row["y0"], row["x1"], row["y1"], height, width)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        metrics = inside_outside_stats(delta_map(clean, edit), x0, y0, x1, y1)
        out_rows.append({**row, **metrics})

    return out_rows


def _psnr_ssim_worker(payload: dict) -> list[dict]:
    clean = load_img_float01(Path(payload["clean_path"]))
    height, width = clean.shape[:2]
    out_rows = []

    for row in payload["rows"]:
        edit_path = Path(row["edit_path"])
        if not edit_path.exists():
            continue

        edit = load_img_float01(edit_path)
        if edit.shape != clean.shape:
            continue

        bbox = clamp_bbox(row["x0"], row["y0"], row["x1"], row["y1"], height, width)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        clean_patch = clean[y0:y1, x0:x1, :]
        edit_patch = edit[y0:y1, x0:x1, :]

        out_rows.append(
            {
                **row,
                "psnr_full": compute_psnr(clean, edit, data_range=1.0),
                "ssim_full": safe_ssim(clean, edit),
                "psnr_patch": compute_psnr(clean_patch, edit_patch, data_range=1.0),
                "ssim_patch": safe_ssim(clean_patch, edit_patch),
            }
        )

    return out_rows


def run_worker_groups(
    payloads: list[dict],
    worker,
    *,
    num_workers: int,
    desc: str,
) -> pd.DataFrame:
    if not payloads:
        return pd.DataFrame()

    rows = []
    if num_workers <= 1:
        iterator = payloads
        for payload in tqdm(iterator, total=len(payloads), desc=desc):
            rows.extend(worker(payload))
        return pd.DataFrame(rows)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        iterator = executor.map(worker, payloads, chunksize=8)
        for batch_rows in tqdm(iterator, total=len(payloads), desc=desc):
            rows.extend(batch_rows)

    return pd.DataFrame(rows)


def to_lpips_tensor_batch(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack([image.transpose(2, 0, 1) for image in images], axis=0)
    tensor = torch.from_numpy(array).float().to(device, non_blocking=True)
    return tensor * 2.0 - 1.0


def compute_lpips_dataframe(variants: pd.DataFrame, *, batch_size: int) -> pd.DataFrame:
    if torch is None:
        raise RuntimeError("LPIPS metric requested, but torch is not installed")
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS metric requested, but the lpips package is not installed") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    batch_meta = []
    batch_clean_full = []
    batch_edit_full = []
    batch_clean_patch = []
    batch_edit_patch = []
    batch_patch_shape = None

    def flush():
        nonlocal batch_patch_shape
        if not batch_meta:
            return

        with torch.no_grad():
            clean_full = to_lpips_tensor_batch(batch_clean_full, device)
            edit_full = to_lpips_tensor_batch(batch_edit_full, device)
            clean_patch = to_lpips_tensor_batch(batch_clean_patch, device)
            edit_patch = to_lpips_tensor_batch(batch_edit_patch, device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    full_vals = model(clean_full, edit_full).view(-1)
                    patch_vals = model(clean_patch, edit_patch).view(-1)
            else:
                full_vals = model(clean_full, edit_full).view(-1)
                patch_vals = model(clean_patch, edit_patch).view(-1)

            full_vals = full_vals.detach().cpu().numpy()
            patch_vals = patch_vals.detach().cpu().numpy()

        for meta, lpips_full, lpips_patch in zip(batch_meta, full_vals, patch_vals):
            rows.append({**meta, "lpips_full": float(lpips_full), "lpips_patch": float(lpips_patch)})

        batch_meta.clear()
        batch_clean_full.clear()
        batch_edit_full.clear()
        batch_clean_patch.clear()
        batch_edit_patch.clear()
        batch_patch_shape = None

    payloads = group_payloads_from_variants(variants)
    for payload in tqdm(payloads, total=len(payloads), desc="LPIPS"):
        clean = load_img_float01(Path(payload["clean_path"]))
        height, width = clean.shape[:2]
        for row in payload["rows"]:
            edit = load_img_float01(Path(row["edit_path"]))
            if edit.shape != clean.shape:
                continue

            bbox = clamp_bbox(row["x0"], row["y0"], row["x1"], row["y1"], height, width)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            clean_patch = clean[y0:y1, x0:x1, :]
            edit_patch = edit[y0:y1, x0:x1, :]
            patch_shape = clean_patch.shape

            if batch_patch_shape is not None and patch_shape != batch_patch_shape:
                flush()

            if batch_patch_shape is None:
                batch_patch_shape = patch_shape

            batch_meta.append(row)
            batch_clean_full.append(clean)
            batch_edit_full.append(edit)
            batch_clean_patch.append(clean_patch)
            batch_edit_patch.append(edit_patch)

            if len(batch_meta) >= batch_size:
                flush()

    flush()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path, *, overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return int(len(df))


def canonical_merge(psnr_ssim_df: pd.DataFrame, lpips_df: pd.DataFrame) -> pd.DataFrame:
    lpips_cols = CANONICAL_KEY_COLUMNS + ["lpips_full", "lpips_patch"]
    lpips_dedup = lpips_df[lpips_cols].drop_duplicates(CANONICAL_KEY_COLUMNS)
    merged = psnr_ssim_df.merge(lpips_dedup, on=CANONICAL_KEY_COLUMNS, how="left")
    return merged.sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)


def build_manifest(
    *,
    args: argparse.Namespace,
    experiment: str,
    experiment_dirname: str,
    bundle_dir: Path,
    selected_metrics: list[str],
    variants: pd.DataFrame,
    outputs: dict[str, dict],
    elapsed_seconds: float,
) -> dict:
    return {
        "schema_version": "1.0",
        "experiment": experiment,
        "experiment_dirname": experiment_dirname,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "bundle_dir": str(bundle_dir.resolve()),
        "selected_metrics": selected_metrics,
        "default_metrics": list(DEFAULT_METRICS),
        "optional_metrics": list(OPTIONAL_METRICS),
        "variant_rows": int(len(variants)),
        "inside_min_threshold": INSIDE_MIN,
        "device": "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
        "num_workers": int(args.num_workers),
        "gpu_batch_size": int(args.gpu_batch_size),
        "lpips_batch_size": int(args.lpips_batch_size),
        "max_images": None if int(args.max_images) <= 0 else int(args.max_images),
        "seed": int(args.seed),
        "elapsed_seconds": elapsed_seconds,
        "outputs": outputs,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    experiment_dirname = input_root.name
    experiment = args.experiment_name or experiment_dirname
    selected_metrics = parse_metrics(args.metrics)
    max_images = None if int(args.max_images) <= 0 else int(args.max_images)

    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)

    start_time = time.time()
    bundle_dir = experiment_bundle_dir(output_root, experiment_dirname)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants_dataframe(
        input_root,
        experiment=experiment,
        experiment_dirname=experiment_dirname,
        max_images=max_images,
    )
    if variants.empty:
        raise RuntimeError(f"No decoder-locality variants were discovered under {input_root}")

    outputs: dict[str, dict] = {}
    outputs["variants"] = {
        "path": str((bundle_dir / "variants.csv.gz").resolve()),
        "rows": write_csv(variants, bundle_dir / "variants.csv.gz", overwrite=args.overwrite),
    }

    if "token_sanity" in selected_metrics:
        token_sanity = build_token_sanity_dataframe(
            input_root,
            experiment=experiment,
            experiment_dirname=experiment_dirname,
            max_images=max_images,
        )
        outputs["token_sanity"] = {
            "path": str((bundle_dir / "token_sanity.csv.gz").resolve()),
            "rows": write_csv(token_sanity, bundle_dir / "token_sanity.csv.gz", overwrite=args.overwrite),
        }

    payloads = group_payloads_from_variants(variants)

    pixel_df = None
    if "pixel_locality" in selected_metrics:
        pixel_df = run_worker_groups(
            payloads,
            _pixel_locality_worker,
            num_workers=max(1, int(args.num_workers)),
            desc="Pixel locality",
        ).sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)
        pixel_name = f"{experiment_dirname}__pixel_metrics.csv.gz"
        outputs["pixel_locality"] = {
            "path": str((bundle_dir / pixel_name).resolve()),
            "rows": write_csv(pixel_df, bundle_dir / pixel_name, overwrite=args.overwrite),
        }

    psnr_ssim_df = None
    if "psnr_ssim" in selected_metrics:
        psnr_ssim_df = run_worker_groups(
            payloads,
            _psnr_ssim_worker,
            num_workers=max(1, int(args.num_workers)),
            desc="PSNR/SSIM",
        ).sort_values(CANONICAL_KEY_COLUMNS).reset_index(drop=True)
        outputs["psnr_ssim"] = {
            "path": str((bundle_dir / "decoder_locality_psnr_ssim_all.csv.gz").resolve()),
            "rows": write_csv(psnr_ssim_df, bundle_dir / "decoder_locality_psnr_ssim_all.csv.gz", overwrite=args.overwrite),
        }

    lpips_df = None
    if "lpips" in selected_metrics:
        lpips_df = compute_lpips_dataframe(
            variants,
            batch_size=max(1, int(args.lpips_batch_size)),
        )
        outputs["lpips"] = {
            "path": str((bundle_dir / "decoder_locality_lpips_all.csv.gz").resolve()),
            "rows": write_csv(lpips_df, bundle_dir / "decoder_locality_lpips_all.csv.gz", overwrite=args.overwrite),
        }

    if "merged_fidelity" in selected_metrics:
        if psnr_ssim_df is None or lpips_df is None:
            raise RuntimeError("merged_fidelity requires both psnr_ssim and lpips")
        merged = canonical_merge(psnr_ssim_df, lpips_df)
        outputs["merged_fidelity"] = {
            "path": str((bundle_dir / "decoder_locality_psnr_ssim_lpips_merged.csv.gz").resolve()),
            "rows": write_csv(
                merged,
                bundle_dir / "decoder_locality_psnr_ssim_lpips_merged.csv.gz",
                overwrite=args.overwrite,
            ),
        }

    manifest = build_manifest(
        args=args,
        experiment=experiment,
        experiment_dirname=experiment_dirname,
        bundle_dir=bundle_dir,
        selected_metrics=selected_metrics,
        variants=variants,
        outputs=outputs,
        elapsed_seconds=time.time() - start_time,
    )
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[done] bundle: {bundle_dir}")
    for name, info in outputs.items():
        print(f"[saved] {name}: {info['rows']:,} rows -> {info['path']}")
    print(f"[saved] manifest -> {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
