# Ordered-prefix reconstruction under distribution shift

The tested systems are **FlexTok d12-d12** and **One-D-Piece S-256**. TiTok is related architectural background, not a third measured model. Read the [offline supervisor report](../../reports/ordered-prefix/ordered-prefix-ood.html) for findings and the distinction between current and historical results.

## Reanalyse the final saved study

Use `tools/prepare_workspace.py` and run `ch09_ordered_prefix_distribution_shift.ipynb` as described in [notebook instructions](../../docs/notebooks.md). This is the verified primary entry point. The raw four CSVs and qualitative assets are on the NAS; they are not committed here.

The primary comparison selects **the same 10,000 images per domain for both systems**, 50 per each of 200 shared WNIDs. ImageNet and ImageNet-R match classes, not scenes. The One-D-Piece source pool contains 50,000/30,000 images; the notebook explicitly filters it to the FlexTok cohort. Equal token counts are neither equal bitrate nor equal compute.

The final notebook uses nine budgets `[8,16,32,64,96,128,160,208,256]`, absolute quality, truncation cost relative to the full reconstruction, shared class-bootstrap draws (5,000, seed 0), direct model contrasts, and exploratory relative-progress ratios. CSVs under `results/ordered_prefix/focused_v3` are the saved tables, not freshly generated inference.

## Generate new exports

Build the external-source containers with [setup](../../docs/setup.md). Configure a new output root and the existing model cache. For example, from the repository root:

```bash
export VQ_NAS_ROOT=/megaverse/storage/nasiri
export MANIFEST_PATH="$PWD/experiments/ordered_prefix_ood/manifests/matched_10k_part1.csv"
export OUTPUT_ROOT="$VQ_NAS_ROOT/vq-tokenizer-properties/new_runs/prefix_part1"
export CACHE_ROOT="$VQ_NAS_ROOT/ordered-prefix-ood-full/cache-nasiri"
MODEL=flextok DATASET_NAME=imagenet \
  DATASET_ROOT=/home/nasiri/datasets/shared/imagenet/val \
  GPU=0 BATCH_SIZE=24 RUNNER=run_prefix_experiment_batched.py \
  bash experiments/ordered_prefix_ood/run_job.sh
```

Repeat for `DATASET_NAME=imagenet_r` and its catalog dataset root. `run_job.sh` mounts the manifest and dataset read-only, and writes outside Git. It accepts `IMAGE_NAME`, `SEED`, `TIMESTEPS`, `GUIDANCE_SCALE`, `BATCH_SIZE` and `RUNNER`; defaults are seed 20260812, 25 steps, guidance 15. It checks required paths first. Output folders contain `per_image_metrics.csv`, `metadata.json`, and reconstruction previews. A resume requires the same manifest; use a new output directory if changing weights/settings/batching. The runner's legacy resume check does not validate every setting.

For One-D-Piece use `MODEL=one_d_piece BATCH_SIZE=8 RUNNER=run_prefix_experiment.py`. Its reconstruction is deterministic. For the historical full source pool use the original `NAS_ROOT/ordered-prefix-ood-full/manifest.csv`, not `matched_10k.csv`. New matched-only exports are a valid new analysis input but not the original larger source pool.

### Preserve the actual FlexTok groups

The recorded primary exports are a union of these disjoint cohorts in **each** domain:

| Group | Manifest | Batch size | Images per domain |
| --- | --- | --- | --- |
| part1 | `manifests/matched_10k_part1.csv` | 24 | 5,000 |
| gpu0 | `manifests/part2_shards/gpu0.csv` | 1 | 500 |
| gpu1 | `manifests/part2_shards/gpu1.csv` | 24 | 2,000 |
| gpu2 | `manifests/part2_shards/gpu2.csv` | 24 | 2,000 |
| gpu3 | `manifests/part2_shards/gpu3.csv` | 1 | 500 |

Run each manifest separately with its batch size and a separate `OUTPUT_ROOT`, using the batched runner. Merge CSVs only after validating unique `(image,prefix_tokens)` keys, all nine budgets, exact manifest coverage and WNID agreement. Retain source-group/batch columns and hashes; the original merged metadata is in `results/ordered_prefix/run_metadata/flextok`. The historical merger is in `legacy/refresh_ch09.py`; **do not run that script on the final notebook**, because it also rewrites an older notebook structure. The final merged CSVs already exist on the NAS.

FlexTok uses 25 denoising steps, guidance 15, APG/norm guidance, and the PyTorch SDPA fallback for the server GPUs. Its generator stream is derived per **batch** and reset for each prefix. Changing batch size, image order, GPU architecture, or source-group membership can change pixel outputs. These instructions preserve known settings, but do not promise bitwise reproduction on arbitrary hardware.

Both systems use RGB conversion, shorter-edge resize to 256 and center crop. LPIPS uses AlexNet on [-1,1] images; PSNR uses RGB MSE/data range 1; SSIM uses the saved scikit-image implementation. Prefix truncation follows each upstream model's decoder interface; it is not a controlled study of an isolated quantizer.

## Manifests and older analyses

The committed 10k manifests are byte copies of the server selections. They contain relative file names/class IDs, not images. To make a **new** cohort (which is not guaranteed to reproduce the historical part1/part2 split):

```bash
python experiments/ordered_prefix_ood/build_manifest.py \
  --imagenet /path/to/imagenet/val --imagenet-r /path/to/imagenet-r \
  --classes 200 --images-per-class 50 --seed 20260812 --output /path/to/new_manifest.csv
```

`--all-images` uses GNU `find -printf` and is intended for Linux/Elysium. For Sketch/ObjectNet, `build_extended_manifests.py --help` lists required roots, outputs and controls. ObjectNet uses its supplied category mappings to construct the ImageNet control; labels are not ordinary one-to-one WNID matches.

`summarize_results.py CSV... --output NEW_DIRECTORY/summary.json` uses adjacent `metadata.json` and writes the JSON summary plus four CSV tables beside it. Defaults: LPIPS primary, 32-token endpoint, 10,000 bootstrap draws, seed 20260812. This is the earlier analysis, **not** the final notebook's joint 5,000-draw analysis. It also includes monotone-envelope/interpolated budget estimates; these are not newly decoded intermediate budgets.

## Historical code: inspect before using

- `legacy/extended/`: the earlier six-budget Sketch/ObjectNet runner, launcher, manifest builder and summarizer. FlexTok guidance was **7.5**, not the current primary's 15. Historical metadata and results stay in the separate NAS extended run tree. Preserve these settings when reproducing those figures; do not relabel them as a guidance-15 replication.
- `legacy/run_flextok_part2_shard.sh`: original lab-specific shard launcher, documents the historical server mounts; use the portable `run_job.sh` for a new run.
- `legacy/benchmark/run_prefix_experiment.py`: batch-performance trial, not a canonical result source.
- `pilots/`: the original single-model prefix smoke tests. Run with their matching upstream source at `/app` in the corresponding container; inspect each script's `--help` and defaults. They are retained for provenance, not a substitute for the main manifest-controlled experiment.
- `scripts/analysis/build_ch09_matched_analysis.py SOURCE DESTINATION` and `add_ch09_reconstruction_grids.py --help`: assembly helpers for the earlier notebook revision. They depend on the original cell structure/saved preview layout, not the final focused notebook.

Saved qualitative images are part of the final notebook's inputs, with original-image hash checks. If running new inference, collect the matching qualitative assets too; do not annotate historical pixels with new metric values. The package includes their extraction/assembly helper and the archived exact inputs on NAS, rather than claiming every new export automatically reproduces the presentation grids.
