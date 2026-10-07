# Thesis experiments: regenerate exports

For inspecting the submitted results, start with the saved NAS exports and [notebooks](notebooks.md). The commands here generate **new** exports; choose a new output directory. They do not overwrite the archived runs. Install the selected model's dependencies first, using [setup](setup.md).

The thesis compares complete pretrained VQGAN and LlamaGen checkpoints, not an isolated causal intervention on their quantizers. Both use 256-pixel inputs, 16×16 token grids and 16,384 code IDs. Preprocessing is retained separately: LlamaGen's upstream center crop uses its bicubic pipeline; the VQGAN exporter uses LANCZOS resize plus center crop. Reference arrays must match the chosen export's preprocessing.

```bash
export VQ_NAS_ROOT=/megaverse/storage/nasiri
export DATASET="$VQ_NAS_ROOT/vq-tokenizer-properties/datasets/imagenet_val"
export LLAMA_CKPT="$VQ_NAS_ROOT/llamagen/checkpoints/vq_ds16_c2i.pt"
export VQGAN_CONFIG="$VQ_NAS_ROOT/vq-gan/checkpoints/vqgan_imagenet_f16_16384/model.yaml"
export VQGAN_CKPT="$VQ_NAS_ROOT/vq-gan/checkpoints/vqgan_imagenet_f16_16384/last.ckpt"
export RUN_ROOT="$VQ_NAS_ROOT/vq-tokenizer-properties/new_runs/example"
mkdir -p "$RUN_ROOT"
```

`scripts/run_export.py` adds the external model checkout to `PYTHONPATH` and runs the project's script from that checkout. Use an absolute `RUN_ROOT`: the runner's working directory is not this repository. `--dry-run` prints the command without inference (after checking the source checkout).

## RQ1 / RQ5: reconstruction and code usage

```bash
python scripts/run_export.py --model llamagen --task reconstruction -- \
  --data-path "$DATASET" --vq-ckpt "$LLAMA_CKPT" --vq-model VQ-16 \
  --sample-dir "$RUN_ROOT/reconstruction/llamagen" --batch-size 32 --seed 0
python scripts/run_export.py --model llamagen --task usage -- \
  --data-path "$DATASET" --vq-ckpt "$LLAMA_CKPT" --dataset-name imagenet \
  --outdir "$RUN_ROOT/usage/llamagen/imagenet" --save-indices
```

In the VQGAN environment:

```bash
python scripts/run_export.py --model vqgan --task reconstruction -- \
  --data-root "$DATASET" --config-path "$VQGAN_CONFIG" --model-path "$VQGAN_CKPT" \
  --outdir "$RUN_ROOT/reconstruction/vqgan" --size 256
python scripts/run_export.py --model vqgan --task usage -- \
  --data-root "$DATASET" --dataset-name imagenet --config-path "$VQGAN_CONFIG" \
  --model-path "$VQGAN_CKPT" --outdir "$RUN_ROOT/usage/vqgan/imagenet" --save-indices
```

Repeat for the prepared ImageNet-V2, Sketch, ObjectNet `images/`, OrganAMNIST, BloodMNIST and RVL-CDIP class directories from the NAS catalog. Set `--dataset-name` consistently with the notebook's folder names; inspect the notebook configuration before assembling a new analysis root. Do not substitute differently sized dataset exports into the archived analysis without updating the recorded counts.

Reconstruction runners produce uint8 image NPZs, image previews and summaries. Usage exporters produce `global_counts.npy`, `position_counts.npy`, `usage.csv`, and optionally token grids in `indices.npz` plus metadata. The analysis computes effective support/perplexity and positional entropy from these count arrays.

## Feature-space metrics and references

The thesis feature metrics use the external OpenAI evaluator. Keep it separate from the optional torch-fidelity implementation: their feature pipelines/metric values are not interchangeable.

```bash
# LlamaGen environment; REFERENCE_OUT is an absolute .npz path.
DATASET_PATH="$DATASET" REFERENCE_OUT="$RUN_ROOT/reference.npz" \
  python scripts/run_export.py --model llamagen --task reference

# Evaluator environment with TensorFlow and its upstream requirements installed.
python scripts/evaluation/evaluate_npz.py "$RUN_ROOT/reference.npz" \
  "$RUN_ROOT/reconstruction/llamagen/RECONSTRUCTION.npz" \
  --upstream third_party/guided_diffusion --output-dir "$RUN_ROOT/evaluation"
```

Replace `RECONSTRUCTION.npz` by the actual exported filename (runners print it). Use archived matching reference NPZs under the NAS references for exact thesis reanalysis. `build_ref_npz.py` is an additional LlamaGen-crop reference builder; invoke it with `PYTHONPATH="$PWD/third_party/llamagen" python scripts/evaluation/build_ref_npz.py --data-root ... --out ...`. Its rounding differs from the earlier reference helper, so do not mix them silently. NPZ arrays can require several GB of RAM; an NPZ is not memory-mapped simply by passing `mmap_mode`.

The alternative wrapper is `python scripts/evaluation/eval_fidelity.py REF.npz SAMPLE.npz --output metrics.json`; it needs torch-fidelity and is retained for provenance, not as the thesis reference evaluator.

## RQ2: global image noise

Exporters share three modes controlled by environment variables. **Global noise used σ=0.1, 0.2, 0.5** in the internal [-1,1] pixel range. Their original H1-oriented default mid level is 0.25, so override it explicitly for RQ2.

```bash
EXPERIMENT_MODE=global_noise NOISE_STD_LOW=0.1 NOISE_STD_MID=0.2 NOISE_STD_HIGH=0.5 \
  DATASET_PATH="$DATASET" VQ_CKPT="$LLAMA_CKPT" SEED=0 BATCH_SIZE=32 \
  OUTDIR="$RUN_ROOT/global_noise/llamagen" \
  python scripts/run_export.py --model llamagen --task robustness
```

VQGAN equivalent, in its own environment:

```bash
EXPERIMENT_MODE=global_noise NOISE_STD_LOW=0.1 NOISE_STD_MID=0.2 NOISE_STD_HIGH=0.5 \
  IMAGENET_VAL_ROOT="$DATASET" CONFIG_PATH="$VQGAN_CONFIG" MODEL_PATH="$VQGAN_CKPT" \
  SEED=0 BATCH_SIZE=32 OUTDIR="$RUN_ROOT/global_noise/vqgan" \
  python scripts/run_export.py --model vqgan --task robustness
```

`MAX_SAMPLES` is an optional smoke-test limit; leave it unset for a full run. Both exporters write per-image PNGs and `metadata_part_*.jsonl` with token grids and intervention information. These exports are large. Noise draws were **not matched between checkpoints** in the archived study. The same seed alone does not fix differences in sampling order or batch operations; the comparisons remain descriptive.

## RQ3: local image perturbations

Use the same commands with `EXPERIMENT_MODE=h1_patch_noise_encoder`, `PATCH_TOK_SIDE=8`, and **`NOISE_STD_MID=0.25`**. An 8×8 token patch occupies 25% of the 16×16 grid. Low/high levels remain 0.1/0.5. The scripts also export the recorded black-mask occlusion condition. Run the RQ3 notebook on the saved raw export and codebook; it computes token flip rates, distance profiles, and image-space responses.

## RQ4: local token edits

First generate codebook relations in the selected environment:

```bash
python scripts/run_export.py --model llamagen --task relations -- \
  --vq-ckpt "$LLAMA_CKPT" --out "$RUN_ROOT/llamagen_relations.npz"
python scripts/run_export.py --model vqgan --task relations -- \
  --config "$VQGAN_CONFIG" --ckpt "$VQGAN_CKPT" --out "$RUN_ROOT/vqgan_relations.npz"
```

VQGAN's helper deliberately retains the saved active-code restriction (`ALIVE_TOKEN_IDS`); it is not a generic all-code neighbor search. Preserve this for thesis reproduction and document a new choice if changing the codebook/checkpoint.

Run robustness mode `h2_patch_token_edit_decoder` with `H2_PATCH_FRACTIONS=0.10,0.25,0.50,0.75` and `CODEBOOK_RELATIONS_NPZ_PATH` pointing to the selected relation file. Keep the model/dataset/checkpoint/output variables from the previous commands. LlamaGen's relation environment variable is also `CODEBOOK_RELATIONS_NPZ_PATH` (see its source). The export replaces tokens using random-uniform, closest, farthest and orthogonal rules and records target boxes and edited token grids. Repeat for ImageNet-V2 with its dataset root.

Build the decoder-locality analysis cache for **each raw run**:

```bash
python scripts/analysis/decoder_locality_analysis.py \
  --input-root "$RUN_ROOT/h2/llamagen" --output-root "$RUN_ROOT/h2_analysis" \
  --metrics token_sanity,pixel_locality,psnr_ssim,lpips,merged_fidelity
```

Use `--help` for worker/batch options. The Chapter 7 notebook consumes merged fidelity CSVs, variant CSVs and pixel-locality summaries under the documented stable H2 archive layout. Its leakage is an outside/inside **response ratio**, not a causal claim about isolated decoder pixels.

## Thesis figure and appendix helpers

The two presentation scripts require the archived thesis `Assets/rq*/` tree, stored separately on the NAS. Make a writable output tree, link/copy `Assets` into it, then set:

```bash
export THESIS_ROOT=/path/to/writable/thesis-work
export THESIS_FIGURE_OUT="$THESIS_ROOT/Images"
export THESIS_RESULTS_OUT="$THESIS_ROOT/Chapters/app_c_supporting_results.tex"
python scripts/thesis_figures/rebuild_review_figures.py
python scripts/thesis_figures/build_supplementary_results.py
```

These regenerate presentation artifacts from aggregate CSVs and saved images; they do not produce new inference. The LaTeX thesis/summary projects remain their own repositories and are not vendored into this code handover.
