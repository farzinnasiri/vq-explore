# Thesis Draft

Working title: TBD

Status: scaffold draft. Result values are intentionally omitted for now. This file is meant to collect the thesis structure, background, experimental protocol, code references, environment details, and blank result sections in one place.

## Abstract

TODO: Write after the final experiments and narrative are fixed.

Possible direction:

This thesis studies image tokenizers used in autoregressive visual generation. Instead of evaluating tokenizers only through aggregate reconstruction quality, it probes their internal behavior: codebook usage, stability under perturbations, encoder locality, decoder locality, and behavior under dataset shift. The work compares a VQGAN tokenizer from Taming Transformers with the tokenizer used in LlamaGen.

## 1. Introduction

Autoregressive models are naturally suited to discrete sequences. In language modeling, a text sequence is already represented as a stream of tokens. Images, however, are continuous high-dimensional signals with strong two-dimensional structure. Directly modeling pixels as a sequence is computationally expensive and imposes an arbitrary ordering on spatial data.

Vector-quantized image tokenizers address this mismatch by mapping images into discrete latent codes. An encoder maps an image into a spatial grid of latent vectors, each latent vector is replaced by an entry from a learned codebook, and a decoder reconstructs the image from the corresponding codebook vectors. The resulting token grid can then be modeled by an autoregressive transformer.

This two-stage setup has become an important foundation for autoregressive image generation. However, the tokenizer is often treated as an infrastructure component and judged mostly by reconstruction metrics such as rFID, PSNR, SSIM, or LPIPS. These metrics are useful, but they do not fully explain how the tokenizer behaves internally. A tokenizer may reconstruct well while using only a small fraction of its codebook, responding non-locally to local perturbations, or producing decoder effects that spread outside the edited token region.

This thesis therefore investigates image tokenizers as objects of study in their own right. The central question is not only whether a tokenizer reconstructs images well, but how its discrete representation is organized and how stable or local that representation is under controlled interventions.

## 2. Background

### 2.1 Autoregressive Image Generation

Autoregressive generation factorizes a sequence probability as:

```text
p(x) = product_i p(x_i | x_<i)
```

For text, this factorization aligns naturally with token order. For images, the input is a two-dimensional continuous signal, so an autoregressive model requires either pixel-level sequences or a learned discrete representation. Pixel-level models are expensive because image sequences are long and local two-dimensional neighborhoods may become far apart after flattening.

Visual tokenization reduces the sequence length and creates a discrete vocabulary over visual patterns. This makes it possible to apply transformer-based next-token prediction to images.

### 2.2 Vector Quantization

The Vector Quantised Variational Autoencoder (VQ-VAE) introduced a discrete latent bottleneck for representation learning. The encoder produces continuous latent vectors, each vector is assigned to its nearest codebook entry, and the decoder reconstructs from the quantized representation. This avoids the purely continuous latent representation of standard VAEs and allows a learned prior to model the discrete latent space.

In simplified form:

```text
z_e = Encoder(x)
k = argmin_j || z_e - e_j ||_2
z_q = e_k
x_hat = Decoder(z_q)
```

Here, `e_j` is a codebook vector and `k` is the discrete token ID.

### 2.3 VQGAN and Taming Transformers

VQGAN extends vector-quantized autoencoding with perceptual and adversarial losses. The goal is to learn a codebook of perceptually meaningful visual constituents and reconstruct sharper images than pixel-loss-only autoencoders. Taming Transformers then models the resulting discrete image tokens with transformers for high-resolution image synthesis.

In this project, the VQGAN baseline is the ImageNet `f=16`, `16384`-code tokenizer from the Taming Transformers codebase. Although the nominal codebook size is 16,384, empirical code usage shows that only a much smaller subset of codes is active for the evaluated data. This affects the design of codebook-relation perturbations: for VQGAN, nearest/farthest/orthogonal replacements are computed over observed alive codes instead of the full codebook.

### 2.4 LlamaGen Tokenizer

LlamaGen revisits autoregressive image generation using Llama-style next-token prediction. It includes VQ tokenizers with downsampling ratios such as `VQ-16` and `VQ-8`, and reports strong reconstruction and generation performance. In this project, LlamaGen is compared against VQGAN as a more recent tokenizer/generator stack.

Unlike the VQGAN setup used here, LlamaGen shows near-full or full codebook usage in the evaluated setting. Therefore, codebook-relation maps for LlamaGen are computed over the full codebook.

## 3. Models Under Study

### 3.1 VQGAN

Codebase:

- `taming-transformers/`

Main tokenizer configuration:

- Model family: VQGAN
- Dataset/checkpoint: ImageNet `f=16`, `16384` codes
- Image size: `256 x 256`
- Token grid: expected `16 x 16`
- Codebook size: `16384`
- Active code subset: TODO: summarize after final code usage analysis

Key scripts:

- `taming-transformers/scripts/reconstruct_imagenet_single.py`
- `taming-transformers/scripts/vqgan_code_usage_export.py`
- `taming-transformers/scripts/vqgan_precompute_codebook_relations.py`
- `taming-transformers/scripts/vqgan_robustness_experiment_dataset_export.py`

### 3.2 LlamaGen

Codebase:

- `LlamaGen/`

Main tokenizer configuration:

- Model family: LlamaGen VQ tokenizer
- Default robustness script model: `VQ-16`
- Alternative tokenizer considered in reconstruction notes: `VQ-8`
- Image size: `256 x 256`
- Token grid for `VQ-16`: expected `16 x 16`
- Codebook size: `16384`
- Codebook embedding dimension: `8`
- Active code subset: TODO: summarize after final code usage analysis

Key scripts:

- `LlamaGen/scripts/reconstruct_imagenet.py`
- `LlamaGen/scripts/llamagen_code_usage_export.py`
- `LlamaGen/scripts/llamagen_precompute_codebook_relations.py`
- `LlamaGen/scripts/llamagen_robustness_experiment_dataset_export.py`

## 4. Datasets

### 4.1 ImageNet Validation

Primary in-distribution evaluation dataset.

Expected layout:

```text
/datasets/imagenet/val/<class_id>/<image_file>
```

Preprocessing:

- resize/crop to `256 x 256`
- convert to RGB
- normalize to model-specific input range

### 4.2 ImageNet-V2

TODO: describe source, subset/version, and directory layout.

Purpose:

- evaluate dataset shift while staying close to ImageNet semantics
- compare codebook usage and reconstruction behavior under mild distribution shift

### 4.3 ImageNet-Sketch

TODO: describe source and preprocessing.

Purpose:

- evaluate stronger distribution shift
- probe whether tokenizers trained for natural images remain stable on sketch-like inputs

### 4.4 Other Candidate Datasets

TODO: list any additional datasets that were tried or should be excluded.

## 5. Experimental Design

### 5.1 Reconstruction Baseline

Goal:

Establish baseline reconstruction quality for each tokenizer before perturbation experiments.

Procedure:

1. Load tokenizer checkpoint.
2. Encode each image into discrete tokens.
3. Decode tokens back to RGB.
4. Save reconstructions.
5. Compute image-level and distribution-level metrics.

Scripts:

- VQGAN: `taming-transformers/scripts/reconstruct_imagenet_single.py`
- LlamaGen: `LlamaGen/scripts/reconstruct_imagenet.py`

Metrics:

- MSE
- PSNR
- SSIM
- LPIPS
- rFID
- sFID
- Inception Score
- precision/recall from evaluator

Results:

TODO: insert final table.

Discussion:

TODO.

### 5.2 Codebook Usage

Goal:

Measure how much of each tokenizer's nominal codebook is actually used and how concentrated the token distribution is.

Procedure:

1. Encode all images.
2. Count token IDs globally.
3. Count token IDs per token-grid position.
4. Export usage tables and summary statistics.

Scripts:

- VQGAN: `taming-transformers/scripts/vqgan_code_usage_export.py`
- LlamaGen: `LlamaGen/scripts/llamagen_code_usage_export.py`

Exported files:

- `global_counts.npy`
- `position_counts.npy`
- `usage.csv`
- `top_codes.csv`
- `summary.json`
- optional `indices.npz`

Metrics:

- active code count
- dead code count
- active fraction
- entropy
- perplexity
- top-10 mass
- top-100 mass
- top-500 mass
- positional entropy

Results:

TODO: insert final table and plots.

Discussion:

TODO: include VQGAN alive-code subset and LlamaGen full-codebook usage.

### 5.3 Global Noise Robustness

Goal:

Measure how stable each tokenizer is when additive Gaussian noise is applied to the entire image.

Perturbation:

```text
x' = clip(x + sigma * epsilon)
epsilon ~ N(0, I)
```

The robustness exporters use the model input space `[-1, 1]`.

Noise levels:

- low: `0.1`
- mid: `0.25`
- high: `0.5`
- xhigh: `1.0`

Script mode:

```text
EXPERIMENT_MODE=global_noise
```

Measured quantities:

- reconstruction degradation
- token flip rate
- token distribution shift
- qualitative changes in reconstruction

Results:

TODO.

Discussion:

TODO.

### 5.4 H1: Encoder Locality Under Patch Noise

Goal:

Test whether local changes in image space produce local changes in token space.

Perturbation:

1. Sample a square token-aligned patch.
2. Convert token patch to the corresponding pixel-space box.
3. Add Gaussian noise only inside that image patch.
4. Re-encode the image.
5. Compare clean and perturbed token grids.

Script mode:

```text
EXPERIMENT_MODE=h1_patch_noise_encoder
```

Patch configuration:

- token patch side: `PATCH_TOK_SIDE=8` by default
- placement: `random` or `center`
- optional black-mask occlusion experiment

Measured quantities:

- token flip fraction inside patch
- token flip fraction outside patch
- leakage ratio
- embedding-distance change between clean and perturbed tokens
- distance-to-patch response curve
- boundary vs far-field leakage

Notebook analysis:

- `vq-explore/notebooks/vq-encoder-locality.ipynb`
- `vq-explore/notebooks/vq-encoder-locality-updated.ipynb`

Results:

TODO.

Discussion:

TODO.

### 5.5 H2: Decoder Locality Under Token Patch Edits

Goal:

Test whether local edits in token space produce local changes in decoded image space.

Procedure:

1. Encode image into a token grid.
2. Choose a contiguous token patch.
3. Replace token IDs inside the patch.
4. Decode the edited token grid.
5. Compare the clean and edited reconstructions.

Script mode:

```text
EXPERIMENT_MODE=h2_patch_token_edit_decoder
```

Patch sweep:

- `10%`
- `25%`
- `50%`
- `75%`

Token edit modes:

- `random_uniform`: replace patch tokens with random code IDs
- `closest`: replace each token with its nearest codebook neighbor
- `farthest`: replace each token with the most distant codebook entry
- `orthogonal`: replace each token with a code whose embedding has near-zero cosine similarity

Measured quantities:

- inside-patch image change
- outside-patch image change
- leakage ratio
- PSNR full image
- PSNR patch
- SSIM full image
- SSIM patch
- LPIPS full image
- LPIPS patch

Analysis script:

- `vq-explore/scripts/decoder_locality_analysis.py`

Notebook analysis:

- `vq-explore/notebooks/vq-decoder-locality.ipynb`
- `vq-explore/notebooks/vq-decoder-locality-v2.ipynb`

Results:

TODO.

Discussion:

TODO.

### 5.6 Codebook Relation Precomputation

Goal:

Support structured token replacement for the decoder-locality experiments.

For each code, precompute:

- nearest neighbor by Euclidean distance
- farthest neighbor by Euclidean distance
- most orthogonal code by cosine similarity

Scripts:

- VQGAN: `taming-transformers/scripts/vqgan_precompute_codebook_relations.py`
- LlamaGen: `LlamaGen/scripts/llamagen_precompute_codebook_relations.py`

Important implementation detail:

VQGAN relation maps are computed over the empirically alive token set, because most of the nominal codebook is dead in the evaluated setting. LlamaGen relation maps are computed over the full codebook, because the tokenizer uses all or nearly all codes.

Exported files:

- `vqgan_codebook_relations.npz`
- `llamagen_codebook_relations.npz`

Stored arrays:

- `alive_token_ids`
- `min_dist_idx`
- `max_dist_idx`
- `ortho_idx`
- `min_dist_val`
- `max_dist_val`
- `ortho_cos_val`
- `n_embed`
- `embed_dim`

Results:

TODO.

## 6. Metrics

### 6.1 MSE and PSNR

Mean squared error measures pixel-level reconstruction error. PSNR expresses the same error on a logarithmic decibel scale. Higher PSNR indicates lower pixel error.

Use:

- reconstruction baseline
- global robustness
- decoder-locality clean-vs-edited comparison

### 6.2 SSIM

SSIM measures structural similarity between two images. It is more perceptual than raw pixel error but still operates as a full-reference image similarity metric.

Use:

- reconstruction baseline
- decoder-locality fidelity

### 6.3 LPIPS

LPIPS uses deep network features to estimate perceptual similarity. Lower LPIPS means the compared images are perceptually closer according to the feature metric.

Use:

- reconstruction baseline
- decoder-locality fidelity

### 6.4 FID and rFID

FID compares feature distributions of generated and real images using Inception features. In this project, reconstruction FID compares reconstructed images against the corresponding dataset reference distribution.

Use:

- reconstruction baseline
- cross-dataset reconstruction comparison

### 6.5 sFID

sFID is a spatial variant of FID used by the evaluator stack. It is included as an additional distribution-level metric for reconstruction quality.

Use:

- reconstruction baseline

### 6.6 Inception Score

Inception Score measures a combination of image recognizability and sample diversity using a pretrained classifier.

Use:

- evaluator output for reconstruction NPZs

### 6.7 Codebook Usage Metrics

Definitions:

- active code: code ID observed at least once
- dead code: code ID never observed
- entropy: uncertainty of the empirical token distribution
- perplexity: effective number of codes used, computed from entropy
- top-k mass: fraction of all token assignments covered by the k most frequent codes
- positional entropy: entropy of token distribution at each spatial token-grid location

Use:

- compare nominal and effective vocabulary size
- diagnose codebook collapse or concentration
- compare in-distribution and OOD usage

### 6.8 Locality Metrics

Encoder locality:

- inside token change
- outside token change
- leakage ratio
- distance-to-patch token flip probability
- distance-to-patch embedding-distance response

Decoder locality:

- inside image change
- outside image change
- leakage ratio
- full-image and patch-level PSNR/SSIM/LPIPS

## 7. Implementation and Code Organization

### 7.1 Repositories

This project uses three local code areas:

```text
taming-transformers/   VQGAN / Taming Transformers codebase plus custom scripts
LlamaGen/              LlamaGen codebase plus custom scripts
vq-explore/            notes, notebooks, analysis scripts, thesis draft
```

### 7.2 Custom Experiment Scripts

Reconstruction:

- `taming-transformers/scripts/reconstruct_imagenet_single.py`
- `LlamaGen/scripts/reconstruct_imagenet.py`

Code usage:

- `taming-transformers/scripts/vqgan_code_usage_export.py`
- `LlamaGen/scripts/llamagen_code_usage_export.py`

Codebook geometry:

- `taming-transformers/scripts/vqgan_precompute_codebook_relations.py`
- `LlamaGen/scripts/llamagen_precompute_codebook_relations.py`

Robustness export:

- `taming-transformers/scripts/vqgan_robustness_experiment_dataset_export.py`
- `LlamaGen/scripts/llamagen_robustness_experiment_dataset_export.py`

Post-hoc decoder locality analysis:

- `vq-explore/scripts/decoder_locality_analysis.py`

### 7.3 Notebook Analysis

Current notebooks:

- `vq-explore/notebooks/vq-code-usage-analysis.ipynb`
- `vq-explore/notebooks/vq-global-robustness-analysis.ipynb`
- `vq-explore/notebooks/vq-encoder-locality.ipynb`
- `vq-explore/notebooks/vq-encoder-locality-updated.ipynb`
- `vq-explore/notebooks/vq-decoder-locality.ipynb`
- `vq-explore/notebooks/vq-decoder-locality-v2.ipynb`
- `vq-explore/notebooks/vq-end-to-end-reconstruction-comparison.ipynb`

TODO: clean notebooks or replace selected analyses with reproducible scripts before final thesis submission.

## 8. Computational Environment

### 8.1 VQGAN / Taming Transformers Container

Dockerfile:

- `taming-transformers/docker/Dockerfile`

Base image:

- `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04`

Python:

- Python `3.8`

Main PyTorch stack:

- `torch==1.13.1+cu117`
- `torchvision==0.14.1+cu117`

Main Python dependencies:

- `numpy==1.19.2`
- `albumentations==0.4.3`
- `opencv-python==4.1.2.30`
- `pytorch-lightning==1.0.8`
- `omegaconf==2.0.0`
- `einops==0.3.0`
- `transformers==4.3.1`
- `scikit-image==0.18.3`

APT dependencies:

- `ffmpeg`
- `libsm6`
- `libxext6`

Build command:

```bash
cd taming-transformers/docker
./build.sh -- -t nasiri/taming:latest .
```

Example run pattern:

```bash
docker run --rm -it \
  --gpus "device=0" \
  --shm-size=10g \
  -v $(pwd):/app \
  -v $(pwd)/../checkpoints:/checkpoints \
  -v /home/nasiri/datasets/shared/imagenet:/datasets/imagenet \
  -w /app \
  nasiri/taming:latest \
  python3 scripts/vqgan_robustness_experiment_dataset_export.py
```

### 8.2 LlamaGen Container

Dockerfile:

- `LlamaGen/docker/Dockerfile`

Base image:

- `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04`

Main PyTorch stack:

- `torch==2.1.2`
- `torchvision==0.16.2`
- CUDA wheels from `cu121`

Main Python dependencies:

- `numpy`
- `Pillow`
- `scikit-image`
- `omegaconf`
- `einops`
- `tqdm`
- `huggingface_hub`
- `jupyter`
- `nbconvert`
- `matplotlib`
- `pandas`
- `lpips`

APT dependencies:

- `libgl1`
- `libglib2.0-0`
- `ffmpeg`
- `libsm6`
- `libxext6`

Build command:

```bash
cd LlamaGen/docker
./build.sh -- -t nasiri/llamagen:latest .
```

Example run pattern:

```bash
docker run --rm -it \
  --gpus "device=0" \
  --shm-size=8g \
  -v $(pwd):/app \
  -v $(pwd)/../checkpoints:/checkpoints \
  -v /home/nasiri/datasets/shared/imagenet:/datasets/imagenet \
  -w /app \
  nasiri/llamagen:latest \
  python3 scripts/reconstruct_imagenet.py
```

### 8.3 Evaluator Container

Dockerfile:

- `taming-transformers/eval/Dockerfile`

Purpose:

- compute distribution-level metrics from reconstruction NPZ files
- support TensorFlow evaluator and torch-fidelity evaluator

Main dependencies:

- `tensorflow-cpu==2.9.0`
- `numpy==1.23.5`
- `scipy==1.10.1`
- `torch`
- `torchvision`
- `torch-fidelity`

Build command:

```bash
cd taming-transformers
docker build -t nasiri/evaluator:latest -f eval/Dockerfile .
```

Example run pattern:

```bash
docker run --rm -it \
  -v /srv/npz:/data \
  -w /data \
  nasiri/evaluator:latest \
  python /work/evaluator.py /data/reference.npz /data/reconstructions.npz
```

## 9. Results

This section is intentionally blank until final runs are selected.

### 9.1 Reconstruction Results

TODO: final table.

### 9.2 Codebook Usage Results

TODO: final table and plots.

### 9.3 Global Noise Robustness Results

TODO.

### 9.4 Encoder Locality Results

TODO.

### 9.5 Decoder Locality Results

TODO.

### 9.6 OOD Results

TODO.

## 10. Discussion

TODO.

Possible discussion themes:

- aggregate reconstruction metrics do not fully characterize tokenizer behavior
- nominal codebook size can differ sharply from effective codebook size
- local image perturbations can produce non-local token changes
- token-space edits can produce visible decoder spillover outside the edited patch
- codebook geometry perturbations may reveal whether embedding-space neighbors correspond to visually gentle edits
- OOD images may expose tokenizer assumptions that are hidden on ImageNet

## 11. Limitations

TODO.

Candidate limitations:

- some analyses are exploratory and need reruns with fixed protocols
- notebook outputs are not yet cleanly reproducible
- some result paths point to external experiment directories
- VQGAN and LlamaGen use different training objectives and codebook behavior, so comparisons should be interpreted carefully
- dead-code handling differs between the models for methodological reasons

## 12. Conclusion

TODO.

## References

- Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. "Neural Discrete Representation Learning." NeurIPS 2017. https://arxiv.org/abs/1711.00937
- Patrick Esser, Robin Rombach, Björn Ommer. "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021. https://arxiv.org/abs/2012.09841
- Taming Transformers codebase. https://github.com/CompVis/taming-transformers
- Aditya Ramesh et al. "Zero-Shot Text-to-Image Generation." 2021. https://arxiv.org/abs/2102.12092
- Peize Sun et al. "Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation." 2024. https://arxiv.org/abs/2406.06525
- LlamaGen codebase. https://github.com/FoundationVision/LlamaGen
- Richard Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018. https://arxiv.org/abs/1801.03924
- Martin Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017. https://arxiv.org/abs/1706.08500
- Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. "Image Quality Assessment: From Error Visibility to Structural Similarity." IEEE Transactions on Image Processing, 2004. https://live.ece.utexas.edu/publications/2004/zwang_ssim_ieeeip2004.pdf
- OpenAI guided-diffusion evaluator repository. https://github.com/openai/guided-diffusion
