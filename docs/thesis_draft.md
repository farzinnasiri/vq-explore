# Thesis Draft

Working title: TBD

Status: scaffold draft. Result values are intentionally omitted for now. This file collects the thesis structure, background, experimental protocol, code references, environment details, and blank result sections in one place.

## Abstract

TODO: Write after the final experiments and narrative are fixed.

Possible direction:

This thesis studies image tokenizers used in autoregressive visual generation. Instead of evaluating tokenizers only through aggregate reconstruction quality, it probes their internal behavior: codebook usage, stability under perturbations, encoder locality, decoder locality, and behavior under dataset shift. The work compares a VQGAN tokenizer from Taming Transformers with the tokenizer used in LlamaGen.

## 1. Introduction

Autoregressive models are naturally suited to discrete sequences. In language modeling, a text sequence is already represented as a stream of tokens. Images, however, are continuous high-dimensional signals with strong two-dimensional structure. Directly modeling pixels as a sequence is computationally expensive and imposes an arbitrary ordering on spatial data.

Vector-quantized image tokenizers address this mismatch by mapping images into discrete latent codes. An encoder maps an image into a spatial grid of latent vectors, each latent vector is replaced by an entry from a learned codebook, and a decoder reconstructs the image from the corresponding codebook vectors. The resulting token grid can then be modeled by an autoregressive transformer.

This two-stage setup has become an important foundation for autoregressive image generation. However, the tokenizer is often treated as an infrastructure component and judged mostly by reconstruction metrics such as rFID, PSNR, SSIM, or LPIPS. These metrics are useful, but they do not fully explain how the tokenizer behaves internally. A tokenizer may reconstruct well while using only a small fraction of its codebook, responding non-locally to local perturbations, or producing decoder effects that spread outside the edited token region.

This thesis therefore investigates image tokenizers as objects of study in their own right. The central question is not only whether a tokenizer reconstructs images well, but how its discrete representation is organized and how stable or local that representation is under controlled interventions.

### 1.1 Research Questions

TODO: refine wording after the final experiment set is fixed.

- RQ1: How do the tokenizers compare in reconstruction quality and effective codebook usage?
- RQ2: How robust are the tokenizers to global image perturbations?
- RQ3: Do local perturbations in image space produce local or non-local changes in token space?
- RQ4: Do local perturbations in token space produce local or non-local changes in reconstructed image space?
- RQ5: How do reconstruction quality and codebook usage change under distribution shift?

### 1.2 Contributions

TODO: write after final results.

## 2. Background

This chapter gives the technical context for the experiments in this thesis. The focus is deliberately narrow: it covers the line of work in which images are first compressed into discrete visual tokens and then modeled as sequences. This is the setting shared by VQ-VAE (van den Oord et al., 2017), VQGAN/Taming Transformers (Esser et al., 2021), DALL-E-style discrete autoencoders (Ramesh et al., 2021), and LlamaGen (Sun et al., 2024). Diffusion models are not reviewed in detail because the experiments do not compare generative sampling methods; they compare the behavior of the tokenizers that make autoregressive image generation possible.

### 2.1 Autoregressive Models: From Text to Images

Autoregressive generation factorizes a sequence probability as:

$$
p(x) = \prod_{i=1}^{n} p(x_i \mid x_{1:i-1})
$$

For text, this factorization aligns naturally with the representation used by the model. A sentence is already a sequence of discrete symbols after tokenization, and next-token prediction can be trained directly on that sequence. Images have a less convenient structure. A color image is usually represented as a dense tensor of continuous pixel intensities with shape `height x width x channels`. The signal is two-dimensional, local neighborhoods are important, and the number of raw values is large: a `256 x 256` RGB image contains `196,608` scalar pixel values before any modeling context is considered.

An autoregressive image model must therefore make two design choices before it can use a standard sequence model. First, it must choose a sequence order for a two-dimensional object, usually by flattening a spatial grid. Second, it must choose the unit of prediction. Pixel-level autoregressive models predict raw or discretized pixel values, but their sequences are long and the resulting model must learn both low-level image statistics and long-range structure. Modern autoregressive image generators often avoid this by introducing a learned image tokenizer. The tokenizer compresses an image into a shorter grid of discrete codes; the transformer then models this grid as a sequence.

In this two-stage design, the first stage is an autoencoder-like tokenizer and the second stage is an autoregressive prior. The tokenizer determines the visual vocabulary and the spatial resolution of the sequence. For example, a tokenizer with downsampling factor `f = 16` maps a `256 x 256` image to a `16 x 16` grid, which contains only `256` image tokens. The autoregressive model then predicts a sequence of 256 discrete token IDs rather than hundreds of thousands of pixel-channel values.

This compression is not merely an engineering convenience. It changes the learning problem. The transformer no longer models pixels directly; it models the distribution of learned visual parts. The quality and behavior of the tokenizer therefore constrain the whole generative system. If the tokenizer loses information, uses its codebook unevenly, or reacts unstably to small input changes, those properties become part of the data distribution seen by the autoregressive model.

> **Figure placeholder 2.1: Two-stage autoregressive image generation.**
> Add a diagram showing: input image -> encoder -> token grid -> flattened token sequence -> autoregressive transformer -> generated token sequence -> decoder -> output image. The figure should highlight the reduction from `256 x 256 x 3` pixels to a `16 x 16` token grid for the `f = 16` setting used in this project.

This framing motivates the central choice in the thesis: rather than evaluating only final generated images, the experiments study the tokenizer itself. Reconstruction metrics test whether the autoencoder preserves images on average, but they do not fully explain how the representation is organized. The following sections review the tokenizer mechanisms that matter for the later research questions.

### 2.2 Image Tokens and Vector Quantization

The Vector Quantised Variational Autoencoder (VQ-VAE) introduced a discrete latent bottleneck for representation learning (van den Oord et al., 2017). The encoder produces continuous latent vectors, each vector is assigned to its nearest codebook entry, and the decoder reconstructs from the quantized representation. This avoids the purely continuous latent representation of standard VAEs and allows a learned prior to model the discrete latent space.

In simplified form:

$$
z_e = Encoder(x)
$$

$$
k = \arg\min_j \lVert z_e - e_j \rVert_2
$$

$$
z_q = e_k
$$

$$
\hat{x} = Decoder(z_q)
$$

Here, `e_j` is a codebook vector and `k` is the discrete token ID.

More explicitly, an encoder maps an image `x` to a latent tensor `z_e(x)` with spatial dimensions `h x w` and channel dimension `d`. Each vector `z_e(x)_{u,v}` is replaced by the nearest entry in a learned codebook `E = {e_1, ..., e_K}`. The result is both a quantized latent tensor `z_q` and an integer token grid `k`:

$$
k_{u,v} = \arg\min_j \lVert z_e(x)_{u,v} - e_j \rVert_2
$$

$$
z_q(x)_{u,v} = e_{k_{u,v}}
$$

The decoder receives `z_q`, not the original continuous encoder output. This forces image information through a finite vocabulary. The codebook size `K` controls the nominal number of possible visual symbols, while the spatial grid size controls how many symbols represent one image.

Training a VQ model requires handling the non-differentiable nearest-neighbor assignment. VQ-VAE uses a straight-through estimator so gradients from the decoder can update the encoder even though the forward pass uses discrete code assignments. The loss also includes a codebook term and a commitment term, encouraging codebook vectors to move toward encoder outputs and encouraging encoder outputs not to fluctuate arbitrarily around the codebook. Following the VQ-VAE formulation, the usual objective can be summarized as:

$$
L = L_{\mathrm{rec}} + \lVert \operatorname{sg}[z_e(x)] - e \rVert_2^2 + \beta \lVert z_e(x) - \operatorname{sg}[e] \rVert_2^2
$$

where `sg[.]` denotes stop-gradient and `beta` controls the commitment penalty. The exact reconstruction loss depends on the model family. Early VQ-VAE work used reconstruction objectives suitable for representation learning and likelihood modeling. VQGAN later added perceptual and adversarial components to improve image sharpness.

The token grid can be interpreted in two equivalent ways. For the autoencoder, it is an index map telling the decoder which codebook vector to use at each spatial location. For the autoregressive prior, it is a sequence over a vocabulary of size `K`. This dual role is why codebook behavior matters: a code that is useful for reconstruction is also a symbol the transformer may need to predict.

Two practical properties are especially important for this thesis.

First, the nominal vocabulary and the empirical vocabulary can differ. A model may have `K = 16,384` entries, but only a subset may appear on a dataset. This can happen because some codes are unused after training or because the evaluated data occupies only part of the learned visual space. The distinction between nominal and active codebook size is central to RQ1 and RQ5.

Second, the quantization boundary can create discontinuities. A small image perturbation may move an encoder output across a nearest-neighbor boundary and change the token ID. This change can be local if only the affected spatial region crosses a boundary, or it can be non-local if the encoder architecture spreads information across a wider area. This motivates the encoder-locality experiment in RQ3.

> **Figure placeholder 2.2: Vector quantization in latent space.**
> Add a schematic with continuous encoder vectors as points, codebook entries as centroids, nearest-neighbor regions as Voronoi cells, and a highlighted point crossing a boundary after a small perturbation. The caption should connect this boundary crossing to token flips in RQ2 and RQ3.

### 2.3 VQGAN and Taming Transformers

VQGAN extends vector-quantized autoencoding with perceptual and adversarial losses (Esser et al., 2021). The goal is to learn a codebook of perceptually meaningful visual constituents and reconstruct sharper images than pixel-loss-only autoencoders. Taming Transformers then models the resulting discrete image tokens with transformers for high-resolution image synthesis.

The main limitation of a pixel-loss autoencoder is that pixel fidelity and perceptual fidelity are not the same objective. Minimizing an average pixel error can encourage blurry reconstructions when several plausible high-frequency explanations exist. VQGAN addresses this by combining vector quantization with a perceptual reconstruction loss and an adversarial discriminator. The perceptual term compares images in a learned feature space, while the adversarial term encourages reconstructions to lie on the natural-image manifold. Taming Transformers uses this improved first-stage model as the tokenizer for transformer-based high-resolution synthesis (Esser et al., 2021).

Architecturally, the VQGAN tokenizer used in this work follows the standard first-stage design from Taming Transformers. An image is passed through a convolutional encoder, projected into the codebook embedding space, quantized by nearest-neighbor lookup, projected back into the decoder latent space, and reconstructed by a convolutional decoder. This produces a sequence of transformations from image space to continuous latent space, from continuous latent space to discrete codebook indices, and finally back to image space.

This separation between encoding, quantization, and decoding is important experimentally because it makes it possible to intervene at different points in the tokenizer. Perturbations applied before encoding test the stability of the encoder and quantizer, while perturbations applied directly to token IDs test the spatial behavior of the decoder.

The VQGAN tokenizer used in this thesis is the ImageNet `f = 16`, `16,384`-code first-stage model from Taming Transformers. Its main configuration is:

- input resolution: `256`
- channels: `3`
- latent channels: `256`
- embedding dimension: `256`
- number of codebook entries: `16,384`
- channel multipliers: `[1, 1, 2, 2, 4]`
- attention at resolution `16`
- loss family: perceptual reconstruction loss with adversarial training

The downsampling factor `f = 16` means that a `256 x 256` image is represented by a `16 x 16` token grid. Each image is therefore reduced to 256 codebook indices. In the original Taming Transformers model table, the ImageNet `f = 16`, `16,384` first-stage model is reported with reconstruction FID values for train and validation reconstructions, which is why it is a natural baseline for this thesis.

One important methodological detail is that the nominal VQGAN codebook can be much larger than the empirically active set. This matters because a token replacement experiment should not silently replace a valid token with a dead code that the model rarely or never uses in the evaluated setting. For VQGAN, the decoder-locality experiments therefore define nearest, farthest, and orthogonal replacements over the observed active subset rather than over the full nominal vocabulary.

> **Figure placeholder 2.3: VQGAN first-stage model.**
> Add an architecture diagram for the VQGAN first-stage path: encoder, projection into codebook space, vector quantization, projection into decoder space, and decoder. Annotate the project settings: `256 x 256` input, `16 x 16` token grid, `K = 16,384`, `embed_dim = 256`.

The literature usually evaluates this first stage by reconstruction quality, especially rFID, LPIPS, or related perceptual metrics. Those metrics are necessary but incomplete for this thesis. A tokenizer can achieve strong average reconstruction while still having representational properties that are important for downstream autoregressive modeling: concentrated code usage, unstable token assignments, or decoder changes outside an edited token patch. The VQGAN baseline is therefore used both as a reconstruction model and as a system whose internal discrete representation can be probed.

### 2.4 LlamaGen Tokenizer

LlamaGen revisits autoregressive image generation using Llama-style next-token prediction (Sun et al., 2024). It includes VQ tokenizers with downsampling ratios such as `VQ-16` and `VQ-8`, and reports strong reconstruction and generation performance. In this project, LlamaGen is compared against VQGAN as a more recent tokenizer/generator stack.

The LlamaGen paper asks whether the next-token prediction paradigm used by large language models can scale to image generation when images are represented as discrete tokens. Its system keeps the broad two-stage structure: an image tokenizer maps images to token grids, and a Llama-style autoregressive model predicts those tokens (Sun et al., 2024). The released tokenizers include downsampling ratios `16` and `8`, corresponding to `16 x 16` and `32 x 32` token grids for `256 x 256` images. In the class-conditional ImageNet setting, the reported rFID values include `2.19` for the `VQ-16` tokenizer at `16 x 16` tokens and `0.59` for the `VQ-8` tokenizer at `32 x 32` tokens.

The LlamaGen tokenizer is structurally similar to the VQGAN first stage but uses a smaller codebook embedding dimension by default. The tokenizer configuration used in this work has the following main properties:

- codebook size: `16,384`
- codebook embedding dimension: `8`
- L2-normalized codebook: enabled
- commitment loss beta: `0.25`
- encoder and decoder channel multipliers for `VQ-16`: `[1, 1, 2, 2, 4]`
- encoder and decoder channel multipliers for `VQ-8`: `[1, 2, 2, 4]`

Conceptually, it supports the same experimental decomposition as the VQGAN first stage: images can be encoded into quantized latent representations, quantized latents can be decoded back to images, and token IDs can be decoded after direct token-space intervention.

The vector quantizer computes nearest-neighbor assignments in codebook space. When L2 normalization is enabled, both encoder outputs and codebook embeddings are normalized before distance computation. This makes code identity depend on angular similarity as well as Euclidean distance after normalization, which is relevant when interpreting nearest-neighbor token substitutions.

For this thesis, the main LlamaGen comparison uses `VQ-16`. This makes the spatial token grid directly comparable to the VQGAN `f = 16` baseline: both map a `256 x 256` image to `16 x 16` tokens with a nominal vocabulary of 16,384 codes. The two tokenizers differ, however, in codebook dimensionality, training recipe, and empirical code usage. Those differences are useful rather than incidental. They allow the experiments to separate properties that are common to discrete image tokenizers from properties that depend on a particular tokenizer design.

Unlike the VQGAN setup used here, LlamaGen shows near-full or full codebook usage in the evaluated setting. Therefore, codebook-relation maps for LlamaGen are computed over the full codebook. This difference is carried into RQ4: a nearest-neighbor replacement in VQGAN is nearest among observed alive codes, while the corresponding LlamaGen replacement is nearest in the full normalized codebook.

> **Figure placeholder 2.4: VQGAN and LlamaGen tokenizer comparison.**
> Add a compact table or diagram comparing the two first-stage tokenizers used in this thesis: grid size, nominal codebook size, embedding dimension, normalization, and codebook-relation domain.

### 2.5 From Reconstruction Quality to Tokenizer Behavior

The common evaluation of image tokenizers begins with reconstruction. Given an input image `x`, the tokenizer produces `x_hat = D(Q(E(x)))`. The reconstruction can be compared against `x` using full-reference metrics such as MSE, PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018), and the distribution of reconstructions can be compared with the distribution of real images using FID or rFID (Heusel et al., 2017). These metrics answer whether the tokenizer preserves the image well enough for downstream use.

However, reconstruction quality alone does not determine whether a tokenizer is a good discrete representation for autoregressive modeling. A tokenizer also defines a vocabulary, a token distribution, and a spatial response pattern. The same rFID can hide different internal behaviors:

- one tokenizer may use most of its codebook, while another uses a small active subset;
- one tokenizer may keep local input changes local in token space, while another produces distant token flips;
- one decoder may confine token edits to the edited region, while another spreads changes across the image;
- one tokenizer may preserve its codebook statistics under dataset shift, while another may concentrate on a smaller set of tokens.

These behaviors are not secondary implementation details. They affect the sequence model trained on top of the tokenizer. Codebook concentration changes the effective vocabulary and token frequency distribution. Non-local encoder responses can make token sequences sensitive to small image changes. Non-local decoder responses can make local token prediction errors visually global. Distribution-shift changes can expose whether a tokenizer has learned a general visual vocabulary or a representation specialized to the training distribution.

The experiments in this thesis are designed around this broader view of tokenizer quality. RQ1 establishes reconstruction and codebook usage. RQ2 measures global stability under additive noise. RQ3 tests encoder locality by perturbing local image patches and measuring token changes. RQ4 tests decoder locality by editing local token patches and measuring image-space changes. RQ5 asks whether reconstruction and codebook behavior remain stable under distribution shift.

> **Figure placeholder 2.5: Taxonomy of tokenizer probes.**
> Add a four-panel explanatory figure: reconstruction baseline, global noise, local image-space perturbation through the encoder, and local token-space perturbation through the decoder. Each panel should show the measured object: image metrics, token flips, leakage outside the patch, or codebook usage.

This thesis therefore treats the tokenizer as a measurable object rather than a black-box preprocessing step. The goal is not to claim that one tokenizer is universally better, but to identify which properties are visible only when the discrete representation is probed directly.

## 3. Datasets, Models, and Metrics

This chapter defines the shared experimental material used across the research-question chapters. Each RQ chapter contains its own specific procedure, measured quantities, results placeholder, and discussion placeholder.

### 3.1 Models Under Study

#### 3.1.1 VQGAN

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

#### 3.1.2 LlamaGen

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

### 3.2 Datasets

#### 3.2.1 ImageNet Validation

Primary in-distribution evaluation dataset.

Expected layout:

```text
/datasets/imagenet/val/<class_id>/<image_file>
```

Preprocessing:

- resize/crop to `256 x 256`
- convert to RGB
- normalize to model-specific input range

#### 3.2.2 ImageNet-V2

TODO: describe source, subset/version, and directory layout.

Purpose:

- evaluate dataset shift while staying close to ImageNet semantics
- compare codebook usage and reconstruction behavior under mild distribution shift

#### 3.2.3 ImageNet-Sketch

TODO: describe source and preprocessing.

Purpose:

- evaluate stronger distribution shift
- probe whether tokenizers trained for natural images remain stable on sketch-like inputs

#### 3.2.4 Other Candidate Datasets

TODO: list any additional datasets that were tried or should be excluded.

### 3.3 Metrics

#### 3.3.1 Reconstruction Metrics

MSE and PSNR measure pixel-level reconstruction error. PSNR expresses the same error on a logarithmic decibel scale. Higher PSNR indicates lower pixel error.

SSIM measures structural similarity between two images. It is more perceptual than raw pixel error but still operates as a full-reference image similarity metric.

LPIPS uses deep network features to estimate perceptual similarity. Lower LPIPS means the compared images are perceptually closer according to the feature metric.

FID compares feature distributions of generated and real images using Inception features. In this project, reconstruction FID compares reconstructed images against the corresponding dataset reference distribution. sFID is a spatial variant of FID used by the evaluator stack. Inception Score measures a combination of image recognizability and sample diversity using a pretrained classifier.

Use:

- reconstruction baseline
- global robustness
- decoder-locality clean-vs-edited comparison
- cross-dataset reconstruction comparison

#### 3.3.2 Codebook Usage Metrics

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

#### 3.3.3 Locality Metrics

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

## 4. RQ1: Reconstruction Baseline and Codebook Usage

### 4.1 Motivation

Before testing robustness or locality, the tokenizers need a common baseline. This chapter establishes reconstruction quality and measures how much of each nominal codebook is actually used.

### 4.2 Reconstruction Baseline

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

### 4.3 Codebook Usage

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

### 4.4 Experimental Results

TODO: insert final reconstruction table.

TODO: insert final codebook usage table and plots.

### 4.5 Discussion

TODO: include VQGAN alive-code subset and LlamaGen full-codebook usage.

## 5. RQ2: Robustness to Global Perturbations

### 5.1 Motivation

This chapter tests whether small image-wide input changes produce stable token grids and reconstructions. Unlike local perturbation experiments, global noise does not test locality; it tests general stability under distributed image degradation.

### 5.2 Experimental Setup

Goal:

Measure how stable each tokenizer is when additive Gaussian noise is applied to the entire image.

Perturbation:

$$
x' = clip(x + \sigma \epsilon)
$$

$$
\epsilon \sim N(0, I)
$$

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

Notebook analysis:

- `vq-explore/notebooks/vq-global-robustness-analysis.ipynb`

### 5.3 Experimental Results

TODO.

### 5.4 Discussion

TODO.

## 6. RQ3: Local Perturbations in Image Space

### 6.1 Motivation

This chapter isolates the encoder side of the tokenizer. If the image perturbation is spatially local, the encoded token changes may also be local, or they may leak into distant token positions because of encoder receptive fields, normalization, or quantization boundaries.

### 6.2 Experimental Setup

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

### 6.3 Experimental Results

TODO.

### 6.4 Discussion

TODO.

## 7. RQ4: Perturbations in Token Space

### 7.1 Motivation

This chapter isolates the decoder side of the tokenizer. By editing a local patch in the token grid and decoding the result, the experiment tests whether token-space changes remain spatially local in image space.

### 7.2 Experimental Setup

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

### 7.3 Codebook Relation Precomputation

Structured token replacement uses precomputed codebook relations.

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

### 7.4 Experimental Results

TODO.

### 7.5 Discussion

TODO.

## 8. RQ5: Robustness to Distribution Shift

### 8.1 Motivation

This chapter studies whether tokenizer behavior changes when inputs move away from the ImageNet validation distribution. The main concern is not only aggregate reconstruction degradation, but also whether codebook usage and token concentration shift under OOD data.

### 8.2 Experimental Setup

Datasets:

- ImageNet validation as the in-distribution reference
- ImageNet-V2 as a mild distribution shift
- ImageNet-Sketch as a stronger distribution shift
- TODO: add or remove candidate datasets after final selection

Procedure:

1. Run reconstruction on each dataset.
2. Export token IDs and codebook usage summaries.
3. Compare reconstruction metrics across datasets.
4. Compare active code counts, entropy, perplexity, top-k mass, and positional entropy across datasets.

Scripts:

- VQGAN reconstruction: `taming-transformers/scripts/reconstruct_imagenet_single.py`
- LlamaGen reconstruction: `LlamaGen/scripts/reconstruct_imagenet.py`
- VQGAN code usage: `taming-transformers/scripts/vqgan_code_usage_export.py`
- LlamaGen code usage: `LlamaGen/scripts/llamagen_code_usage_export.py`

Measured quantities:

- MSE
- PSNR
- SSIM
- LPIPS
- rFID
- active code count
- entropy and perplexity
- top-k mass
- positional entropy

### 8.3 Experimental Results

TODO.

### 8.4 Discussion

TODO.

## 9. Cross-Question Discussion

TODO.

Possible discussion themes:

- aggregate reconstruction metrics do not fully characterize tokenizer behavior
- nominal codebook size can differ sharply from effective codebook size
- local image perturbations can produce non-local token changes
- token-space edits can produce visible decoder spillover outside the edited patch
- codebook geometry perturbations may reveal whether embedding-space neighbors correspond to visually gentle edits
- OOD images may expose tokenizer assumptions that are hidden on ImageNet
- encoder-side and decoder-side locality failures may have different causes and different implications for autoregressive generation

## 10. Limitations

TODO.

Candidate limitations:

- some analyses are exploratory and need reruns with fixed protocols
- notebook outputs are not yet cleanly reproducible
- some result paths point to external experiment directories
- VQGAN and LlamaGen use different training objectives and codebook behavior, so comparisons should be interpreted carefully
- dead-code handling differs between the models for methodological reasons

## 11. Implementation and Code Organization

### 11.1 Repositories

This project uses three local code areas:

```text
taming-transformers/   VQGAN / Taming Transformers codebase plus custom scripts
LlamaGen/              LlamaGen codebase plus custom scripts
vq-explore/            notes, notebooks, analysis scripts, thesis draft
```

### 11.2 Custom Experiment Scripts

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

### 11.3 Notebook Analysis

Current notebooks:

- `vq-explore/notebooks/vq-code-usage-analysis.ipynb`
- `vq-explore/notebooks/vq-global-robustness-analysis.ipynb`
- `vq-explore/notebooks/vq-encoder-locality.ipynb`
- `vq-explore/notebooks/vq-encoder-locality-updated.ipynb`
- `vq-explore/notebooks/vq-decoder-locality.ipynb`
- `vq-explore/notebooks/vq-decoder-locality-v2.ipynb`
- `vq-explore/notebooks/vq-end-to-end-reconstruction-comparison.ipynb`

TODO: clean notebooks or replace selected analyses with reproducible scripts before final thesis submission.

## 12. Computational Environment

### 12.1 VQGAN / Taming Transformers Container

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

### 12.2 LlamaGen Container

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

### 12.3 Evaluator Container

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

## 13. Conclusion

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
