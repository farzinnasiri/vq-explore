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

Autoregressive visual generation rests on a family of methods that compress images into discrete visual tokens and then model those tokens as sequences. This review covers that line of work: VQ-VAE (van den Oord et al., 2017), VQGAN/Taming Transformers (Esser et al., 2021), DALL-E-style discrete autoencoders (Ramesh et al., 2021), and LlamaGen (Sun et al., 2024). Diffusion models are not reviewed in detail because the experiments do not compare generative sampling methods; they compare the behavior of the tokenizers that make autoregressive image generation possible. Several figure placeholders in this chapter are pedagogical redraws inspired by Miranda (2021), adapted to match the notation and scope of this thesis.

### 2.1 Autoregressive Models: From Text to Images

Autoregressive generation factorizes a sequence probability as:

```text
p(x) = product over i of p(x_i | x_1, ..., x_{i-1})
```

For text, this factorization aligns naturally with the representation used by the model. A sentence is already a sequence of discrete symbols after tokenization, and next-token prediction can be trained directly on that sequence. Images have a less convenient structure. A color image is usually represented as a dense tensor of continuous pixel intensities with shape `height x width x channels`. The signal is two-dimensional, local neighborhoods are important, and the number of raw values is large: a `256 x 256` RGB image contains `196,608` scalar pixel values before any modeling context is considered.

An autoregressive image model must therefore make two design choices before it can use a standard sequence model. First, it must choose a sequence order for a two-dimensional object, usually by flattening a spatial grid. Second, it must choose the unit of prediction. Pixel-level autoregressive models predict raw or discretized pixel values, but their sequences are long and the resulting model must learn both low-level image statistics and long-range structure. Modern autoregressive image generators often avoid this by introducing a learned image tokenizer. The tokenizer compresses an image into a shorter grid of discrete codes; the transformer then models this grid as a sequence.

The scaling constraint is central. If a transformer receives a sequence of length `T`, full self-attention has `O(T^2)` time and memory cost. For image-like inputs, `T` can be very large if we flatten at pixel level. A concrete order-of-magnitude comparison is:

```text
pixel-level sequence for 224x224 RGB: T = 224 * 224 * 3 = 150,528
token-level sequence for 256x256 with f=16: T = 16 * 16 = 256
```

This gap motivates the two-stage design used in VQ-based image generation: first learn a compact discrete representation, then model token dependencies autoregressively.

> **Figure placeholder 2.1a: Pixel Flattening vs Token Sequence Length.**
> Place this immediately after the complexity paragraph. Left panel: a `224 x 224 x 3` image tensor expanded into a long 1D sequence; annotate `T = 150,528`. Right panel: a `16 x 16` token grid flattened to length `256`; annotate `T = 256`. Add a small note below: `self-attention cost grows quadratically in T`.

In this two-stage design, the first stage is an autoencoder-like tokenizer and the second stage is an autoregressive prior. The tokenizer determines the visual vocabulary and the spatial resolution of the sequence. For example, a tokenizer with downsampling factor `f = 16` maps a `256 x 256` image to a `16 x 16` grid, which contains only `256` image tokens. The autoregressive model then predicts a sequence of 256 discrete token IDs rather than hundreds of thousands of pixel-channel values.

This compression is not merely an engineering convenience. It changes the learning problem. The transformer no longer models pixels directly; it models the distribution of learned visual parts. The quality and behavior of the tokenizer therefore constrain the whole generative system. If the tokenizer loses information, uses its codebook unevenly, or reacts unstably to small input changes, those properties become part of the data distribution seen by the autoregressive model.

> **Figure placeholder 2.1: Two-stage autoregressive image generation.**
> Add a diagram showing: input image -> encoder -> token grid -> flattened token sequence -> autoregressive transformer -> generated token sequence -> decoder -> output image. The figure should highlight the reduction from `256 x 256 x 3` pixels to a `16 x 16` token grid for the `f = 16` setting used in this project.

> **Figure placeholder 2.1b: Local Features vs Long-Range Dependencies.**
> Place this at the end of Section 2.1. Three blocks: (1) CNN feature hierarchy (low-level edges/colors -> mid-level motifs -> high-level parts), (2) discrete tokenization/codebook stage, (3) transformer modeling long-range relationships among tokens. Caption idea: local composition and global dependency modeling are separated but complementary.

This framing motivates the central choice in the thesis: rather than evaluating only final generated images, the experiments study the tokenizer itself. Reconstruction metrics test whether the autoencoder preserves images on average, but they do not fully explain how the representation is organized.

### 2.2 Image Tokens and Vector Quantization

The Vector Quantised Variational Autoencoder (VQ-VAE) introduced a discrete latent bottleneck for representation learning (van den Oord et al., 2017). The encoder produces continuous latent vectors, each vector is assigned to its nearest codebook entry, and the decoder reconstructs from the quantized representation. This avoids the purely continuous latent representation of standard VAEs and allows a learned prior to model the discrete latent space.

In simplified form:

```text
z_e = Encoder(x)
k = argmin_j || z_e - e_j ||_2
z_q = e_k
x_hat = Decoder(z_q)
```

Here, `e_j` is a codebook vector and `k` is the discrete token ID.

More explicitly, an encoder maps an image `x` to a latent tensor `z_e(x)` with spatial dimensions `h x w` and channel dimension `d`. Each vector `z_e(x)_{u,v}` is replaced by the nearest entry in a learned codebook `E = {e_1, ..., e_K}`. The result is both a quantized latent tensor `z_q` and an integer token grid `k`:

```text
k_{u,v} = argmin_j || z_e(x)_{u,v} - e_j ||_2
z_q(x)_{u,v} = e_{k_{u,v}}
```

The decoder receives `z_q`, not the original continuous encoder output. This forces image information through a finite vocabulary. The codebook size `K` controls the nominal number of possible visual symbols, while the spatial grid size controls how many symbols represent one image.

Training a VQ model requires handling the non-differentiable nearest-neighbor assignment. VQ-VAE uses a straight-through estimator so gradients from the decoder can update the encoder even though the forward pass uses discrete code assignments. The loss also includes a codebook term and a commitment term, encouraging codebook vectors to move toward encoder outputs and encouraging encoder outputs not to fluctuate arbitrarily around the codebook. Following the VQ-VAE formulation, the usual objective can be summarized as:

```text
L = L_rec
  + || sg[z_e(x)] - e ||_2^2
  + beta * || z_e(x) - sg[e] ||_2^2
```

where `sg[.]` denotes stop-gradient and `beta` controls the commitment penalty. The exact reconstruction loss depends on the model family. Early VQ-VAE work used reconstruction objectives suitable for representation learning and likelihood modeling. VQGAN later added perceptual and adversarial components to improve image sharpness.

The token grid can be interpreted in two equivalent ways. For the autoencoder, it is an index map telling the decoder which codebook vector to use at each spatial location. For the autoregressive prior, it is a sequence over a vocabulary of size `K`. This dual role is why codebook behavior matters: a code that is useful for reconstruction is also a symbol the transformer may need to predict.

From a signal-processing viewpoint, vector quantization can be interpreted as mapping many continuous latent vectors to a finite set of representative centroids. In intuitive terms, one learns a dictionary of visual prototypes (the codebook) and replaces each latent vector by the index of its nearest prototype. This gives the model a discrete alphabet without requiring pixel-level sequence modeling.

> **Figure placeholder 2.2a: VQ as Clustering and Codebook Construction.**
> Place this after the paragraph above. Three panels: (1) latent vectors as points in 2D/3D, (2) partition into clusters with one centroid per cluster, (3) lookup-table style codebook with entries `z_0, z_1, ..., z_{K-1}`. Caption should state that quantization maps each latent to the nearest centroid and emits a discrete index.

Two practical properties are especially important for this thesis.

First, the nominal vocabulary and the empirical vocabulary can differ. A model may have `K = 16,384` entries, but only a subset may appear on a dataset. This can happen because some codes are unused after training or because the evaluated data occupies only part of the learned visual space. The distinction between nominal and active codebook size is central to RQ1 and RQ5.

Second, the quantization boundary can create discontinuities. A small image perturbation may move an encoder output across a nearest-neighbor boundary and change the token ID. This change can be local if only the affected spatial region crosses a boundary, or it can be non-local if the encoder architecture spreads information across a wider area. This motivates the encoder-locality experiment in RQ3.

> **Figure placeholder 2.2: Vector quantization in latent space.**
> Add a schematic with continuous encoder vectors as points, codebook entries as centroids, nearest-neighbor regions as Voronoi cells, and a highlighted point crossing a boundary after a small perturbation. The caption should connect this boundary crossing to token flips in RQ2 and RQ3.

> **Figure placeholder 2.2b: From Encoded Feature Map to Discrete Token Sequence.**
> Place this at the end of Section 2.2. Show: image `x` -> encoder output `z_hat` (continuous grid) -> quantized grid `z_q` (indices into codebook) -> flattened token sequence `s_0, s_1, ..., s_n`. Caption should emphasize that the transformer is trained on codebook indices, not raw pixels.

### 2.3 VQGAN and Taming Transformers

VQGAN extends vector-quantized autoencoding with perceptual and adversarial losses (Esser et al., 2021). The goal is to learn a codebook of perceptually meaningful visual constituents and reconstruct sharper images than pixel-loss-only autoencoders. Taming Transformers then models the resulting discrete image tokens with transformers for high-resolution image synthesis.

The main limitation of a pixel-loss autoencoder is that pixel fidelity and perceptual fidelity are not the same objective. Minimizing an average pixel error can encourage blurry reconstructions when several plausible high-frequency explanations exist. VQGAN addresses this by combining vector quantization with a perceptual reconstruction loss and an adversarial discriminator. The perceptual term compares images in a learned feature space, while the adversarial term encourages reconstructions to lie on the natural-image manifold. Taming Transformers uses this improved first-stage model as the tokenizer for transformer-based high-resolution synthesis (Esser et al., 2021).

Architecturally, the VQGAN tokenizer used in this work follows the standard first-stage design from Taming Transformers. An image is passed through a convolutional encoder, projected into the codebook embedding space, quantized by nearest-neighbor lookup, projected back into the decoder latent space, and reconstructed by a convolutional decoder. This produces a sequence of transformations from image space to continuous latent space, from continuous latent space to discrete codebook indices, and finally back to image space.

In practical implementations, this stage is trained with a reconstruction objective plus perceptual and adversarial terms. The intuition is that pixel-level matching alone can underweight perceptual realism, while adversarial and perceptual losses encourage sharper and more semantically consistent reconstructions.

> **Figure placeholder 2.3a: Why Add Adversarial Training to the Tokenizer.**
> Place this before the tokenizer configuration bullet list. Diagram: generator pathway reconstructing `x_hat` from `x`, discriminator scoring realism, and a perceptual-feature branch comparing `x` and `x_hat`. Caption should explain that the first stage jointly optimizes reconstruction fidelity and perceptual realism.

This separation between encoding, quantization, and decoding is important experimentally because it makes it possible to intervene at different points in the tokenizer. Perturbations applied before encoding test the stability of the encoder and quantizer, while perturbations applied directly to token IDs test the spatial behavior of the decoder.

The concrete configuration used in this thesis is specified in Chapter 3. Conceptually, the VQGAN baseline represents the older perceptual-adversarial tokenizer paradigm: a first-stage model optimized to reconstruct natural images sharply enough that the resulting discrete representation becomes useful for transformer-based generation.

One important methodological detail is that the nominal VQGAN codebook can be much larger than the empirically active set. This matters because a token replacement experiment should not silently replace a valid token with a dead code that the model rarely or never uses in the evaluated setting. For VQGAN, the decoder-locality experiments therefore define nearest, farthest, and orthogonal replacements over the observed active subset rather than over the full nominal vocabulary.

> **Figure placeholder 2.3: VQGAN first-stage model.**
> Add an architecture diagram for the VQGAN first-stage path: encoder, projection into codebook space, vector quantization, projection into decoder space, and decoder. Annotate the project settings: `256 x 256` input, `16 x 16` token grid, `K = 16,384`, `embed_dim = 256`.

> **Figure placeholder 2.3b: Two-Stage Training Schedule.**
> Place this at the end of Section 2.3. Panel A: train tokenizer (encoder/quantizer/decoder, plus discriminator). Panel B: freeze tokenizer, encode dataset into token sequences, train autoregressive transformer with next-token prediction on indices.

The literature usually evaluates this first stage by reconstruction quality, especially rFID, LPIPS, or related perceptual metrics. Those metrics are necessary but incomplete for this thesis. A tokenizer can achieve strong average reconstruction while still having representational properties that are important for downstream autoregressive modeling: concentrated code usage, unstable token assignments, or decoder changes outside an edited token patch. The VQGAN baseline is therefore used both as a reconstruction model and as a system whose internal discrete representation can be probed.

### 2.4 LlamaGen Tokenizer

LlamaGen revisits autoregressive image generation using Llama-style next-token prediction (Sun et al., 2024). It includes VQ tokenizers with downsampling ratios such as `VQ-16` and `VQ-8`, and reports strong reconstruction and generation performance. In this project, LlamaGen is compared against VQGAN as a more recent tokenizer/generator stack.

The LlamaGen paper asks whether the next-token prediction paradigm used by large language models can scale to image generation when images are represented as discrete tokens. Its system keeps the broad two-stage structure: an image tokenizer maps images to token grids, and a Llama-style autoregressive model predicts those tokens (Sun et al., 2024). The released tokenizers include downsampling ratios `16` and `8`, corresponding to `16 x 16` and `32 x 32` token grids for `256 x 256` images. In the class-conditional ImageNet setting, the reported rFID values include `2.19` for the `VQ-16` tokenizer at `16 x 16` tokens and `0.59` for the `VQ-8` tokenizer at `32 x 32` tokens.

The LlamaGen tokenizer is structurally similar to the VQGAN first stage but belongs to a later design regime in which stronger reconstruction performance and denser codebook usage are treated as prerequisites for scalable next-token image modeling. The concrete hyperparameters used in this thesis are specified in Chapter 3.

Conceptually, it supports the same experimental decomposition as the VQGAN first stage: images can be encoded into quantized latent representations, quantized latents can be decoded back to images, and token IDs can be decoded after direct token-space intervention.

The vector quantizer computes nearest-neighbor assignments in codebook space. When L2 normalization is enabled, both encoder outputs and codebook embeddings are normalized before distance computation. This changes the geometry of codebook interventions. On the unit sphere, Euclidean distance and cosine similarity are tightly linked: the farthest code under Euclidean distance is the one with the most negative cosine similarity, whereas an orthogonal intervention targets cosine similarity near zero. These are therefore distinct operations and should not be conflated.

For this thesis, the main LlamaGen comparison uses `VQ-16`. This makes the spatial token grid directly comparable to the VQGAN `f = 16` baseline: both map a `256 x 256` image to `16 x 16` tokens with the same nominal vocabulary size. The two tokenizers differ, however, in codebook dimensionality, normalization, training recipe, and empirical code usage. Those differences are useful rather than incidental. They allow the experiments to separate properties that are common to discrete image tokenizers from properties that depend on a particular tokenizer design.

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

For the second-stage autoregressive model, training follows the same conditional factorization used in language models, but over visual token indices. A simple plain-text form is:

```text
given sequence s = (s_0, s_1, ..., s_n):
model learns p(s_i | s_0, ..., s_{i-1}) for each i
overall sequence likelihood: p(s) = product over i of p(s_i | s_<i)
```

This formulation is useful for interpretation: robustness or locality failures in the tokenizer modify the token sequence distribution seen by the transformer, which then affects both likelihood and sampling behavior.

The experiments in this thesis are designed around this broader view of tokenizer quality. RQ1 establishes reconstruction and codebook usage. RQ2 measures global stability under additive noise. RQ3 tests encoder locality by perturbing local image patches and measuring token changes. RQ4 tests decoder locality by editing local token patches and measuring image-space changes. RQ5 asks whether reconstruction and codebook behavior remain stable under distribution shift.

> **Figure placeholder 2.5: Taxonomy of tokenizer probes.**
> Add a four-panel explanatory figure: reconstruction baseline, global noise, local image-space perturbation through the encoder, and local token-space perturbation through the decoder. Each panel should show the measured object: image metrics, token flips, leakage outside the patch, or codebook usage.

> **Figure placeholder 2.5a: Next-Token Prediction Over Visual Indices.**
> Place this after Figure placeholder 2.5. Show a token prefix `s_0 ... s_4` followed by a masked/unknown `s_5`, and a probability distribution over candidate next indices. Caption should connect visual autoregression to standard language-model next-token training.

This thesis therefore treats the tokenizer as a measurable object rather than a black-box preprocessing step. The goal is not to claim that one tokenizer is universally better, but to identify which properties are visible only when the discrete representation is probed directly.

The next chapter turns this conceptual taxonomy into a concrete specification by fixing the model instances, datasets, and measurement definitions used in the experiments.

## 3. Datasets, Models, and Metrics

The following sections fix the model instances, datasets, and measurement definitions used across all research-question chapters. Each RQ chapter then contains its own specific procedure and results.

### 3.1 Models Under Study

This thesis compares two tokenizer families that both support autoregressive modeling over discrete visual indices, but make different architectural and training choices.

#### 3.1.1 VQGAN Tokenizer (Taming Transformers)

The first tokenizer is the ImageNet VQGAN first-stage model introduced in Taming Transformers (Esser et al., 2021). The configuration used in this work corresponds to the commonly used `f = 16`, `K = 16,384` setting.

Core configuration used in experiments:

- input image resolution: `256 x 256`
- downsampling factor: `f = 16`
- token-grid resolution: `16 x 16`
- nominal codebook size: `K = 16,384`
- codebook embedding dimension: `256`
- first-stage objective family: reconstruction + perceptual + adversarial terms

The Taming Transformers model card reports reconstruction-level distribution metrics for this first-stage setting and provides a standard baseline for visual tokenization at `256 x 256` resolution.

#### 3.1.2 LlamaGen Tokenizer

The second tokenizer is from LlamaGen (Sun et al., 2024), which revisits large-scale autoregressive image generation with a Llama-style next-token objective and reports both `VQ-16` and `VQ-8` tokenizers.

Core configuration used in experiments:

- default comparison tokenizer: `VQ-16`
- alternative tokenizer for comparison context: `VQ-8`
- input image resolution: `256 x 256`
- token-grid resolution for `VQ-16`: `16 x 16`
- nominal codebook size: `K = 16,384`
- codebook embedding dimension: `8`
- codebook L2 normalization: enabled

In published LlamaGen results, `VQ-16` and `VQ-8` report strong reconstruction metrics, and codebook utilization is substantially higher than classical sparse-codebook behavior often observed in older VQ settings.

#### 3.1.3 Comparability and Caveats

The two tokenizers are directly comparable along one axis: both can represent `256 x 256` images as `16 x 16` discrete token grids with the same nominal vocabulary size. This allows controlled analysis of codebook usage, robustness, and locality without conflating sequence length.

However, the comparison is not architecture-neutral. Differences in codebook dimension, normalization, and training recipe can produce different internal geometries even when external metrics are similar. These differences are treated as informative rather than confounding: they are precisely what allows the experiments to separate properties common to discrete image tokenizers from properties specific to a particular design.

> **Figure placeholder 3.1a: Experimental Model Comparison Table.**
> Add a compact table with rows = tokenizers (`VQGAN f16`, `LlamaGen VQ-16`, optional `LlamaGen VQ-8`) and columns = input resolution, token grid size, codebook size, embedding dimension, normalization, and reported reconstruction metrics.

### 3.2 Datasets

#### 3.2.1 ImageNet Validation

The primary in-distribution evaluation set is ImageNet-1K validation (Deng et al., 2009; Russakovsky et al., 2015). This split provides 50,000 labeled images over 1,000 classes and is the standard benchmark domain for both tokenizers studied here.

Preprocessing protocol:

- convert image to RGB
- resize and center-crop to `256 x 256`
- apply tokenizer-specific normalization

For reproducibility, all models are evaluated with deterministic preprocessing and no data augmentation at test time.

#### 3.2.2 ImageNet-V2

ImageNet-V2 (Recht et al., 2019) replicates the original ImageNet data-collection process and provides new test images for the same semantic label space. It is used here as a near-domain shift benchmark to measure whether codebook usage and reconstruction behavior remain stable under resampled natural-image statistics. The exact ImageNet-V2 variant used in each run (for example, matched-frequency style subsets) is reported with the corresponding results tables.

#### 3.2.3 ImageNet-Sketch

ImageNet-Sketch (Wang et al., 2019) emphasizes shape cues over natural texture statistics, making it a stronger shift benchmark than ImageNet-V2. It probes failure modes that arise when natural-image texture priors are weakened and stress-tests codebook assignments under stylistic and structural abstraction. Together, the two OOD sets form a severity ladder from mild resampling to pronounced distributional departure.

#### 3.2.4 Other Candidate Datasets

Additional OOD datasets may be included for ablations, but the core cross-dataset analysis is anchored on the three datasets above so that distribution-shift conclusions remain interpretable.

> **Figure placeholder 3.2a: Dataset Shift Severity Ladder.**
> Add a three-column visual panel with example thumbnails from ImageNet validation, ImageNet-V2, and ImageNet-Sketch. Caption should explain the intended shift progression: in-distribution -> mild natural-image shift -> strong sketch-like shift.

### 3.3 Metrics

No single scalar captures reconstruction fidelity, perceptual quality, distribution matching, and discrete-representation behavior simultaneously, so the metric stack is intentionally redundant. The sections below define each metric, note what it misses, and explain how it is interpreted in combination with the others.

> **Figure placeholder 3.3a: Metric Taxonomy for Tokenizer Evaluation.**
> Add a matrix with rows = metrics and columns = evaluation questions (`pixel fidelity`, `perceptual fidelity`, `distribution match`, `diversity/coverage`, `codebook efficiency`, `locality/stability`). Mark which metric informs which question.

#### 3.3.1 Pixel-Domain Reconstruction Metrics (MSE, PSNR)

Mean squared error (MSE) and peak signal-to-noise ratio (PSNR) are full-reference pixel metrics. For image pairs `(x, x_hat)` with `N` scalar pixels:

```text
MSE(x, x_hat) = (1/N) * sum_i (x_i - x_hat_i)^2
PSNR(x, x_hat) = 10 * log10(MAX^2 / MSE)
```

For normalized images in `[0, 1]`, `MAX = 1`. Lower MSE and higher PSNR indicate better pixel agreement. Both metrics are sensitive to small spatial misalignment and high-frequency texture shifts, which makes them useful for establishing reconstruction baselines and measuring controlled perturbation deltas, but insufficient on their own as a proxy for perceptual realism.

#### 3.3.2 Structural Similarity (SSIM)

SSIM (Wang et al., 2004) compares local luminance, contrast, and structure instead of only pointwise pixel error. For local windows:

```text
SSIM(x, y) = l(x, y) * c(x, y) * s(x, y)
```

where `l`, `c`, and `s` are luminance, contrast, and structure comparisons. Reported values are typically averaged across windows. SSIM is closer to perceptual structure than pixel-error metrics, though it remains a full-reference metric rather than a learned perceptual model.

#### 3.3.3 Learned Perceptual Similarity (LPIPS)

LPIPS (Zhang et al., 2018) compares deep feature activations between two images across multiple network layers:

```text
LPIPS(x, y) = sum_l w_l * || phi_l(x) - phi_l(y) ||_2^2
```

where `phi_l` are normalized deep features and `w_l` are learned layer weights. Lower LPIPS indicates higher perceptual similarity. It aligns more closely with human perceptual judgments than shallow pixel metrics, though it remains dependent on the specific backbone and calibration choices.

#### 3.3.4 Distribution Metrics (FID, sFID, Inception Score)

FID (Heusel et al., 2017) compares two Gaussian approximations in Inception feature space, with means `mu_r`, `mu_g` and covariances `Sigma_r`, `Sigma_g`:

```text
FID = ||mu_r - mu_g||_2^2
      + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r * Sigma_g)^(1/2))
```

Lower FID indicates closer real-vs-generated (or real-vs-reconstructed) feature distributions.

sFID is a spatial variant used in the OpenAI evaluator stack (Dhariwal and Nichol, 2021; guided-diffusion evaluator). It is reported as an additional distribution-level signal and should be interpreted alongside FID rather than as a replacement.

Inception Score (Salimans et al., 2016) evaluates generated samples by combining label confidence and marginal diversity:

```text
IS = exp( E_x [ KL( p(y|x) || p(y) ) ] )
```

Higher IS favors samples that are individually classifiable (`low entropy p(y|x)`) while collectively diverse (`high entropy p(y)`). FID and sFID measure distribution matching in feature space; IS measures a combination of sample quality and diversity but is not a direct real-vs-model distance.

#### 3.3.5 Precision and Recall for Generative Coverage

Precision/recall for generative models (Kynkäänniemi et al., 2019) estimates fidelity and coverage separately in feature space:

- precision: fraction of generated samples that lie in the support region of real samples
- recall: fraction of real samples covered by the generated-sample support

This separation is important when two models have similar FID but different mode coverage.

> **Figure placeholder 3.3b: Precision-Recall Trade-off in Feature Space.**
> Add a 2D schematic with real manifold and generated manifold overlays, showing high-precision/low-recall and low-precision/high-recall failure modes.

#### 3.3.6 Codebook Usage Metrics

Let `K` be codebook size and `p_k` be empirical token frequency for code `k`. A purely literal definition of activity as `p_k > 0` is often too brittle for finite datasets, because a code that appears once or twice may be operationally negligible while still counting as active. For this reason, codebook usage should be reported at two levels:

- raw activity: whether a code appears at least once
- effective activity: whether a code exceeds a minimum support threshold or belongs to the cumulative mass covering almost all assignments

In this thesis, the default descriptive quantities are defined from the raw empirical distribution, but active-subset analyses should also report an effective-support criterion when the distinction materially affects conclusions.

```text
raw_active_count = number of k such that p_k > 0
effective_active_count = number of k such that count(k) >= tau
active_fraction = active_count / K
entropy H = - sum_k p_k * log(p_k)
perplexity = exp(H)
top-k mass = sum_{j in top-k codes} p_j
```

Here `tau` denotes a minimum frequency threshold chosen to suppress one-off or extremely rare codes when needed for robustness analyses. An equivalent alternative is to define the effective subset by cumulative mass, for example the smallest set of codes whose frequencies account for `99.9%` of all assignments.

Positional entropy is computed per token-grid location `(u, v)` using the local token distribution at that position across images.

Active fraction and perplexity together quantify effective vocabulary size. Top-k mass quantifies how concentrated assignments are, with high concentration indicating near-collapse onto a small subset of codes. Positional entropy reveals whether individual grid positions have specialized to particular codebook regions. Raw and effective activity should be compared whenever rare-code behavior is consequential for downstream interventions such as token replacement.

#### 3.3.7 Locality and Robustness Metrics

For clean tokens `t` and perturbed tokens `t'`, define token flip indicators `1[t_i != t'_i]`.

```text
flip_rate(region) = mean_i in region 1[t_i != t'_i]
leakage_ratio = flip_rate(outside_region) / flip_rate(inside_region)
```

Encoder-locality metrics use token-space regions induced by image-space perturbation masks. Decoder-locality metrics use image-space change inside/outside edited token patches, together with patch/full-image PSNR, SSIM, and LPIPS.

For global perturbation experiments, robustness is reported as metric degradation curves vs perturbation strength (for example, sigma levels for Gaussian noise).

> **Figure placeholder 3.3c: Locality Leakage Curve.**
> Add a plot of response magnitude vs distance from perturbation boundary, with separate curves for each tokenizer and confidence bands across images.

#### 3.3.8 Reporting Protocol

To avoid over-interpreting single scalars, each metric is reported with distribution-aware summaries:

- mean and standard deviation over samples
- when relevant, bootstrap confidence intervals
- paired comparisons for clean vs perturbed variants on the same images

All major claims in later chapters are based on metric bundles (pixel + perceptual + distribution + codebook/locality), not on a single number.

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
- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A Large-Scale Hierarchical Image Database." CVPR 2009. https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf
- Olga Russakovsky et al. "ImageNet Large Scale Visual Recognition Challenge." IJCV 2015. https://arxiv.org/abs/1409.0575
- Patrick Esser, Robin Rombach, Björn Ommer. "Taming Transformers for High-Resolution Image Synthesis." CVPR 2021. https://arxiv.org/abs/2012.09841
- Taming Transformers codebase. https://github.com/CompVis/taming-transformers
- Aditya Ramesh et al. "Zero-Shot Text-to-Image Generation." 2021. https://arxiv.org/abs/2102.12092
- Peize Sun et al. "Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation." 2024. https://arxiv.org/abs/2406.06525
- LlamaGen codebase. https://github.com/FoundationVision/LlamaGen
- Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019. https://arxiv.org/abs/1902.10811
- Haohan Wang, Songwei Ge, Zachary C. Lipton, and Eric P. Xing. "Learning Robust Global Representations by Penalizing Local Predictive Power." NeurIPS 2019. https://arxiv.org/abs/1905.13549
- Richard Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR 2018. https://arxiv.org/abs/1801.03924
- Martin Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017. https://arxiv.org/abs/1706.08500
- Tim Salimans et al. "Improved Techniques for Training GANs." NeurIPS 2016. https://arxiv.org/abs/1606.03498
- Tuomas Kynkäänniemi et al. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS 2019. https://arxiv.org/abs/1904.06991
- Prafulla Dhariwal and Alexander Nichol. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021. https://arxiv.org/abs/2105.05233
- Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. "Image Quality Assessment: From Error Visibility to Structural Similarity." IEEE Transactions on Image Processing, 2004. https://live.ece.utexas.edu/publications/2004/zwang_ssim_ieeeip2004.pdf
- OpenAI guided-diffusion evaluator repository. https://github.com/openai/guided-diffusion
- Leandro J. V. Miranda. "Understanding CLIP with VQGAN." 2021. https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/
