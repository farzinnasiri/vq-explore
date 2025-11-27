# VQ

## Table of Contents

1. [Phase 1: VQGAN Inference and Evaluation](#phase-1-vqgan-inference-and-evaluation)
   - [Objective](#objective)
   - [Experimental Setup](#experimental-setup)
   - [Results](#results)
   - [Analysis](#analysis)
   - [Next Steps](#next-steps)

## Phase 1: VQGAN Inference and Evaluation

### Objective
- Understand the VQGAN inference path
- Run reconstructions on ImageNet
- Evaluate with IS, FID, sFID, LPIPS, PSNR, SSIM

### Experimental Setup

#### Environment
- **Hardware**: Single GPU - RTX 1080 Ti; container RAM ~28 GB
- **Python Stack**: `numpy==1.19.5`, `scipy==1.9.3`, `pillow==9.0.0`, `scikit-image==0.18.3`, `torchvision==0.14.1+cu117`
- **Model**: `VQModel` loaded from `CONFIG_PATH` and `MODEL_PATH`

#### Data
- **Dataset**: ImageNet `val` at `/datasets/imagenet/val` (50,000 images in 1,000 classes)

#### Pipeline Summary
- Encode–decode per image or batch
- Convert back to `[0,1]` and `uint8` for metrics
- Metrics: LPIPS (VGG), PSNR, SSIM, FID, IS,

### Results

#### Metrics Comparison

| Run | IS | FID | sFID | LPIPS | PSNR | SSIM | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| VQGAN (paper, f=16, 16,384) | — | 4.98 | — | 0.38 ± 0.07 | — | — | NLL 2.32 × 10^3; "reconstruction error" terminology used in paper (not LPIPS); see footnote for IS sampling variants |
| Our experiment | 46.76 | 7.53 | 5.59 | 0.286 | 19.65 | 0.486 | Precision 0.70442; Recall 0.6451; metrics via OpenAI evaluator |

#### Sample Reconstructions (Original VS Reconstructed)

<p align="center">
  <img src="vqgan_samples/000000_n04507155_orig.png" width="200" alt="orig 0">
  <img src="vqgan_samples/000000_n04507155_recon.png" width="200" alt="recon 0">
  <img src="vqgan_samples/000007_n02326432_orig.png" width="200" alt="orig 1">
  <img src="vqgan_samples/000007_n02326432_recon.png" width="200" alt="recon 1">
  <br>
  <img src="vqgan_samples/000014_n02109961_orig.png" width="200" alt="orig 2">
  <img src="vqgan_samples/000014_n02109961_recon.png" width="200" alt="recon 2">
  <img src="vqgan_samples/000021_n02099601_orig.png" width="200" alt="orig 3">
  <img src="vqgan_samples/000021_n02099601_recon.png" width="200" alt="recon 3">
</p>



Footnote:
- Inception Scores (IS) reported for class-conditional ImageNet (256×256) vary by sampling strategy and acceptance rate.
- Standard sampling (no rejection):
  - 70.6 ± 1.8 (mixed k, p=1.0, acceptance 1.0)
  - 74.3 ± 1.8 (k=973, p=0.88, acceptance 1.0)
  - 78.6 ± 1.1 (k=250, p=1.0, acceptance 1.0)
- With rejection sampling (higher quality):
  - 280.3 ± 5.5 (k=600, p=1.0, acceptance 0.05)
  - 304.8 ± 3.6 (mixed k, p=1.0, acceptance 0.05)
  - 402.7 ± 2.9 (mixed k, p=1.0, acceptance 0.005)

### Analysis

#### Notes
- Black images were caused by float16 instability during export; resolved by disabling `autocast`
- Recon tensors occasionally exceed `[-1,1]`, but clamping prior to `uint8` normalization keeps images valid
- sFID confirms spatial structure is preserved; FID is consistent with photorealistic reconstructions(As of 27th Nov 2025, we could not reproduce reported numbers)

#### Metrics Interpretation
- **Evaluation computed on exported reconstructions**
- **Metrics**: Lower is better except IS
- **sFID**: Uses spatial Inception features and is more sensitive to structural/layout errors than standard FID

### Next Steps

#### Codebook Perturbation
**Goal**: Test robustness and semantics of the learned visual vocabulary

**Methods**:
- Zero-out random codes (10–50%)
- Permute indices before decode
- Swap a code with its nearest neighbors

**Procedure**:
- Re-decode with the perturbed codebook
- Save a small grid
- Measure IS/FID/sFID deltas relative to baseline

**Expected Signals**:
- Collapse under heavy ablation ⇒ redundancy low
- Minor FID change ⇒ redundancy/high overlap
- Spatial artifacts under shuffles ⇒ strong code–structure coupling

#### Codebook Quality Assessment
**Goal**: Quantify how effectively the 16,384-codebook is used

**Metrics**:
- Usage histogram across val set
- Perplexity \(exp(H)\) where \(H\) is entropy of code freq
- Count of dead codes (never used)

**Procedure**:
- Encode ImageNet val
- Collect discrete indices
- Aggregate per-code counts
- Report top-100 most/least used codes

**Interpretation**:
- Low perplexity or many dead codes ⇒ capacity underused
- Uniform usage with few dead codes ⇒ healthy codebook

#### Evaluation Framework
- **Standardized Evaluation**: Ensure comparable metrics with `torch-fidelity`
- **Enhanced Metrics**: Consider additional evaluation protocols for comprehensive assessment

### Useful Links
- [Illustrated VQGAN-CLIP Guide](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) – visual walk-through of VQGAN and CLIP interaction

## Phase 2: LlamaGen



## Citations

- Esser, Rombach, Ommer. "Taming Transformers for High-Resolution Image Synthesis." arXiv:2012.09841. https://arxiv.org/abs/2012.09841
- OpenAI Guided Diffusion Evaluations (used for metrics computation): https://github.com/openai/guided-diffusion/tree/main/evaluations
