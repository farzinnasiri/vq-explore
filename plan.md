# Research Plan

## Phase 1

### Objective

Build a strong understanding of the VQ tokenization field and prepare a recap document of existing tokenizers and their properties. No experiments are required at this stage.

---

### Reading Roadmap

1. **VQ-VAE** & **VQGAN** → cover the basics of vector quantization and image tokenization.
2. **LlamaGen** & **Chameleon** → examples of making AR and multimodal modeling work in practice.
3. **TiTok** → 1-dimensional token sequences.
4. **FlexTok** & **Semanticist** → flexible sequence lengths and semantic orderings.
5. After these, **expand the literature search** to include additional VQ-based tokenizers not listed in the thesis proposal.

---

### Deliverable

A **recap document** that:

* Summarizes each tokenizer and its key properties as reported in the respective papers.
* Notes whether public **code and checkpoints** are available.
* Provides a general overview of the field of VQ tokenization.

---

### Focus Areas (from Supervisor’s Draft List)

#### Design Choices (factors we control)

* Training data mixture and augmentations (most public tokenizers use ImageNet).
* 1D vs 2D token sequences.
* Causality enforced in encoder (and whether ever tried in decoder).
* Flexible number of tokens.
* Semantic forcing (e.g. REPA).

#### Not Controllable Properties (factors we observe)

* Codebook size.
* Token usage statistics: entropy, frequency, skew.
* Tokenizer reconstruction quality (e.g., rFID).
* Tokenizer reconstruction on specific subsets (faces, animals, etc.) and out-of-distribution images (e.g., LAION).

#### Effects (outcomes we measure)

* Tokenizer reconstruction ability (in-distribution vs out-of-distribution).
* Tokenizer robustness to perturbations (token values, token order, codebook changes, AR mode).
* AR learnability (noted as tricky to measure).
* AR generation quality (gFID).
* Image understanding from tokens (e.g. linear classification on token representation).
* Robustness to OOD images (both tokenization/reconstruction and AR generation).
* Forgetting during finetuning (to be discussed later).


---
