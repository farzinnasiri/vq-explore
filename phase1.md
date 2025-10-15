# 0. Background: From Pixels to Tokens

Autoregressive (AR) models, which predict the next element in a sequence based on preceding elements, have achieved remarkable success in natural language processing (NLP). This paradigm is a natural fit for text, which is an inherently discrete, one-dimensional sequence. However, applying this same logic to image generation presents a series of fundamental challenges. This section explores these challenges, contrasts the autoregressive approach with the dominant diffusion paradigm, and introduces vector quantization (VQ) as the critical technology that bridges this gap, effectively creating a new "language" for images.

## 0.1 The Challenge: Why Autoregressive Models are Harder for Images than Text

At its core, an AR model factorizes the joint probability distribution of a sequence $x = (x_1, x_2, \ldots, x_N)$ using the chain rule of probability:

$$
p(x) = \prod_{i=1}^{N} p(x_i | x_{<i})
$$

This "next-token prediction" approach, which powers models like GPT (Radford et al., 2019), is fundamentally challenged when the data is not a simple 1D sequence.

* **The Dimensionality Curse:** Images are high-dimensional, continuous signals. A modest 256x256 pixel image, when flattened, results in a sequence of 65,536 pixels, each with three color channels. Modeling such an incredibly long sequence pixel-by-pixel, as early models like PixelRNN attempted (van den Oord et al., 2016), is computationally prohibitive and forces the model to learn extremely long-range dependencies.

* **Lack of Natural Ordering:** Unlike text, images are 2D grids with strong spatial correlations in all directions. There is no intrinsically "correct" way to flatten an image into a 1D sequence. The conventional approach, a raster scan (row-by-row), imposes an arbitrary sequential order. This makes it difficult for the model to capture complex spatial relationships, as spatially close pixels in the 2D grid can become very distant in the 1D sequence, breaking long-range coherence.

* **Spatial Redundancy:** Natural images contain significant spatial redundancy, such as large, uniform areas of color or texture. A pixel-level AR model must expend considerable capacity learning to predict these redundant patterns, which is an inefficient use of model parameters.

## 0.2 Competing Paradigms: Autoregressive vs. Diffusion Models

Before the recent resurgence of AR models, the dominant paradigm in high-fidelity image generation was diffusion models. Understanding their differences is key to appreciating the motivations behind modern VQ-AR systems.

* **Core Mechanism Contrast:**
    * **Autoregressive (AR) Models** operate **sequentially**. They generate an image token by token, where each prediction is conditioned on all previously generated tokens. This is analogous to a writer composing a sentence one word at a time. This unidirectional process can lead to error propagation, where an early mistake impacts the entire subsequent generation.
    * **Diffusion Models** operate **holistically and iteratively**. The process begins with random noise and gradually refines the entire image over dozens of steps by predicting and removing noise (Ho et al., 2020). This iterative refinement allows the model to correct mistakes and achieve global consistency.

* **Key Trade-offs:**
    * **Speed vs. Quality:** AR models are substantially faster at inference, requiring only one forward pass per generated token. Diffusion models must perform many full-image passes, making them slower. However, this iterative process is why diffusion models have traditionally excelled in generating photorealistic, highly detailed images.
    * **Scalability and Simplicity:** AR models leverage the next-token prediction paradigm, which has well-understood scaling properties from NLP. The challenge, therefore, becomes one of representation: if one can represent images as sequences of "visual tokens," AR Transformers might scale for vision the way GPT scaled for text (Yu et al., 2022).

## 0.3 The Bridge: Vector Quantization for Image Tokenization

To overcome the challenges of applying AR models to continuous, high-dimensional images, the field adopted a crucial intermediate step: **vector quantization (VQ)**. A VQ-based tokenizer serves as a bridge, converting a continuous image into a discrete sequence of tokens, effectively creating a "language" for images that AR models can generate.

This idea was pioneered by the Vector-Quantised Variational Autoencoder (VQ-VAE), which integrated a learnable discrete codebook into an autoencoder framework (van den Oord & Vinyals, 2017). The process works as follows:
1.  An **encoder** maps patches of the input image to continuous vectors.
2.  Each vector is then "quantized" by replacing it with the nearest vector from a finite, learned **codebook**.
3.  The sequence of indices of these codebook vectors becomes the discrete token representation of the image.
4.  A **decoder** reconstructs the image from this sequence of discrete tokens.

By converting an image from a grid of 65,536 pixels into, for example, a 16x16 grid of 256 tokens, VQ dramatically shortens the sequence length and forces the model to learn a vocabulary of meaningful visual patterns rather than raw pixel values. This tokenized representation is then suitable for a powerful AR model, such as a Transformer, to learn its distribution.

To make this concept more intuitive, consider these analogies:

* **The LEGO Set:** A real-world object has continuous surfaces. A VQ tokenizer acts like a system that represents this object using a finite set of standard LEGO bricks (the codebook). The reconstructed object is built by assembling these discrete bricks according to a blueprint—the sequence of token IDs.
* **The Cookbook:** Visual patterns like textures and edges are like complex flavors. A VQ codebook is a cookbook with a finite number of "flavor profile" recipes. The tokenizer analyzes a patch of the image, finds the closest matching recipe, and represents the image as a sequence of these recipe IDs.

By successfully representing images as discrete token sequences, VQ enables a new class of powerful and efficient generative models. Evaluating the quality of these tokenized representations and the final generated images requires a specialized set of metrics, which are detailed in the following section.

---
### References

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In *Advances in Neural Information Processing Systems 33* (NeurIPS 2020).

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*, 1(8), 9.

van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel recurrent neural networks. In *Proceedings of the 33rd International Conference on Machine Learning* (ICML).

van den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., ... & Le, Q. V. (2022). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. *arXiv preprint arXiv:2206.10789*.

# 1. Evaluation Metrics in Image Generation

The evaluation of generative models, particularly in the domain of image synthesis, is a multifaceted challenge. The quality of a generated image can be assessed from several perspectives: its fidelity as an independent sample, its realism compared to the distribution of real-world images, and its faithfulness to a specific reference image. Consequently, a variety of metrics have been developed, each designed to capture different aspects of performance. These metrics can be broadly categorized based on whether the generation task is content-variant, allowing for multiple correct outputs, or content-invariant, where there is a single ground-truth target. Further metrics have been developed specifically for models employing vector quantization, focusing on the efficiency and representational quality of the learned discrete codebooks.

## 1.1 Content-Variant Metrics

Content-variant metrics are essential for tasks where the generative model is expected to produce a diverse range of outputs from a given input, such as unconditional image generation from a noise vector. In these scenarios, there is no single "correct" image, so evaluation must focus on the overall quality and diversity of the generated distribution.

### 1.1.1 Inception Score (IS)

The Inception Score (IS) was one of the first widely adopted metrics for quantitatively assessing the performance of generative models (Salimans et al., 2016). It is designed to simultaneously measure two key properties: the quality of individual images (realism) and the diversity of the set of generated images.

The intuition behind IS is twofold. First, a high-quality, realistic image should be easily classifiable by a pre-trained image classifier like Inception-v3. This means the conditional probability distribution $p(y|x)$ over the labels $y$ for a given generated image $x$ should have low entropy; the model should be confident about what object is in the image. Second, for the generated set to be diverse, the model should produce a wide variety of objects. Therefore, the marginal probability distribution over all labels, $p(y) = \int p(y|x=G(z))dz$, should have high entropy, indicating a uniform distribution of classes.

The Inception Score combines these two ideas using the Kullback-Leibler (KL) divergence, calculated as:

$$
\text{IS}(G) = \exp\left(\mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) || p(y))]\right)
$$

A higher IS indicates that the generated images are both individually distinct (low conditional entropy) and collectively diverse (high marginal entropy).

### 1.1.2 Fréchet Inception Distance (FID)

While IS measures quality and diversity, it does not directly compare the generated distribution to the real data distribution. The Fréchet Inception Distance (FID) addresses this limitation by measuring the distance between the feature distributions of real and generated images (Heusel et al., 2017).

The process involves embedding both a set of real images and a set of generated images into a feature space using a pre-trained Inception-v3 model. The activations from a specific layer (typically the final pooling layer) are collected for each set. These collections of feature vectors are then modeled as multivariate Gaussian distributions. FID calculates the Wasserstein-2 distance between these two Gaussians.

Given the mean ($\mu_r$, $\mu_g$) and covariance matrices ($\Sigma_r$, $\Sigma_g$) of the real and generated feature distributions, the FID is calculated as:

$$
\text{FID}(r, g) = ||\mu_r - \mu_g||^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

A lower FID score signifies that the statistics of the generated image features are more similar to those of real images, indicating higher quality and realism. Variants such as Reconstruction FID (rFID) and Generation FID (gFID) adapt this metric for specific conditional tasks.

## 1.2 Content-Invariant Metrics

Content-invariant metrics are most suitable for tasks where there is a single, well-defined ground-truth image that the generated output should match. These are common in image-to-image translation, super-resolution, and restoration tasks.

### 1.2.1 Learned Perceptual Image Patch Similarity (LPIPS)

Traditional pixel-wise metrics like L2 distance often fail to capture perceptual similarity; two images can be perceptually very different yet have a small pixel-wise error. The Learned Perceptual Image Patch Similarity (LPIPS) metric was developed to better align with human perceptual judgment (Zhang et al., 2018).

LPIPS computes the distance between the deep feature activations of two images. It feeds two images (a generated image and its ground-truth reference) through a pre-trained deep network (such as VGG) and extracts features from multiple layers. The distance is then calculated as a weighted sum of the L2 distances between the normalized feature activations from each layer. This "perceptual loss" has been shown to be highly correlated with how humans perceive the similarity between images. A lower LPIPS score indicates that two images are more perceptually similar.

### 1.2.2 Structural Similarity Index Metric (SSIM)

The Structural Similarity Index Metric (SSIM) is designed to measure image quality degradation as a change in the perception of structural information. Unlike pixel-wise error metrics, SSIM evaluates the similarity between two images, $x$ and $y$, based on three components: luminance, contrast, and structure (Wang et al., 2004).

The three components are defined as:

* **Luminance:** $l(x, y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$
* **Contrast:** $c(x, y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$
* **Structure:** $s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$

where $\mu$ is the mean, $\sigma$ is the standard deviation, and $\sigma_{xy}$ is the covariance. The final SSIM score is a combination of these three, typically with exponents $\alpha, \beta, \gamma$ set to 1:

$$
\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma
$$

The score ranges from -1 to 1, where 1 indicates a perfect match. Mean SSIM (MSSIM) is the average SSIM value calculated over multiple local windows of an image.

### 1.2.3 Peak Signal-to-Noise Ratio (PSNR)

Peak Signal-to-Noise Ratio (PSNR) is a classic engineering metric used to quantify the reconstruction quality of lossy compression (Gonzalez & Woods, 2018). It is derived from the Mean Squared Error (MSE) between a ground-truth image $I$ and a generated or compressed image $K$.

First, the MSE is calculated:

$$
\text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2
$$

where $m \times n$ are the image dimensions. PSNR is then defined in decibels (dB) as:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
$$

Here, $\text{MAX}_I$ is the maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image). A higher PSNR generally indicates a higher-quality reconstruction, though it does not always correlate well with human perception.

### 1.2.4 Summary of Image Quality Metrics

| **Metric** | **Measures** | **Value Range** | **Direction** | **Interpretation** | **Main Use Case** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Content-Variant Metrics** |
| **Inception Score (IS)** | Image quality and diversity | 1 to 10+ (theoretical max log(1000) ≈ 6.9) | Higher is better | IS > 8: Excellent, 5–8: Good, 3–5: Fair, <3: Poor | Unconditional generation |
| **Fréchet Inception Distance (FID)**| Distribution similarity | 0 to 300+ | Lower is better | FID < 30: Excellent, 30–50: Good, 50–100: Fair, >100: Poor | General image generation quality |
| **rFID (conditional)** | Conditional FID | 0 to 300+ | Lower is better | Similar to FID, for conditional tasks | Super-resolution, inpainting |
| **gFID (geometric)** | Geometric variant of FID | 0 to 300+ | Lower is better | Alternative FID calculation | Research contexts |
| **Content-Invariant Metrics** |
| **LPIPS** | Learned perceptual similarity | 0 to 1+ | Lower is better | LPIPS < 0.1: Very similar, 0.1–0.3: Similar, >0.3: Different | Perceptual quality assessment |
| **SSIM** | Structural similarity | -1 to 1 | Higher is better | > 0.9: Excellent, 0.8–0.9: Good, 0.6–0.8: Fair, <0.6: Poor | Pixel-level structural comparison |
| **MSSIM** | Multi-scale structural similarity | 0 to 1 | Higher is better | Similar to SSIM, more robust to scale | Different viewing distances/scales |
| **PSNR** | Peak signal-to-noise ratio | 10 to 50+ dB | Higher is better | > 40dB: Excellent, 30–40dB: Good, 20–30dB: Fair, <20dB: Poor| Traditional pixel-level quality |

## 1.3 Metrics for Vector Quantization and Autoregressive Models

For generative models that rely on a discrete codebook (vector quantization) followed by an autoregressive model to generate token sequences, a specialized set of metrics is required to evaluate the intermediate representation and the sequence modeling performance. This two-stage approach was prominently established by models like VQ-VAE (van den Oord & Vinyals, 2017).

### 1.3.1 Codebook and Token Utilization Metrics

* **Codebook Utilization Rate:** This metric measures the percentage of available vectors (codes) in the codebook that are actually used during the encoding of a representative dataset. Low utilization, or "codebook collapse," suggests that the model is relying on only a small subset of its learned codes, indicating representational inefficiency.
* **Token Usage Entropy:** This quantifies the uniformity of the token distribution. For a codebook of size $K$, the entropy is calculated as $H(p) = -\sum_{k=1}^{K} p_k \log_2(p_k)$, where $p_k$ is the frequency of the $k$-th token. Higher entropy indicates a more balanced and efficient use of the entire codebook.
* **Token Frequency Distribution Skew:** This measures the concentration of usage among a few popular tokens. High skew indicates an imbalanced distribution where a few codes dominate, which can be detrimental to learning a rich representation.

### 1.3.2 Autoregressive Generation Performance

The metrics used to evaluate the second stage of these models are adapted from natural language processing, where autoregressive models are standard. The application of these models to pixel-level generation established their use in the image domain (van den Oord et al., 2016).

* **Bits Per Token (BPT):** This metric measures the compression efficiency of the autoregressive model, calculated as the average negative log-likelihood of the token sequences. A lower BPT indicates that the model is better at predicting the next token, implying a superior understanding of the token distribution.
* **Negative Log-Likelihood (NLL):** Directly measures how well the autoregressive model predicts the sequence of tokens produced by the encoder. A lower NLL signifies better modeling of the tokenized data's probability distribution.
* **Perplexity:** Calculated as $\exp(\text{NLL})$, perplexity is an intuitive measure of how "surprised" the model is by a sequence of tokens. Lower perplexity indicates better sequence modeling and prediction capabilities.

### 1.3.3 Summary of VQ and AR Metrics

| **Metric** | **Measures** | **Direction** | **Interpretation** | **Main Use Case** |
| :--- | :--- | :--- | :--- | :--- |
| **Codebook & Token Utilization** |
| **Codebook Utilization** | Percentage of codebook used | Higher is better | High rate indicates efficient codebook; low rate suggests collapse | Diagnosing VQ model efficiency |
| **Token Usage Entropy** | Uniformity of token usage | Higher is better | High entropy means balanced usage; low means unbalanced | Assessing codebook balance |
| **Token Frequency Skew** | Concentration of token usage | Lower is better | Low skew means balanced usage; high skew means few tokens dominate | Identifying token over-specialization|
| **Autoregressive Performance** |
| **Bits Per Token (BPT)** | Compression efficiency | Lower is better | Lower BPT means better prediction and modeling of token distribution | Evaluating AR model compression |
| **NLL** | Predictive accuracy of model | Lower is better | Lower NLL means the model is less surprised by the data sequence | Fundamental AR model evaluation |
| **Perplexity** | Model's surprise at data | Lower is better | Lower perplexity indicates better sequence modeling and prediction | Intuitive measure of AR performance |

---
### References

Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4th ed.). Pearson.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. In *Advances in Neural Information Processing Systems 29* (NIPS 2016).

van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel recurrent neural networks. In *Proceedings of the 33rd International Conference on Machine Learning* (ICML).

van den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600-612.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR).

---
# 2. Literature Review: The Evolution of Visual Tokenization

The application of autoregressive (AR) models to image generation is a story of representation. Early attempts to model pixels directly were computationally intractable and failed to capture global structure. The pivotal innovation that unlocked the potential of AR models for vision was the development of vector-quantized (VQ) tokenizers, which transform continuous images into discrete sequences amenable to the powerful "next-token prediction" paradigm that revolutionized natural language processing. This section traces the historical evolution of this "visual language," from its foundational concepts to the sophisticated, semantically-aware tokenizers of the modern era.

## 2.1 Foundational Era: From Discrete Codes to High-Fidelity Tokens (2017-2021)

The journey began with the **Vector Quantised-Variational AutoEncoder (VQ-VAE)** in 2017 (van den Oord & Vinyals, 2017). This work introduced a discrete latent bottleneck into a VAE, elegantly solving the "posterior collapse" problem where a powerful decoder might otherwise ignore the latent codes. By forcing the representation into a discrete form, VQ-VAE enabled the training of an AR prior (like PixelCNN) on these codes to generate coherent, albeit blurry, images. It was the crucial proof-of-concept that discrete image representations could work.

While groundbreaking, VQ-VAE's reconstructions suffered from low fidelity due to its reliance on pixel-wise losses. The breakthrough in quality came with **VQGAN** in 2021 (Esser et al., 2021). By incorporating a patch-based adversarial loss and a perceptual loss (LPIPS), VQGAN forced its decoder to generate realistic details and textures. This leap in reconstruction fidelity produced much sharper tokens and made high-resolution AR generation a practical reality, establishing VQGAN as the de facto standard tokenizer for the next several years.

This two-stage VQ-AR approach culminated in OpenAI's **DALL·E** (Ramesh et al., 2021). A landmark model, DALL·E scaled the paradigm to text-to-image synthesis by training a massive 12-billion parameter transformer to model a unified stream of text tokens followed by image tokens from a discrete VAE. It demonstrated that complex, novel visual concepts could be generated and composed from natural language, proving the power of treating image tokens like words in a foreign language.

## 2.2 The Modern Era: Scaling, Efficiency, and Semantics (2022-Present)

Following the success of DALL·E, the field was briefly dominated by diffusion models, which offered superior photorealism. However, a new wave of research sought to re-establish the viability of AR models by addressing their limitations in speed, coherence, and semantic understanding.

**Scaling and Unification:** The work on **LlamaGen** (Sun et al., 2024) demonstrated that with sufficient scale and an improved VQGAN-style tokenizer, a standard Llama architecture could achieve state-of-the-art FID scores on ImageNet, surpassing prominent diffusion models and re-igniting interest in AR. Concurrently, Meta's **Chameleon (CM3leon)** (Li et al., 2023) pushed towards a unified multimodal architecture. It employed an "early-fusion" approach where image and text tokens are processed by a single transformer, enabling the model to natively understand and generate arbitrarily interleaved sequences of images and text.

**The Shift to 1D and Flexibility:** Recognizing the inefficiency of rigid 2D token grids, **TiTok** (Yu et al., 2024) introduced a Transformer-based 1D tokenizer. It showed that an image could be represented by a highly compact sequence of just 32 tokens, leading to a dramatic acceleration in generation speed. Building on this, **FlexTok** (Bachmann et al., 2025) introduced variable-length token sequences. Using nested dropout, it learns an ordered representation where initial tokens capture high-level semantics and subsequent tokens add finer details, enabling a coarse-to-fine generation process.

**The Rise of Semantics:** A key frontier became bridging the gap between tokens for reconstruction (pixel detail) and for understanding (abstract concepts). **UniTok** (Ma et al., 2025) addressed this by using multi-codebook quantization to expand the representational capacity of tokens, allowing them to capture both semantic and perceptual information. Taking this further, **Semanticist** (Wang et al., 2025) introduced a highly structured latent space with a provable PCA-like hierarchy, explicitly decoupling high-level semantic content from low-level spectral details to create a highly efficient and interpretable visual language.

This progression reveals a clear trajectory. The field moved from making images discrete (VQ-VAE), to making them perceptually accurate (VQGAN), to making them controllable (DALL·E). The latest research is focused on making the token sequences themselves more intelligent—more efficient (TiTok), more adaptive (FlexTok), and more semantically structured (Semanticist). In this new paradigm, the tokenizer is no longer a simple compression utility; it is a powerful inductive bias that pre-organizes visual information, transforming the AR model's task from raw creation to the logical synthesis of meaningful components.

## 2.3 Timeline of Key Developments in Visual Tokenization

| **Year** | **Model / Paper** | **Key Contribution** |
| :--- | :--- | :--- |
| **2017** | **VQ-VAE** | Introduced discrete latent bottlenecks for images, enabling AR priors. |
| **2021** | **VQGAN** | Added adversarial and perceptual losses for high-fidelity token reconstruction. |
| **2021** | **DALL·E** | Scaled the VQ-AR approach for large-scale text-to-image generation. |
| **2023** | **Chameleon (CM3leon)** | Unified image and text generation with a single early-fusion transformer. |
| **2024** | **LlamaGen** | Proved that scaled AR models can outperform diffusion models in fidelity. |
| **2024** | **TiTok** | Introduced an extremely compressed 1D token representation (e.g., 32 tokens). |
| **2024** | **IBQ** | Developed a method to train massive codebooks (262k+) without code collapse. |
| **2024** | **XQ-GAN / SeQ-GAN**| Advanced GAN-based tokenizers with flexible frameworks and training strategies. |
| **2025** | **UniTok** | Used multi-codebook quantization to unify tokens for generation and understanding. |
| **2025** | **FlexTok** | Enabled variable-length, coarse-to-fine generation with ordered 1D tokens. |
| **2025** | **Semanticist** | Created a hierarchical, PCA-like token structure for semantic decoupling. |
| **2025** | **HITA / TokenFlow** | Introduced specialized architectures (holistic tokens, dual codebooks) for improved coherence and semantic alignment. |

---

### References 

Esser, P., Rombach, R., & Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR).

Li, Y., Geng, S., Li, B., Liu, F., Yang, F., et al. (2023). CM3leon: A Multi-modal, Causal-masked, Multi-task Model-in-the-Loop. *arXiv preprint arXiv:2305.07295*.

Ma, Y., Zhang, X., Li, H., & Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. *Project Page*.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., et al. (2021). Zero-Shot Text-to-Image Generation. In *Proceedings of the 38th International Conference on Machine Learning* (ICML).

Sun, J., Wang, R., Li, Y., Wang, W., & Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. *arXiv preprint arXiv:2404.09344*.

van den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. *Project Page*.

Yu, J., Wang, C., Wang, X., & Wang, J. (2024). An Image is Worth 32 Tokens: A Transformer-based 1D Tokenizer. *arXiv preprint arXiv:2404.05923*.

Bachmann, R., et al. (2025). FlexTok: A Trainable Variable-Length Tokenizer for Language-Interfaced Vision. *Project Page*.

---
# 3. A Detailed Analysis of Key Visual Tokenizers

This section provides a detailed summary of the key tokenizers and associated models that have defined the field of autoregressive image generation. Each entry outlines the problem it addresses, its core design innovations, key results, and notable limitations.

## 3.1 Core Tokenizers and Models

### 3.1.1 VQ-VAE (2017) – Learning Discrete Image Codes

* **Problem:** Standard VAEs, when paired with powerful autoregressive decoders, often suffer from "posterior collapse," where the decoder ignores the latent code. The goal was to enable the use of powerful discrete generative models on images.
* **Design:** A CNN encoder maps an image to a feature map, where each vector is quantized to the nearest entry in a learned codebook. A straight-through estimator and a commitment loss are used for training. This forces the model to learn a discrete representation.
* **Results:** VQ-VAE proved that discrete bottlenecks could be trained end-to-end, enabling AR models like PixelCNN to generate new images from the learned codes.
* **Limitations:** Reconstructions were often blurry and lacked fine texture detail due to reliance on MSE loss. The codebook often had underutilized entries ("codebook collapse").

### 3.1.2 VQGAN (2021) – High-Fidelity Image Tokens

* **Problem:** VQ-VAE reconstructions were overly smooth and missed the high-frequency details necessary for photorealistic generation.
* **Design:** VQGAN augmented the VQ-VAE framework with a patch-based GAN discriminator and a perceptual loss (LPIPS). These additions encourage the tokenizer to preserve realistic detail and texture.
* **Results:** Dramatically improved the visual fidelity of reconstructions, producing sharp, photo-realistic images. VQGAN's high-quality tokens became the basis for the first compelling transformer-based image generators.
* **Limitations:** Adversarial training could introduce instability. The model still primarily optimized for reconstruction over semantics and could encode visual quirks to fool the discriminator.

### 3.1.3 DALL·E (2021) – Large-Scale AR Image Generation

* **Problem:** To demonstrate that AR models can generate complex, novel images from text descriptions at a very large scale.
* **Design:** Used a proprietary discrete VAE (dVAE) to encode 256x256 images into a 32×32 grid of 1024 tokens from a large codebook of 8192 entries. A 12-billion parameter Transformer was then trained to model sequences of text and image tokens.
* **Results:** DALL·E produced an unprecedented diversity of creative and coherent images from text prompts, proving that visual tokens combined with transformers could power a highly capable generative model.
* **Limitations:** Images were not fully photo-realistic and often contained artifacts. The tokenizer and model struggled to render text coherently within images.

### 3.1.4 LlamaGen (2024) – Scaling Autoregression to Beat Diffusion

* **Problem:** To prove that a "GPT-for-images," when properly scaled, could outperform leading diffusion models in image generation fidelity and speed.
* **Design:** A family of AR models (up to 3.1B parameters) based on the LLaMA architecture. It uses an improved VQGAN-style tokenizer with a large codebook (8192) and training optimizations to improve reconstruction and codebook utilization.
* **Results:** Achieved a state-of-the-art FID of ~2.18 on ImageNet 256x256, outperforming comparable diffusion models. It also demonstrated significantly faster inference speeds.
* **Limitations:** While strong on benchmarks like ImageNet, external evaluations noted it struggled with some complex compositional prompts (e.g., precise spatial relations) compared to guided diffusion models.

### 3.1.5 Chameleon / CM3leon (2023) – Mixed-Modality Tokens for Images & Text

* **Problem:** To unify image understanding and generation with text in a single "early-fusion" model, requiring a tokenizer that produces image tokens compatible with text tokens.
* **Design:** A large transformer that ingests a single, interleaved sequence of text and image tokens. The image tokenizer converts 512x512 images into a 32x32 grid of 1024 tokens from an 8192-entry codebook, with special oversampling on faces during training to improve fidelity.
* **Results:** Achieved state-of-the-art or competitive results across a breadth of tasks, including image captioning, VQA, and image generation, demonstrating that discrete image tokens can be treated just like word tokens in a unified model.
* **Limitations:** The tokenizer struggles to reconstruct fine-grained text within images, limiting its OCR capabilities. Image generation quality was competitive but not state-of-the-art compared to specialized models.

### 3.1.6 TiTok (2024) – Transformer Tokenizer with 32 Tokens per Image

* **Problem:** To drastically reduce the AR sequence length to accelerate generation by removing the redundancy of fixed 2D grids.
* **Design:** A Transformer-based encoder that uses a small set of learnable query vectors (e.g., 32) to attend over the entire image and produce a highly compressed 1D sequence of discrete tokens.
* **Results:** Represented a 256x256 image with just 32 tokens while achieving an excellent gFID of 1.97 on ImageNet. This enabled generation speeds hundreds of times faster than diffusion.
* **Limitations:** The extreme compression can lead to the loss of fine details. Evaluations showed weaker reconstruction of small text and faces compared to less compressed tokenizers.

### 3.1.7 UniTok (2025) – Unified Tokenizer for Generation and Understanding

* **Problem:** To bridge the gap between high-fidelity generative tokens and semantically meaningful tokens for understanding tasks like classification.
* **Design:** A multi-codebook quantization scheme. A latent feature vector is split into several chunks, and each chunk is quantized with its own independent sub-codebook. This exponentially increases the effective vocabulary size and is trained with a mix of reconstruction and semantic losses.
* **Results:** Achieved state-of-the-art performance on both tasks simultaneously: a near-perfect reconstruction FID of 0.38 and 78.6% zero-shot classification accuracy on ImageNet, outperforming CLIP.
* **Limitations:** The design is more complex, requiring the training and management of multiple codebooks.

### 3.1.8 FlexTok (2025) – Flexible-Length 1D Tokens (Coarse-to-Fine)

* **Problem:** Fixed-length token sequences are inefficient, forcing a one-size-fits-all generation cost regardless of image complexity.
* **Design:** A multi-stage pipeline that produces an ordered 1D token sequence. Nested dropout during training forces the earliest tokens to encode the most salient, high-level information. A decoder can then reconstruct a plausible image from any prefix of the sequence.
* **Results:** Enables an adaptive, coarse-to-fine generation process. A recognizable image can be generated from a few tokens and progressively refined by generating more, providing a "budget knob" for AR generation.
* **Limitations:** The overall system is complex, involving a VAE, a transformer, and a rectified flow decoder. The notion of "image complexity" that determines token length is implicit.

### 3.1.9 Semanticist (2025) – Semantic-First Tokenization with Diffusion Decoding

* **Problem:** Standard tokenizers entangle high-level semantics with low-level spectral details, making their representations inefficient and hard to interpret.
* **Design:** Imposes a PCA-like structure on a 1D token sequence using a nested CFG training strategy, ensuring tokens are ordered by importance. It uses a diffusion-based decoder to explicitly decouple semantic content (low-frequency) from spectral detail (high-frequency).
* **Results:** Achieved state-of-the-art reconstruction FID while creating a highly interpretable latent space. AR models required significantly fewer tokens for high-quality generation, and the tokens showed strong semantic properties for downstream tasks.
* **Limitations:** The diffusion decoder is slower than a standard VQ decoder, increasing computational load during inference.

## 3.2 Other Notable Advances in Tokenization

* **HITA (2025):** Introduced a holistic-to-local design where a few "holistic" tokens capture global context before patch tokens are generated. This improved image coherence and accelerated AR model training.
* **XQ-GAN (2024):** An open-source, extensible tokenizer framework that combines multiple quantization methods (residual, product, multi-scale) and robust training techniques like codeword dropout.
* **IBQ (2024):** A technical breakthrough enabling the training of massive codebooks (e.g., 262k entries) by backpropagating gradients through the discrete code selection, overcoming "codebook collapse."
* **SeQ-GAN (2024):** Focused on a two-phase training objective: first, a semantic-aware loss to ensure tokens capture global content, followed by GAN finetuning to restore detail.
* **VGQ (2025):** An experimental tokenizer that represents image patches with parametric 2D Gaussians instead of pixel grids, aiming to make tokens more geometrically aware of shape and position.
* **TokenFlow (2025):** A dual-codebook architecture where one codebook captures semantics (aligned with CLIP) and the other captures pixel details, linked by a shared index for unified understanding and generation.

---

### References

(Note: This list is comprehensive, including all sources mentioned in the detailed analysis.)

Bachmann, R., et al. (2025). FlexTok: A Trainable Variable-Length Tokenizer for Language-Interfaced Vision. *Project Page available at flextok.epfl.ch*.

Esser, P., Rombach, R., & Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR).

Gu, J., et al. (2024). SeQ-GAN: Semantic-Quantized GAN for High-Fidelity Image Generation. *arXiv preprint arXiv:2404.03043*.

Li, Y., Geng, S., Li, B., Liu, F., Yang, F., et al. (2023). CM3leon: A Multi-modal, Causal-masked, Multi-task Model-in-the-Loop. *arXiv preprint arXiv:2305.07295*.

Ma, Y., Zhang, X., Li, H., & Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. *Project Page available at foundationvision.github.io*.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., et al. (2021). Zero-Shot Text-to-Image Generation. In *Proceedings of the 38th International Conference on Machine Learning* (ICML).

Shi, T., et al. (2024). Index Backpropagation for High-Performance VQ-VAE. *arXiv preprint arXiv:2404.05877*.

Sun, J., Wang, R., Li, Y., Wang, W., & Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. *arXiv preprint arXiv:2404.09344*.

van den Oord, A., & Vinyals, O. (2017). Neural discrete representation learning. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. *Project Page available at visual-gen.github.io*.

Yu, J., Wang, C., Wang, X., & Wang, J. (2024). An Image is Worth 32 Tokens: A Transformer-based 1D Tokenizer. *arXiv preprint arXiv:2404.05923*.

Zheng, Y., et al. (2025). HITA: A Holistic Image Tokenizer for Autoregressive Generation. *arXiv preprint arXiv:2501.07689*.

# 4. A Taxonomy of Design Choices in Visual Tokenization

To build a causal understanding of how VQ tokenizers function, we can analyze them through a structured framework: how specific **Design Choices** lead to certain **Observed Properties** in the trained tokenizer, which in turn have downstream **Effects** on the autoregressive (AR) model's performance and capabilities. This section systematically deconstructs the design space of visual tokenizers, providing a framework to understand why each decision matters.

## 4.1 A Framework for Tokenizer Design

We organize this taxonomy by five major design aspects: training data, sequence structure, encoder causality, token flexibility, and semantic guidance.

### 4.1.1 Training Data Mixture & Augmentations

* **Design Choice:** The composition of the dataset used to train the tokenizer, including its domain (e.g., ImageNet vs. broad web data), class distribution, and any oversampling or data augmentation strategies applied.
* **Why it matters:** The tokenizer will be biased to best represent the kinds of images it sees most frequently. A tokenizer trained only on object-centric datasets like ImageNet may lack the codebook capacity for concepts like legible text or nuanced facial expressions. Conversely, a tokenizer trained on a diverse web-scale dataset will learn a more universal codebook.
* **Observed Properties:**
    * **Codebook Usage:** A tokenizer trained on narrow data may exhibit lower codebook utilization, as the variety of visual patterns is limited.
    * **Subset-Specific Reconstruction Quality:** The tokenizer's performance will vary across different image categories. For instance, if faces are underrepresented in training, the tokenizer will likely have a higher reconstruction error on faces. Meta's Chameleon team actively countered this by **doubling the proportion of face images in training**, which resulted in the tokenizer allocating more capacity to fine facial features.
    * **Out-of-Distribution (OOD) Reconstruction:** A tokenizer trained on a specific domain (e.g., natural images) will struggle to reconstruct OOD inputs like medical X-rays, approximating them with its vocabulary of "natural" codes. A broader training set yields a more robust, universal codebook.
* **Effects on AR and Other Tasks:**
    * **AR Generation Quality:** The AR model's generative capabilities are fundamentally limited by the tokenizer's vocabulary. If the tokenizer has not learned to encode certain structures (e.g., readable text), the AR model cannot generate them. This is reflected in metrics like low Text Accuracy (T-ACC) for generations.
    * **OOD Generation Robustness:** A tokenizer trained on curated data can cause the AR model to fail on unusual prompts. TokenFlow, aiming for a "unified" tokenizer, was trained on multi-domain data, which had the effect of allowing a discrete-token multimodal model to **surpass a strong vision-language baseline (LLaVA)** on understanding tasks.
    * **Forgetting during Fine-tuning:** Fine-tuning a tokenizer on a new domain can cause it to overwrite codes that were essential for the original domain, a phenomenon known as catastrophic forgetting. Using a larger or multi-codebook architecture (like UniTok) can help mitigate this.
* **Example & Takeaway:**
    Chameleon's tokenizer was trained on a curated mix of 1.4B image-text pairs. The **observed property** was strong performance on faces (due to oversampling) but poor reconstruction of small text. The **effect** was that the multimodal model struggled with OCR-based tasks. In contrast, UniTok's training on diverse data with semantic supervision led to an **observed property** of surpassing continuous VAEs and CLIP on their respective metrics.
    **Takeaway:** *The scope and distribution of training data directly shape a tokenizer’s vocabulary and its ability to capture diverse content. Achieving high fidelity on specific categories like faces or text often requires explicit oversampling or data augmentation.*

### 4.1.2 Token Sequence Type: 1D vs. 2D (Grid) and Generation Order

* **Design Choice:** Whether the tokenizer outputs a **2D grid** of tokens corresponding to image patches or a **1D sequence** with no explicit spatial structure. For 2D grids, the serialization order (e.g., raster-scan) is also a design choice.
* **Why it matters:** This choice critically impacts sequence length and how the AR model learns dependencies. A 2D grid flattened in raster order creates an arbitrary sequence where spatially distant patches can become temporally distant, making it difficult to model global relationships. A 1D approach can encode global information or order tokens by importance, breaking this rigid structure.
* **Observed Properties:**
    * **Token Frequency Distribution:** 2D grid tokenizers are more prone to a **skewed distribution**, where tokens for common backgrounds (e.g., "sky blue") appear very frequently due to patch-level redundancy. 1D holistic tokenizers may have a more balanced distribution.
    * **Reconstruction Quality:** 2D grids excel at local reconstruction. 1D tokenizers that perform high compression (like TiTok) may achieve good overall FID but exhibit higher pixel-level error on fine details, resulting in lower performance on text and face reconstruction benchmarks like TokBench.
    * **Holistic vs. Local Encoding:** 1D tokens are often **holistic**, with each token encoding a mix of global and local information. 2D tokens are inherently **local**. This is observable in token correlations: in 2D grids, neighboring tokens are highly correlated, whereas 1D holistic tokens may be more independent.
* **Effects on AR and Other Tasks:**
    * **AR Model Sequence Length:** This is the most direct effect. 2D grids produce long sequences (e.g., 256-1024 tokens), which slows generation. 1D tokenizers like TiTok can reduce this to just 32 tokens, enabling massive speedups and better scalability to higher resolutions.
    * **AR Learnability & Coherence:** 2D raster-scan order can hinder the learning of **global consistency**. As observed in HITA's experiments, a vanilla AR model can generate an image where one half is a fish and the other is a bird due to this lack of global context. 1D or hybrid approaches with holistic tokens can mitigate this.
    * **AR Model Perplexity:** A highly compressed 1D token that encapsulates complex global content likely has a higher conditional entropy, making it harder for an AR model to predict. However, hybrid models like HITA, which provide a global context token first, can actually **reduce perplexity** for the subsequent local tokens.
* **Example & Takeaway:**
    **TiTok (1D)** achieved extremely fast generation and state-of-the-art FID, but the **observed property** was weaker reconstruction of fine details. **HITA (1D+2D hybrid)** introduced holistic tokens before the 2D grid, which had the **effect** of improving global coherence and accelerating AR model convergence. **FlexTok (1D hierarchical)** produced tokens ordered by importance, which had the unique **effect** of enabling progressive, coarse-to-fine generation, a capability impossible with fixed-grid tokenizers.
    **Takeaway:** *The choice between 1D and 2D token sequences represents a trade-off between compression and a built-in structural prior. 2D grids offer a locality prior beneficial for detail, while 1D sequences offer radical compression and flexibility, enabling faster generation and learned hierarchies.*

### 4.1.3 Causal Structure in Encoder (and Decoder)

* **Design Choice:** Whether the tokenizer's encoder is architecturally designed to output tokens in a specific **causal or hierarchical order** (e.g., using a unidirectional transformer or nested dropout), rather than an unordered set.
* **Why it matters:** Imposing a causal order on the latent codes—such as a PCA-like ranking of importance—can yield more interpretable and semantically organized tokens. This structure aligns naturally with the sequential nature of AR models, potentially simplifying the learning task.
* **Observed Properties:**
    * **Token Importance Variance:** With a causal encoder, tokens exhibit **diminishing information content**. As demonstrated by Semanticist and FlexTok, the first few tokens carry the bulk of the semantic information, and subsequent tokens add finer details. This is observable as a steep drop in reconstruction error with the first few tokens.
    * **Qualitatively Different Tokens:** Hierarchical tokenizers often learn qualitatively different roles for tokens. HITA's holistic tokens were observed to encode style and overall shape, while its patch tokens handled fine detail.
    * **Graceful Degradation:** Reconstructions from hierarchical encoders **degrade gracefully** as tokens are removed from the end of the sequence, producing a blurry but semantically correct image rather than a partially missing one.
* **Effects on AR and Other Tasks:**
    * **AR Learnability:** A causal structure simplifies the AR model's task. By generating the "broad strokes" first, the model has a coherent global context on which to condition the generation of details. HITA observed a 2x convergence speedup with this approach.
    * **Global Coherence:** This design choice drastically reduces the chance of generating incoherent images, as demonstrated by HITA's ability to solve the "fish-bird" problem in inpainting.
    * **Controllability & Editability:** Hierarchical tokens enable novel forms of control. Semanticist demonstrated **style transfer** by swapping only the first few tokens between images. FlexTok allows for **on-demand, progressive generation**, where a user can generate a quick preview and request more detail if needed.
* **Example & Takeaway:**
    **Semanticist**'s design creates a PCA-like token hierarchy. The **observed property** is that dropping later tokens preserves semantic meaning. The **effect** is that AR models require fewer tokens, and the latent space is more interpretable. **FlexTok**'s use of nested dropout yields an ordered token sequence, with the **effect** of enabling AR models to generate a recognizable class object with as few as 8 tokens.
    **Takeaway:** *Imposing a causal or hierarchical order in the tokenizer’s latent space creates a powerful synergy with AR models. It leads to more coherent generation, faster training, and novel capabilities for interactive control and editing, aligning the generation process more closely with human artistic creation.*

### 4.1.4 Flexible Number of Tokens (Dynamic Length)

* **Design Choice:** Allowing the tokenizer to produce a **variable number of tokens per image**, rather than a fixed-length sequence, adapting to the image's complexity.
* **Why it matters:** Not all images are equally complex. A fixed-length tokenizer is inefficient, wasting capacity on simple images while potentially lacking enough for complex ones. A flexible token count allows for a more efficient allocation of resources, similar to variable bitrate encoding in video.
* **Observed Properties:**
    * **Rate-Distortion Behavior:** A flexible tokenizer can be evaluated along a rate-distortion curve, showing how reconstruction quality improves as the token budget increases. This allows for a continuous trade-off, unlike the single operating point of a fixed tokenizer.
    * **Adaptive Token Allocation:** An adaptive tokenizer will naturally use fewer tokens for simple images and more for complex ones. This distribution of token counts can be observed and correlated with image complexity metrics.
* **Effects on AR and Other Tasks:**
    * **Efficiency of AR Generation:** The AR model can terminate generation early for simple prompts or images, saving significant computation. This is a core feature enabled by FlexTok's design.
    * **User-Controlled Quality vs. Speed Trade-off:** This design enables interactive applications where a user can generate a fast preview and then request a higher-fidelity version by having the model generate more tokens.
    * **Improved Average Performance:** For a given average token budget, an adaptive tokenizer can allocate its resources more intelligently across a dataset, leading to better overall FID and LPIPS scores compared to a fixed-length baseline.
    * **Robustness to Complexity Extremes:** An adaptive tokenizer can handle extremely detailed images by simply generating more tokens, avoiding the "quality cliff" where a fixed tokenizer runs out of capacity and fails.
* **Example & Takeaway:**
    **FlexTok** is the prime example, where its design explicitly allows for a variable-length output. The **observed property** is that images transition from coarse outlines to full detail as the number of tokens increases from ~20 to ~300. The **effect** is the ability to perform on-demand, progressive generation with an AR model.
    **Takeaway:** *Variable token counts make a tokenizer more data-efficient and versatile. This design leads to dynamic control over the speed-quality trade-off, better utilization of modeling capacity, and more robust performance across images of varying complexity, making AR models more efficient and adaptive.*

### 4.1.5 Semantic Forcing and External Guidance

* **Design Choice:** Incorporating **semantic knowledge** into the tokenizer's training, typically by aligning its representations with features from a powerful pretrained foundation model like CLIP or DINOv2.
* **Why it matters:** A standard VQ tokenizer is only optimized for reconstruction and does not guarantee that its tokens will be semantically meaningful. Adding a semantic loss encourages the tokens to encode human-relevant concepts, which improves their utility for downstream tasks and can aid in generating more coherent compositions.
* **Observed Properties:**
    * **Semantic Clustering:** The token embeddings from a semantically-guided tokenizer will exhibit stronger clustering by semantic class.
    * **Improved Understanding Metrics:** These tokenizers show higher **zero-shot classification accuracy** or text-image retrieval scores. UniTok, for example, achieved 78.6% accuracy on ImageNet, outperforming CLIP itself.
    * **Linearly Separable Representations:** The token representations become more linearly separable, as demonstrated by the strong performance of simple linear probes on the embeddings from models like Semanticist and UniTok.
* **Effects on AR and Other Tasks:**
    * **Bridging Generation and Understanding:** The most significant effect is **closing the gap between discrete and continuous representations for understanding tasks**. As shown by TokenFlow, a multimodal model using its discrete tokens could outperform LLaVA, which uses continuous features.
    * **Improved Generative Coherence:** By encoding clear semantic concepts, these tokens may help AR models with complex compositional tasks, such as correctly binding attributes to objects.
    * **Faster Convergence:** Using a semantic loss, such as the REPA loss in FlexTok, can provide a strong, high-level learning signal that **greatly accelerates the convergence** of the tokenizer's training.
* **Example & Takeaway:**
    **UniTok**'s use of multi-codebooks and CLIP supervision resulted in tokens that excelled at both reconstruction and classification. **TokenFlow**'s dual-codebook design, with one codebook for semantics and one for detail, enabled its discrete tokens to be used in a model that surpassed a strong continuous-feature baseline in VQA.
    **Takeaway:** *Semantic guidance ensures that discrete representations carry meaningful, human-interpretable information. This enhances AR generation, enables the direct use of tokens for multimodal understanding tasks, and effectively unifies generative and discriminative representations within a single, powerful tokenizer.*

## 4.2 Application of the Framework to Key Tokenizers

### 4.2.1 VQ-VAE

* **Design Choices:** 2D grid sequence; standard convolutional encoder/decoder; key innovation was the discrete bottleneck with a straight-through estimator.
* **Observed Properties:** Small codebook size (K=512); prone to **codebook collapse**; blurry reconstructions due to MSE loss.
* **Effects:** Successfully enabled AR priors by avoiding posterior collapse, but final generation quality was limited by the poor reconstruction baseline.

### 4.2.2 VQGAN

* **Design Choices:** Maintained the 2D grid; key innovation was **semantic forcing** via a PatchGAN adversarial loss and a perceptual (LPIPS) loss.
* **Observed Properties:** Supported larger codebooks (up to 16,384); dramatically improved reconstruction quality (rFID), resulting in sharp, textured images.
* **Effects:** Provided a high-fidelity foundation that made megapixel-scale AR generation practical; adversarial training improved decoder robustness.

### 4.2.3 LlamaGen

* **Design Choices:** A scaling achievement using a 2D grid and a standard Llama architecture with 2D positional embeddings.
* **Observed Properties:** High codebook utilization (97%); strong reconstruction quality (rFID of 0.94) from its improved VQGAN-style tokenizer.
* **Effects:** Proved that standard LLM architectures scale effectively for visual token modeling, achieving a gFID of 2.18 that surpassed leading diffusion models.

### 4.2.4 TiTok

* **Design Choices:** Key innovation was shifting to a highly compressed **1D sequence** using a Transformer-based encoder with learnable queries.
* **Observed Properties:** Extremely low, fixed token count (e.g., 32); good overall reconstruction (rFID 2.21) but traded some fine-detail fidelity for compression.
* **Effects:** A massive acceleration in AR inference speed (hundreds of times faster); simplified the AR learning task; achieved excellent gFID (1.97).

### 4.2.5 FlexTok

* **Design Choices:** A **1D ordered sequence** with **variable length**, enabled by nested dropout; employed **semantic forcing** via a rectified flow decoder and a REPA (DINOv2 alignment) loss.
* **Observed Properties:** An emergent **hierarchical token structure** (coarse-to-fine); demonstrated adaptive compression based on image complexity.
* **Effects:** Enabled **adaptive and progressive generation**, allowing for a dynamic trade-off between speed and quality.

### 4.2.6 Semanticist

* **Design Choices:** A **1D causal sequence** with a mathematically guaranteed **PCA-like structure**, enforced by a nested CFG strategy and a diffusion-based decoder.
* **Observed Properties:** A highly structured latent space with **semantic-spectrum decoupling**; tokens are ordered by importance, contributing orthogonal information.
* **Effects:** The semantically pure "language" is very easy for an AR model to learn; tokens are highly effective for downstream classification (63.5% accuracy from a linear probe) and provide a more interpretable representation.

## 4.3 Comparison of Key Visual Tokenizers

This table provides a comparative overview of the key VQ tokenizers discussed, summarizing their architectural designs, observed performance properties, and their downstream effects on autoregressive models and related tasks.

| Tokenizer (Year) | Sequence & Tokens (256px) | Key Innovation | Semantic Guidance / Loss | rFID (Recon.) | Key Properties | Key Effects & Gen. FID | Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Tokenizers** |
| **VQ-VAE** (2017) | 2D Grid (e.g., 32x32) | Foundational discrete VAE bottleneck. | Pixel-wise MSE | High (~15-20), blurry | Small codebook (512-1024), prone to **codebook collapse**. | Enabled AR models on discrete codes but with limited generation quality. | [Multiple Implementations](https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py) |
| **VQGAN** (2021) | 2D Grid (e.g., 16x16) | Augments VQ-VAE with GAN + perceptual losses. | Adversarial + LPIPS Loss | Low (~5-8), sharp | Larger codebook (1024-16,384), captures high-frequency patterns. | Enabled high-fidelity, photorealistic AR generation. Became foundational. | [Official Code & Models](https://github.com/CompVis/taming-transformers) |
| **DALL·E dVAE** (2021)| 2D Grid (32x32) / 1024 Tokens | Scaled for massive text-to-image training. | Perceptual Loss (inferred) | Excellent (reported ~0.13) | Large codebook (8,192), high code usage due to massive dataset. | Powered DALL·E 1, proving AR models can handle complex prompts. | [Tokenizer Weights Available](https://github.com/openai/dalle-2) |
| **LlamaGen VQ** (2024)| 2D Grid (e.g., 24x24) | Careful scaling of VQGAN for a large AR model. | Perceptual Loss | Very low (0.94) | Large codebook (16,384), high usage (~97%). | Enabled SOTA **gFID (~2.18)**, proving scaled AR can outperform diffusion. | [Official Code & Models](https://github.com/FoundationVision/LlamaGen) |
| **Chameleon** (2023) | 2D Grid (32x32) / 1024 Tokens | Designed for early-fusion multimodal models. | Perceptual Loss + Biased training data (oversampled faces). | N/A | Improved face reconstruction; struggled with small text/OCR. | Enabled a single model for interleaved image/text generation & understanding. | [Model on HuggingFace](https://huggingface.co/facebook/chameleon-7b) |
| **TiTok** (2024) | 1D Sequence / 32 Tokens | Ultra-compact representation via Transformer encoder. | Perceptual Loss | Good (2.21) | Sacrifices fine detail (text/faces) for extreme compression. | Drastically accelerates AR gen. (>400x). Achieved SOTA **gFID (1.97)**. | [Official Code & Demo](https://yucornetto.github.io/TiTok/) |
| **SoftVQ** (2024) | 1D Continuous Sequence | "Soft" quantization (weighted mixture of codewords). | DINO alignment loss | FID 1.78 (with DiT) | High representational capacity per token; fully differentiable. | Accelerates training and inference (>18x faster). Loses discrete indexing. **gFID 1.78** (with DiT). | [Official Code & Models](https://github.com/kakaobrain/soft-vq) |
| **XQ-GAN** (2024) | 2D Grid | Extensible framework combining multiple quantization methods (RQ, PQ, etc.). | Supports CLIP/DINOv2 alignment. | Excellent (0.64) | Highly flexible, allowing trade-offs between different objectives. | Strong open-source baseline. Used with VAR model to achieve **gFID of 2.60**. | [Official Code & Models](https://github.com/kent-lcc/XQ-GAN) |
| **UniTok** (2025) | 2D Grid with 1D codes | Multi-codebook quantization to expand vocabulary. | Combined reconstruction + CLIP objectives. | SOTA (0.38) | Semantically rich (78.6% zero-shot acc.), preserves fine details. | Unified generation and understanding; tokens are powerful features. | [Official Code & Models](https://github.com/FoundationVision/UniTok) |
| **FlexTok** (2025) | 1D Ordered Seq. / Variable (8-256) | Variable-length tokens via nested dropout. | REPA (DINOv2) loss on rectified flow decoder. | Variable | Hierarchical tokens (coarse-to-fine). | Enables adaptive, progressive AR generation; user can trade speed vs. quality. | [Official Code & Project Page](https://flextok.epfl.ch/) |
| **Semanticist** (2025)| 1D Causal Seq. | Enforces a PCA-like structure on tokens; diffusion decoder. | Nested CFG training for semantic-spectrum decoupling. | SOTA (0.72) | Highly structured, interpretable latent space. Strong linear separability (63.5% acc.). | Simplifies AR learning (**gFID 2.57** with 32 tokens). Enables semantic editing. | [Project Page](https://visual-gen.github.io/semanticist/) |
| **Extended Tokenizers** |
| **HITA** (2025) | Hybrid 1D+2D (Holistic + Patch) | Generates global "holistic" tokens first for context. | Injects DINOv2 features into holistic tokens. | N/A | Holistic tokens capture global semantics. AR model trains ~2x faster. | Improves global coherence; enables zero-shot inpainting/style transfer. | [Official Code & Models](https://github.com/CVMI-Lab/Hita) |
| **TokenFlow** (2025) | 2D Grid | Dual-codebook architecture (semantic + pixel) with a shared index. | Semantic codebook aligned with CLIP. | Excellent (0.63 @ 384px) | Decoupled representation for semantics and detail. | First discrete tokenizer to enable a model to surpass LLaVA on understanding tasks. | [Official Code (Announced)](https://github.com/ByteFlow-AI/TokenFlow) |
| **IBQ** (2024) | 2D Grid | Index Backpropagation for training massive codebooks without collapse. | Perceptual Loss | Extremely high quality | Near-perfect codebook utilization (~96%) on huge codebooks (up to 262k). | Removes codebook size bottleneck. AR models achieved **gFID of ~2.05**. | [Official Code & Models](https://github.com/TencentARC/SEED-Voken) |
| **GigaTok** (2025) | 1D Sequence | A 2.9B parameter tokenizer, proving scalability. | Semantic regularization with DINOv2. | SOTA | Semantically organized latent space even at massive scale. | SOTA AR generation (**gFID ~1.7-2.0**). Scaling tokenizer improves both gen & understanding. | [Official Code & Models](https://github.com/silentview/GigaTok) |


### References

Esser, P., Rombach, R., & Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR).

Gu, J., et al. (2024). SeQ-GAN: Semantic-Quantized GAN for High-Fidelity Image Generation. *arXiv preprint arXiv:2404.03043*.

Li, Y., et al. (2024). TokenFlow: A Unified Tokenizer for Vision and Language. *arXiv preprint arXiv:2412.03069*.

Ma, Y., Zhang, X., Li, H., & Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. *Project Page available at foundationvision.github.io*.

Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. In *Advances in Neural Information Processing Systems 32* (NeurIPS 2019).

Sun, J., Wang, R., Li, Y., Wang, W., & Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. *arXiv preprint arXiv:2404.09344*.

Tian, Y., et al. (2024). VAR: Visual Autoregressive Modeling with Scalable Tokenization. In *International Conference on Learning Representations* (ICLR 2025).

TokBench Collaborators. (2025). TokBench: A Comprehensive Benchmark for Visual Tokenizers. *arXiv preprint arXiv:2505.18142v2*.

Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. *Project Page available at visual-gen.github.io*.

Zheng, Y., et al. (2025). HITA: A Holistic Image Tokenizer for Autoregressive Generation. *arXiv preprint arXiv:2507.02358v1*.

# 5. Frontiers, Tensions, and Emerging Themes

This final section outlines the key challenges, emerging themes, and research gaps that will likely shape the next generation of models.

## 5.1 Key Tensions and Foundational Trade-offs

The design of visual tokenizers is governed by a set of competing priorities. These core tensions represent the fundamental trade-offs that researchers must navigate.

* **The Reconstruction vs. Generation Dilemma:** This remains the most critical tension. As highlighted by GigaTok, naively scaling a tokenizer to achieve better reconstruction fidelity (lower rFID) can produce an exponentially more complex latent space. This makes the distribution of token sequences harder for the AR model to learn, leading to higher perplexity and, paradoxically, worse final generation quality (higher gFID). Managing this trade-off is a central design challenge.

* **Sequence Length vs. Information Richness:** There is an ongoing debate between using a short sequence of highly information-dense tokens (TiTok) versus a longer sequence of simpler tokens. While short sequences offer massive efficiency gains, longer sequences may be necessary for complex prompts that require fine-grained detail (FlexTok). The optimal balance is unclear and may be task-dependent.

* **Unified vs. Specialized Tokenizers:** The push towards unified models (UniTok, TokenFlow) that serve both understanding and generation is a compelling research direction. However, this introduces complexity (e.g., dual codebooks, multi-objective losses) and raises the question of a **generalization vs. specialization tension**. It remains an open question whether a single "jack-of-all-trades" tokenizer can consistently outperform specialized models. For instance, a tokenizer optimized purely for artistic generation might prioritize texture fidelity over semantic accuracy, a trade-off a unified model might not make.

## 5.2 Open Research Questions and Future Directions

Several open questions and research gaps remain, pointing toward exciting avenues for future work.

### 5.2.1 Out-of-Distribution (OOD) Robustness

The vast majority of research evaluates tokenizers on in-distribution datasets. There is a significant lack of investigation into how these models perform on OOD images (e.g., medical imagery, abstract art, sketches), their robustness to common corruptions, or their vulnerability to adversarial attacks. Future work could introduce adaptive codebooks or fine-tuning methods to adapt tokenizers to new domains without catastrophic forgetting.

### 5.2.2 Compositionality, Disentanglement, and Spatial Reasoning

AR models still falter on prompts requiring complex spatial arrangements or precise counting. This may stem from tokenizing by arbitrary patches rather than semantic entities. An emerging research direction is **object-centric and disentangled tokenization**. Instead of a single token trying to represent shape, texture, and semantics all at once, future tokenizers might produce multiple, parallel token streams—one for geometry, one for appearance, one for high-level concepts. This could involve channel-wise VQ or hybrid VQ-segmentation models to align tokens more closely with how humans describe scenes—by objects and their relations.

### 5.2.3 Hybrid Autoregressive and Diffusion Models

The line between AR and diffusion paradigms is blurring. Semanticist's use of a diffusion decoder hints at powerful hybrid models: AR for the high-level, structured sequence of concepts, and diffusion for the low-level pixel synthesis. This opens up possibilities for multi-stage generation: an AR model could generate a coarse token sequence, which is then refined by a diffusion model or even a second "error correction" AR model. Understanding the optimal way to combine the speed and structure of AR with the refinement capabilities of diffusion is a key frontier.

### 5.2.4 Multi-Scale and Hierarchical Generation

Beyond a single sequence, **multi-scale tokenization** offers a promising path. Models like VAR have explored tokenizing an image at multiple resolutions, allowing an AR model to predict a sequence of scale-wise tokens. This could enable more efficient handling of both global layouts (via coarse tokens) and fine details (via fine-grained tokens), potentially improving both quality and generation speed for high-resolution images.

### 5.2.5 Multi-Modality and Unified Models

While models like TokenFlow have begun to unify vision and text, the next frontier is incorporating other modalities like audio, video, and 3D. An open challenge is creating a single tokenizer and AR model that can process and generate arbitrarily interleaved sequences of image, text, and audio tokens, as explored by early work like AToken. This raises fundamental questions about whether a universal codebook is feasible or if modality-specific vocabularies are necessary.

### 5.2.6 Tokenization for Fine-Tuning and Personalization

Diffusion models have powerful personalization techniques like DreamBooth. Developing analogous methods for VQ-AR models is a key open challenge. This might involve learning new codebook entries for a specific object or style and fine-tuning the AR model to use these new "visual words." Research into few-shot learning with discrete visual vocabularies is needed to enable efficient personalization without catastrophic forgetting.

## 5.3 Emerging Architectural and Conceptual Themes

Several powerful concepts are emerging that redefine the role of the tokenizer and its components.

* **Structured Latent Spaces:** There is a clear trend away from an unstructured "bag of tokens" toward sequences with inherent order and meaning. The hierarchical structures in FlexTok and the mathematically-guaranteed, PCA-like orthogonality in Semanticist are prime examples. Future research will likely explore more sophisticated structures, such as graph-based representations, to make the latent space even more interpretable and efficient.

* **The Evolving Role of the Decoder:** The tokenizer's decoder is no longer just a simple upsampling network. In VQGAN, it became an adversarial player. In FlexTok, it is a rectified flow model capable of decoding partial sequences. In Semanticist, it is a diffusion model used to enforce semantic-spectral decoupling. The decoder is increasingly being used as an active tool during training to impose desirable structural and semantic properties onto the latent space itself.

* **Continuous vs. Discrete Revisited:** While discrete VQ tokenizers have dominated, models like SoftVQ are re-introducing continuous representations. By allowing a token to be a "soft" mixture of multiple codebook entries, these models aim to increase representational capacity and avoid information loss from hard quantization. The future may lie in hybrid approaches that combine the structured nature of discrete tokens with the expressive power of continuous representations.

## 5.4 Gaps in Methodology and Evaluation

Progress in the field is hampered by several gaps in how models are developed and evaluated.

* **Standardizing Evaluation:** There is a critical need for more comprehensive and standardized benchmarks. This includes **OOD benchmarks** that test generalization beyond natural images, **compositional metrics** to evaluate spatial accuracy and object counting, and a more rigorous protocol for **FID reporting** to ensure comparability across publications.
* **AR Learnability Metrics:** The field lacks a standard metric for comparing the "language complexity" each tokenizer produces. The "AR probing" approach from GigaTok, which evaluates a fixed, lightweight AR model on different tokenizers, could provide a much-needed, apples-to-apples comparison of AR learnability.
* **Theoretical Understanding:** The theoretical frameworks for predicting tokenizer performance from design choices remain limited. A deeper theoretical understanding could accelerate progress and reduce reliance on empirical trial-and-error.
* **Practical and Methodological Gaps:** Key factors like **training stability**, **computational efficiency**, and **architectural compatibility** with different downstream models are often under-reported but are crucial for practical adoption. Furthermore, **human perceptual alignment** remains an underexplored evaluation dimension that could reveal biases in our current automated metrics.

## 5.5 The Future of Visual Language

The journey of VQ tokenizers—from VQ-VAE’s blurry patches to the semantic-rich, flexible sequences of today—has been remarkable. The field is converging on a powerful "visual sentence" metaphor. The progression from 2D grids (a "page" of tokens) to 1D sequences (a simple "sentence") and now to ordered, hierarchical sequences (a "grammatically structured sentence") indicates a drive to create a true, compositional language of images.

As the tokenizer becomes more adept at creating this structured language, the role of the AR model may shift from that of a raw creator to a logical synthesizer, tasked with arranging these meaningful visual components into a coherent narrative. Answering the open questions outlined above will likely involve an interplay of ideas: merging paradigms (AR + diffusion hybrids), drawing from NLP techniques for images, and rethinking the very notion of visual “tokens” to be more aligned with human perception. The coming years will determine if autoregressive image generation, armed with these advanced tokenizers, can truly become the dominant approach for generative AI.

---

### References

(Note: This list combines all unique sources cited in the provided text for this section.)

Chang, H., et al. (2023). MaskGIT: Masked Generative Image Transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR).

Gao, P., et al. (2024). ACDiT: Autoregressive Conditional Diffusion Transformers for High-Fidelity Image Synthesis. *arXiv preprint arXiv:2404.09559*.

Li, Y., et al. (2024). TokenFlow: A Unified Tokenizer for Vision and Language. *arXiv preprint arXiv:2412.03069*.

Ma, Y., Zhang, X., Li, H., & Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. *Project Page available at foundationvision.github.io*.

Sun, J., Wang, R., Li, Y., Wang, W., & Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. *arXiv preprint arXiv:2404.09344*.

Tian, Y., et al. (2024). VAR: Visual Autoregressive Modeling with Scalable Tokenization. In *International Conference on Learning Representations* (ICLR 2025).

TokBench Collaborators. (2025). TokBench: A Comprehensive Benchmark for Visual Tokenizers. *arXiv preprint arXiv:2505.18142v2*.

Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. *Project Page available at visual-gen.github.io*.

Yu, Z., et al. (2025). GigaTok: A Billion-Parameter Visual Tokenizer. *Project Page available at silentview.github.io*.
