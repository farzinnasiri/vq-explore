---


---

<h1 id="background-from-pixels-to-tokens">0. Background: From Pixels to Tokens</h1>
<p>Autoregressive (AR) models, which predict the next element in a sequence based on preceding elements, have achieved remarkable success in natural language processing (NLP). This paradigm is a natural fit for text, which is an inherently discrete, one-dimensional sequence. However, applying this same logic to image generation presents a series of fundamental challenges. This section explores these challenges, contrasts the autoregressive approach with the dominant diffusion paradigm, and introduces vector quantization (VQ) as the critical technology that bridges this gap, effectively creating a new “language” for images.</p>
<h2 id="the-challenge-why-autoregressive-models-are-harder-for-images-than-text">0.1 The Challenge: Why Autoregressive Models are Harder for Images than Text</h2>
<p>At its core, an AR model factorizes the joint probability distribution of a sequence <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>x</mi><mi>N</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">x = (x_1, x_2, \ldots, x_N)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">x</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner">…</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.328331em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.10903em;">N</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span> using the chain rule of probability:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mi mathvariant="normal">∣</mi><msub><mi>x</mi><mrow><mo>&lt;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">
p(x) = \prod_{i=1}^{N} p(x_i | x_{&lt;i})
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 3.10601em; vertical-align: -1.27767em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.82834em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mrel mtight">=</span><span class="mord mtight">1</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∏</span></span></span><span class="" style="top: -4.30001em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.10903em;">N</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.27767em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mord">∣</span><span class="mord"><span class="mord mathnormal">x</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mrel mtight">&lt;</span><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.17737em;"><span class=""></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span></span></p>
<p>This “next-token prediction” approach, which powers models like GPT (Radford et al., 2019), is fundamentally challenged when the data is not a simple 1D sequence.</p>
<ul>
<li>
<p><strong>The Dimensionality Curse:</strong> Images are high-dimensional, continuous signals. A modest 256x256 pixel image, when flattened, results in a sequence of 65,536 pixels, each with three color channels. Modeling such an incredibly long sequence pixel-by-pixel, as early models like PixelRNN attempted (van den Oord et al., 2016), is computationally prohibitive and forces the model to learn extremely long-range dependencies.</p>
</li>
<li>
<p><strong>Lack of Natural Ordering:</strong> Unlike text, images are 2D grids with strong spatial correlations in all directions. There is no intrinsically “correct” way to flatten an image into a 1D sequence. The conventional approach, a raster scan (row-by-row), imposes an arbitrary sequential order. This makes it difficult for the model to capture complex spatial relationships, as spatially close pixels in the 2D grid can become very distant in the 1D sequence, breaking long-range coherence.</p>
</li>
<li>
<p><strong>Spatial Redundancy:</strong> Natural images contain significant spatial redundancy, such as large, uniform areas of color or texture. A pixel-level AR model must expend considerable capacity learning to predict these redundant patterns, which is an inefficient use of model parameters.</p>
</li>
</ul>
<h2 id="competing-paradigms-autoregressive-vs.-diffusion-models">0.2 Competing Paradigms: Autoregressive vs. Diffusion Models</h2>
<p>Before the recent resurgence of AR models, the dominant paradigm in high-fidelity image generation was diffusion models. Understanding their differences is key to appreciating the motivations behind modern VQ-AR systems.</p>
<ul>
<li>
<p><strong>Core Mechanism Contrast:</strong></p>
<ul>
<li><strong>Autoregressive (AR) Models</strong> operate <strong>sequentially</strong>. They generate an image token by token, where each prediction is conditioned on all previously generated tokens. This is analogous to a writer composing a sentence one word at a time. This unidirectional process can lead to error propagation, where an early mistake impacts the entire subsequent generation.</li>
<li><strong>Diffusion Models</strong> operate <strong>holistically and iteratively</strong>. The process begins with random noise and gradually refines the entire image over dozens of steps by predicting and removing noise (Ho et al., 2020). This iterative refinement allows the model to correct mistakes and achieve global consistency.</li>
</ul>
</li>
<li>
<p><strong>Key Trade-offs:</strong></p>
<ul>
<li><strong>Speed vs. Quality:</strong> AR models are substantially faster at inference, requiring only one forward pass per generated token. Diffusion models must perform many full-image passes, making them slower. However, this iterative process is why diffusion models have traditionally excelled in generating photorealistic, highly detailed images.</li>
<li><strong>Scalability and Simplicity:</strong> AR models leverage the next-token prediction paradigm, which has well-understood scaling properties from NLP. The challenge, therefore, becomes one of representation: if one can represent images as sequences of “visual tokens,” AR Transformers might scale for vision the way GPT scaled for text (Yu et al., 2022).</li>
</ul>
</li>
</ul>
<h2 id="the-bridge-vector-quantization-for-image-tokenization">0.3 The Bridge: Vector Quantization for Image Tokenization</h2>
<p>To overcome the challenges of applying AR models to continuous, high-dimensional images, the field adopted a crucial intermediate step: <strong>vector quantization (VQ)</strong>. A VQ-based tokenizer serves as a bridge, converting a continuous image into a discrete sequence of tokens, effectively creating a “language” for images that AR models can generate.</p>
<p>This idea was pioneered by the Vector-Quantised Variational Autoencoder (VQ-VAE), which integrated a learnable discrete codebook into an autoencoder framework (van den Oord &amp; Vinyals, 2017). The process works as follows:</p>
<ol>
<li>An <strong>encoder</strong> maps patches of the input image to continuous vectors.</li>
<li>Each vector is then “quantized” by replacing it with the nearest vector from a finite, learned <strong>codebook</strong>.</li>
<li>The sequence of indices of these codebook vectors becomes the discrete token representation of the image.</li>
<li>A <strong>decoder</strong> reconstructs the image from this sequence of discrete tokens.</li>
</ol>
<p>By converting an image from a grid of 65,536 pixels into, for example, a 16x16 grid of 256 tokens, VQ dramatically shortens the sequence length and forces the model to learn a vocabulary of meaningful visual patterns rather than raw pixel values. This tokenized representation is then suitable for a powerful AR model, such as a Transformer, to learn its distribution.</p>
<p>To make this concept more intuitive, consider these analogies:</p>
<ul>
<li><strong>The LEGO Set:</strong> A real-world object has continuous surfaces. A VQ tokenizer acts like a system that represents this object using a finite set of standard LEGO bricks (the codebook). The reconstructed object is built by assembling these discrete bricks according to a blueprint—the sequence of token IDs.</li>
<li><strong>The Cookbook:</strong> Visual patterns like textures and edges are like complex flavors. A VQ codebook is a cookbook with a finite number of “flavor profile” recipes. The tokenizer analyzes a patch of the image, finds the closest matching recipe, and represents the image as a sequence of these recipe IDs.</li>
</ul>
<p>By successfully representing images as discrete token sequences, VQ enables a new class of powerful and efficient generative models. Evaluating the quality of these tokenized representations and the final generated images requires a specialized set of metrics, which are detailed in the following section.</p>
<hr>
<h3 id="references">References</h3>
<p>Ho, J., Jain, A., &amp; Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In <em>Advances in Neural Information Processing Systems 33</em> (NeurIPS 2020).</p>
<p>Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., &amp; Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. <em>OpenAI Blog</em>, 1(8), 9.</p>
<p>van den Oord, A., Kalchbrenner, N., &amp; Kavukcuoglu, K. (2016). Pixel recurrent neural networks. In <em>Proceedings of the 33rd International Conference on Machine Learning</em> (ICML).</p>
<p>van den Oord, A., &amp; Vinyals, O. (2017). Neural discrete representation learning. In <em>Advances in Neural Information Processing Systems 30</em> (NIPS 2017).</p>
<p>Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., … &amp; Le, Q. V. (2022). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. <em>arXiv preprint arXiv:2206.10789</em>.</p>
<h1 id="evaluation-metrics-in-image-generation">1. Evaluation Metrics in Image Generation</h1>
<p>The evaluation of generative models, particularly in the domain of image synthesis, is a multifaceted challenge. The quality of a generated image can be assessed from several perspectives: its fidelity as an independent sample, its realism compared to the distribution of real-world images, and its faithfulness to a specific reference image. Consequently, a variety of metrics have been developed, each designed to capture different aspects of performance. These metrics can be broadly categorized based on whether the generation task is content-variant, allowing for multiple correct outputs, or content-invariant, where there is a single ground-truth target. Further metrics have been developed specifically for models employing vector quantization, focusing on the efficiency and representational quality of the learned discrete codebooks.</p>
<h2 id="content-variant-metrics">1.1 Content-Variant Metrics</h2>
<p>Content-variant metrics are essential for tasks where the generative model is expected to produce a diverse range of outputs from a given input, such as unconditional image generation from a noise vector. In these scenarios, there is no single “correct” image, so evaluation must focus on the overall quality and diversity of the generated distribution.</p>
<h3 id="inception-score-is">1.1.1 Inception Score (IS)</h3>
<p>The Inception Score (IS) was one of the first widely adopted metrics for quantitatively assessing the performance of generative models (Salimans et al., 2016). It is designed to simultaneously measure two key properties: the quality of individual images (realism) and the diversity of the set of generated images.</p>
<p>The intuition behind IS is twofold. First, a high-quality, realistic image should be easily classifiable by a pre-trained image classifier like Inception-v3. This means the conditional probability distribution <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mi mathvariant="normal">∣</mi><mi>x</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">p(y|x)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mord">∣</span><span class="mord mathnormal">x</span><span class="mclose">)</span></span></span></span></span> over the labels <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>y</mi></mrow><annotation encoding="application/x-tex">y</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span></span></span></span></span> for a given generated image <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>x</mi></mrow><annotation encoding="application/x-tex">x</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">x</span></span></span></span></span> should have low entropy; the model should be confident about what object is in the image. Second, for the generated set to be diverse, the model should produce a wide variety of objects. Therefore, the marginal probability distribution over all labels, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mo>∫</mo><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mi mathvariant="normal">∣</mi><mi>x</mi><mo>=</mo><mi>G</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mi>d</mi><mi>z</mi></mrow><annotation encoding="application/x-tex">p(y) = \int p(y|x=G(z))dz</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.11112em; vertical-align: -0.30612em;"></span><span class="mop op-symbol small-op" style="margin-right: 0.19445em; position: relative; top: -0.00056em;">∫</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mord">∣</span><span class="mord mathnormal">x</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">G</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.04398em;">z</span><span class="mclose">))</span><span class="mord mathnormal">d</span><span class="mord mathnormal" style="margin-right: 0.04398em;">z</span></span></span></span></span>, should have high entropy, indicating a uniform distribution of classes.</p>
<p>The Inception Score combines these two ideas using the Kullback-Leibler (KL) divergence, calculated as:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>IS</mtext><mo stretchy="false">(</mo><mi>G</mi><mo stretchy="false">)</mo><mo>=</mo><mi>exp</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><msub><mi mathvariant="double-struck">E</mi><mrow><mi>x</mi><mo>∼</mo><msub><mi>p</mi><mi>g</mi></msub></mrow></msub><mo stretchy="false">[</mo><msub><mi>D</mi><mrow><mi>K</mi><mi>L</mi></mrow></msub><mo stretchy="false">(</mo><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mi mathvariant="normal">∣</mi><mi>x</mi><mo stretchy="false">)</mo><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∣</mi><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo fence="true">)</mo></mrow></mrow><annotation encoding="application/x-tex">
\text{IS}(G) = \exp\left(\mathbb{E}_{x \sim p_g} [D_{KL}(p(y|x) || p(y))]\right)
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord text"><span class="mord">IS</span></span><span class="mopen">(</span><span class="mord mathnormal">G</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.20001em; vertical-align: -0.35001em;"></span><span class="mop">exp</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner"><span class="mopen delimcenter" style="top: 0em;"><span class="delimsizing size1">(</span></span><span class="mord"><span class="mord mathbb">E</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">x</span><span class="mrel mtight">∼</span><span class="mord mtight"><span class="mord mathnormal mtight">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.34732em;"><span class=""></span></span></span></span></span></span><span class="mopen">[</span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.02778em;">D</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.328331em;"><span class="" style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">K</span><span class="mord mathnormal mtight">L</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mord">∣</span><span class="mord mathnormal">x</span><span class="mclose">)</span><span class="mord">∣∣</span><span class="mord mathnormal">p</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">))]</span><span class="mclose delimcenter" style="top: 0em;"><span class="delimsizing size1">)</span></span></span></span></span></span></span></span></p>
<p>A higher IS indicates that the generated images are both individually distinct (low conditional entropy) and collectively diverse (high marginal entropy).</p>
<h3 id="fréchet-inception-distance-fid">1.1.2 Fréchet Inception Distance (FID)</h3>
<p>While IS measures quality and diversity, it does not directly compare the generated distribution to the real data distribution. The Fréchet Inception Distance (FID) addresses this limitation by measuring the distance between the feature distributions of real and generated images (Heusel et al., 2017).</p>
<p>The process involves embedding both a set of real images and a set of generated images into a feature space using a pre-trained Inception-v3 model. The activations from a specific layer (typically the final pooling layer) are collected for each set. These collections of feature vectors are then modeled as multivariate Gaussian distributions. FID calculates the Wasserstein-2 distance between these two Gaussians.</p>
<p>Given the mean (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>μ</mi><mi>r</mi></msub></mrow><annotation encoding="application/x-tex">\mu_r</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord"><span class="mord mathnormal">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>μ</mi><mi>g</mi></msub></mrow><annotation encoding="application/x-tex">\mu_g</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.716668em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathnormal">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span>) and covariance matrices (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>r</mi></msub></mrow><annotation encoding="application/x-tex">\Sigma_r</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi mathvariant="normal">Σ</mi><mi>g</mi></msub></mrow><annotation encoding="application/x-tex">\Sigma_g</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.969438em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span>) of the real and generated feature distributions, the FID is calculated as:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>FID</mtext><mo stretchy="false">(</mo><mi>r</mi><mo separator="true">,</mo><mi>g</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∣</mi><msub><mi>μ</mi><mi>r</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>g</mi></msub><mi mathvariant="normal">∣</mi><msubsup><mi mathvariant="normal">∣</mi><mn>2</mn><mn>2</mn></msubsup><mo>+</mo><mtext>Tr</mtext><mo stretchy="false">(</mo><msub><mi mathvariant="normal">Σ</mi><mi>r</mi></msub><mo>+</mo><msub><mi mathvariant="normal">Σ</mi><mi>g</mi></msub><mo>−</mo><mn>2</mn><mo stretchy="false">(</mo><msub><mi mathvariant="normal">Σ</mi><mi>r</mi></msub><msub><mi mathvariant="normal">Σ</mi><mi>g</mi></msub><msup><mo stretchy="false">)</mo><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">
\text{FID}(r, g) = ||\mu_r - \mu_g||^2_2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord text"><span class="mord">FID</span></span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.02778em;">r</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">g</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord">∣∣</span><span class="mord"><span class="mord mathnormal">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.15022em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathnormal">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mord">∣</span><span class="mord"><span class="mord">∣</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -2.453em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.247em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord text"><span class="mord">Tr</span></span><span class="mopen">(</span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.969438em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.22411em; vertical-align: -0.286108em;"></span><span class="mord">2</span><span class="mopen">(</span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mord"><span class="mord">Σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">g</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose"><span class="mclose">)</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.938em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">1/2</span></span></span></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span></span></p>
<p>A lower FID score signifies that the statistics of the generated image features are more similar to those of real images, indicating higher quality and realism. Variants such as Reconstruction FID (rFID) and Generation FID (gFID) adapt this metric for specific conditional tasks.</p>
<h2 id="content-invariant-metrics">1.2 Content-Invariant Metrics</h2>
<p>Content-invariant metrics are most suitable for tasks where there is a single, well-defined ground-truth image that the generated output should match. These are common in image-to-image translation, super-resolution, and restoration tasks.</p>
<h3 id="learned-perceptual-image-patch-similarity-lpips">1.2.1 Learned Perceptual Image Patch Similarity (LPIPS)</h3>
<p>Traditional pixel-wise metrics like L2 distance often fail to capture perceptual similarity; two images can be perceptually very different yet have a small pixel-wise error. The Learned Perceptual Image Patch Similarity (LPIPS) metric was developed to better align with human perceptual judgment (Zhang et al., 2018).</p>
<p>LPIPS computes the distance between the deep feature activations of two images. It feeds two images (a generated image and its ground-truth reference) through a pre-trained deep network (such as VGG) and extracts features from multiple layers. The distance is then calculated as a weighted sum of the L2 distances between the normalized feature activations from each layer. This “perceptual loss” has been shown to be highly correlated with how humans perceive the similarity between images. A lower LPIPS score indicates that two images are more perceptually similar.</p>
<h3 id="structural-similarity-index-metric-ssim">1.2.2 Structural Similarity Index Metric (SSIM)</h3>
<p>The Structural Similarity Index Metric (SSIM) is designed to measure image quality degradation as a change in the perception of structural information. Unlike pixel-wise error metrics, SSIM evaluates the similarity between two images, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>x</mi></mrow><annotation encoding="application/x-tex">x</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">x</span></span></span></span></span> and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>y</mi></mrow><annotation encoding="application/x-tex">y</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span></span></span></span></span>, based on three components: luminance, contrast, and structure (Wang et al., 2004).</p>
<p>The three components are defined as:</p>
<ul>
<li><strong>Luminance:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>l</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>2</mn><msub><mi>μ</mi><mi>x</mi></msub><msub><mi>μ</mi><mi>y</mi></msub><mo>+</mo><msub><mi>C</mi><mn>1</mn></msub></mrow><mrow><msubsup><mi>μ</mi><mi>x</mi><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>μ</mi><mi>y</mi><mn>2</mn></msubsup><mo>+</mo><msub><mi>C</mi><mn>1</mn></msub></mrow></mfrac></mrow><annotation encoding="application/x-tex">l(x, y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.01968em;">l</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.62807em; vertical-align: -0.64242em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.985651em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mathnormal mtight">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.746314em;"><span class="" style="top: -2.214em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight">x</span></span></span><span class="" style="top: -2.786em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.746314em;"><span class="" style="top: -2.214em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span><span class="" style="top: -2.786em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.424886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.50732em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">2</span><span class="mord mtight"><span class="mord mathnormal mtight">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight">x</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span><span class="mord mtight"><span class="mord mathnormal mtight">μ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: 0em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.64242em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></li>
<li><strong>Contrast:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>c</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>2</mn><msub><mi>σ</mi><mi>x</mi></msub><msub><mi>σ</mi><mi>y</mi></msub><mo>+</mo><msub><mi>C</mi><mn>2</mn></msub></mrow><mrow><msubsup><mi>σ</mi><mi>x</mi><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>σ</mi><mi>y</mi><mn>2</mn></msubsup><mo>+</mo><msub><mi>C</mi><mn>2</mn></msub></mrow></mfrac></mrow><annotation encoding="application/x-tex">c(x, y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">c</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.62807em; vertical-align: -0.64242em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.985651em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.746314em;"><span class="" style="top: -2.214em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight">x</span></span></span><span class="" style="top: -2.786em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.746314em;"><span class="" style="top: -2.214em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span><span class="" style="top: -2.786em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.424886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.50732em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">2</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight">x</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.64242em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></li>
<li><strong>Structure:</strong> <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>s</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msub><mi>σ</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub><mo>+</mo><msub><mi>C</mi><mn>3</mn></msub></mrow><mrow><msub><mi>σ</mi><mi>x</mi></msub><msub><mi>σ</mi><mi>y</mi></msub><mo>+</mo><msub><mi>C</mi><mn>3</mn></msub></mrow></mfrac></mrow><annotation encoding="application/x-tex">s(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal">s</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.52797em; vertical-align: -0.54232em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.985651em;"><span class="" style="top: -2.655em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight">x</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">3</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.50732em;"><span class="pstrut" style="height: 3em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.164543em;"><span class="" style="top: -2.357em; margin-left: -0.03588em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">x</span><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.281886em;"><span class=""></span></span></span></span></span></span><span class="mbin mtight">+</span><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">C</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.317314em;"><span class="" style="top: -2.357em; margin-left: -0.07153em; margin-right: 0.0714286em;"><span class="pstrut" style="height: 2.5em;"></span><span class="sizing reset-size3 size1 mtight"><span class="mord mtight">3</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.143em;"><span class=""></span></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.54232em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span></span></span></span></span></li>
</ul>
<p>where <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>μ</mi></mrow><annotation encoding="application/x-tex">\mu</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal">μ</span></span></span></span></span> is the mean, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>σ</mi></mrow><annotation encoding="application/x-tex">\sigma</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span></span></span></span></span> is the standard deviation, and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>σ</mi><mrow><mi>x</mi><mi>y</mi></mrow></msub></mrow><annotation encoding="application/x-tex">\sigma_{xy}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.716668em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.151392em;"><span class="" style="top: -2.55em; margin-left: -0.03588em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">x</span><span class="mord mathnormal mtight" style="margin-right: 0.03588em;">y</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span></span></span></span></span> is the covariance. The final SSIM score is a combination of these three, typically with exponents <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>α</mi><mo separator="true">,</mo><mi>β</mi><mo separator="true">,</mo><mi>γ</mi></mrow><annotation encoding="application/x-tex">\alpha, \beta, \gamma</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.0037em;">α</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.05278em;">β</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.05556em;">γ</span></span></span></span></span> set to 1:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>SSIM</mtext><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">[</mo><mi>l</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><msup><mo stretchy="false">]</mo><mi>α</mi></msup><mo>⋅</mo><mo stretchy="false">[</mo><mi>c</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><msup><mo stretchy="false">]</mo><mi>β</mi></msup><mo>⋅</mo><mo stretchy="false">[</mo><mi>s</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><msup><mo stretchy="false">]</mo><mi>γ</mi></msup></mrow><annotation encoding="application/x-tex">
\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord text"><span class="mord">SSIM</span></span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">[</span><span class="mord mathnormal" style="margin-right: 0.01968em;">l</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mclose"><span class="mclose">]</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.714392em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.0037em;">α</span></span></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.14911em; vertical-align: -0.25em;"></span><span class="mopen">[</span><span class="mord mathnormal">c</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mclose"><span class="mclose">]</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.899108em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.05278em;">β</span></span></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mopen">[</span><span class="mord mathnormal">s</span><span class="mopen">(</span><span class="mord mathnormal">x</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">y</span><span class="mclose">)</span><span class="mclose"><span class="mclose">]</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.714392em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.05556em;">γ</span></span></span></span></span></span></span></span></span></span></span></span></span></p>
<p>The score ranges from -1 to 1, where 1 indicates a perfect match. Mean SSIM (MSSIM) is the average SSIM value calculated over multiple local windows of an image.</p>
<h3 id="peak-signal-to-noise-ratio-psnr">1.2.3 Peak Signal-to-Noise Ratio (PSNR)</h3>
<p>Peak Signal-to-Noise Ratio (PSNR) is a classic engineering metric used to quantify the reconstruction quality of lossy compression (Gonzalez &amp; Woods, 2018). It is derived from the Mean Squared Error (MSE) between a ground-truth image <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>I</mi></mrow><annotation encoding="application/x-tex">I</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.07847em;">I</span></span></span></span></span> and a generated or compressed image <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>K</mi></mrow><annotation encoding="application/x-tex">K</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.07153em;">K</span></span></span></span></span>.</p>
<p>First, the MSE is calculated:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>MSE</mtext><mo>=</mo><mfrac><mn>1</mn><mrow><mi>m</mi><mi>n</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>m</mi><mo>−</mo><mn>1</mn></mrow></munderover><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></munderover><mo stretchy="false">[</mo><mi>I</mi><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>−</mo><mi>K</mi><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo><msup><mo stretchy="false">]</mo><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">
\text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord text"><span class="mord">MSE</span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 3.21489em; vertical-align: -1.41378em;"></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.32144em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathnormal">mn</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.686em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.80111em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mrel mtight">=</span><span class="mord mtight">0</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span><span class="" style="top: -4.3em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">m</span><span class="mbin mtight">−</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.27767em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.80111em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.05724em;">j</span><span class="mrel mtight">=</span><span class="mord mtight">0</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span><span class="" style="top: -4.30001em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">n</span><span class="mbin mtight">−</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.41378em;"><span class=""></span></span></span></span></span><span class="mopen">[</span><span class="mord mathnormal" style="margin-right: 0.07847em;">I</span><span class="mopen">(</span><span class="mord mathnormal">i</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.05724em;">j</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.11411em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.07153em;">K</span><span class="mopen">(</span><span class="mord mathnormal">i</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord mathnormal" style="margin-right: 0.05724em;">j</span><span class="mclose">)</span><span class="mclose"><span class="mclose">]</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.864108em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span></span></p>
<p>where <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow><annotation encoding="application/x-tex">m \times n</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.66666em; vertical-align: -0.08333em;"></span><span class="mord mathnormal">m</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">×</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">n</span></span></span></span></span> are the image dimensions. PSNR is then defined in decibels (dB) as:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>PSNR</mtext><mo>=</mo><mn>10</mn><mo>⋅</mo><msub><mrow><mi>log</mi><mo>⁡</mo></mrow><mn>10</mn></msub><mrow><mo fence="true">(</mo><mfrac><msubsup><mtext>MAX</mtext><mi>I</mi><mn>2</mn></msubsup><mtext>MSE</mtext></mfrac><mo fence="true">)</mo></mrow></mrow><annotation encoding="application/x-tex">
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord text"><span class="mord">PSNR</span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.64444em; vertical-align: 0em;"></span><span class="mord">10</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">⋅</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 2.51437em; vertical-align: -0.95003em;"></span><span class="mop"><span class="mop">lo<span style="margin-right: 0.01389em;">g</span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.206968em;"><span class="" style="top: -2.45586em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">10</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.24414em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner"><span class="mopen delimcenter" style="top: 0em;"><span class="delimsizing size3">(</span></span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.56434em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord text"><span class="mord">MSE</span></span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord"><span class="mord text"><span class="mord">MAX</span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.887338em;"><span class="" style="top: -2.453em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07847em;">I</span></span></span><span class="" style="top: -3.13623em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.247em;"><span class=""></span></span></span></span></span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.686em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mclose delimcenter" style="top: 0em;"><span class="delimsizing size3">)</span></span></span></span></span></span></span></span></p>
<p>Here, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mtext>MAX</mtext><mi>I</mi></msub></mrow><annotation encoding="application/x-tex">\text{MAX}_I</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.83333em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord text"><span class="mord">MAX</span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.328331em;"><span class="" style="top: -2.55em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07847em;">I</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> is the maximum possible pixel value of the image (e.g., 255 for an 8-bit grayscale image). A higher PSNR generally indicates a higher-quality reconstruction, though it does not always correlate well with human perception.</p>
<h3 id="clip-score--semantic-similarity">1.2.4 CLIP Score / Semantic Similarity</h3>
<p>When evaluating how well a generation respects a text prompt or preserves content, researchers often use CLIP-based metrics. The CLIP Score measures the semantic similarity between an image and a text description by calculating the cosine similarity of their embeddings from the pre-trained CLIP model.</p>
<ul>
<li><strong>Image-Text Similarity:</strong> In text-to-image generation, the CLIP score between the generated image and its input prompt measures alignment.</li>
<li><strong>Image-Image Similarity:</strong> For reconstruction tasks, the CLIP similarity between the reconstructed and original images can measure the preservation of semantics, acting as another form of perceptual metric. A tokenizer could have a low LPIPS score (getting textures right) but scramble the overall meaning (e.g., reconstructing an object with the wrong shape but correct local texture). Semantic metrics can catch this discrepancy.</li>
</ul>
<p>Some research evaluates the quality of token representations directly by measuring zero-shot classification accuracy using CLIP or by training linear probes on the tokens. A high accuracy indicates that the tokens retain strong semantic information.</p>
<p><strong>Limitations:</strong> CLIP-based metrics are dependent on the biases of the pre-trained model. They might not penalize subtle but important errors (e.g., a dog vs. a wolf may be semantically close in CLIP space but are distinct classes). However, they are a useful complement to pixel-level and perceptual metrics.</p>
<h3 id="robustness-and-out-of-distribution-ood-metrics">1.2.5 Robustness and Out-of-Distribution (OOD) Metrics</h3>
<p>These metrics, which are less standardized, test models on images outside their training distribution to see if performance degrades. This can involve qualitative inspection of outputs for OOD images or adding perturbations (like noise) to inputs to see if the resulting tokens or images are stable. There isn’t a common scalar metric here, but the concept is critical: a robust tokenizer will produce reasonable tokens for a medical image even if trained only on natural images. As generative models are applied to broader datasets, tokenizers trained on web-scale data are expected to be more robust than those trained on narrow datasets like ImageNet.</p>
<h3 id="summary-of-image-quality-metrics">1.2.6 Summary of Image Quality Metrics</h3>

<table>
<thead>
<tr>
<th align="left"><strong>Metric</strong></th>
<th align="left"><strong>Measures</strong></th>
<th align="left"><strong>Value Range</strong></th>
<th align="left"><strong>Direction</strong></th>
<th align="left"><strong>Interpretation</strong></th>
<th align="left"><strong>Main Use Case</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Content-Variant Metrics</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>Inception Score (IS)</strong></td>
<td align="left">Image quality and diversity</td>
<td align="left">1 to 10+ (theoretical max log(1000) ≈ 6.9)</td>
<td align="left">Higher is better</td>
<td align="left">IS &gt; 8: Excellent, 5–8: Good, 3–5: Fair, &lt;3: Poor</td>
<td align="left">Unconditional generation</td>
</tr>
<tr>
<td align="left"><strong>Fréchet Inception Distance (FID)</strong></td>
<td align="left">Distribution similarity</td>
<td align="left">0 to 300+</td>
<td align="left">Lower is better</td>
<td align="left">FID &lt; 30: Excellent, 30–50: Good, 50–100: Fair, &gt;100: Poor</td>
<td align="left">General image generation quality</td>
</tr>
<tr>
<td align="left"><strong>rFID (conditional)</strong></td>
<td align="left">Conditional FID</td>
<td align="left">0 to 300+</td>
<td align="left">Lower is better</td>
<td align="left">Similar to FID, for conditional tasks</td>
<td align="left">Super-resolution, inpainting</td>
</tr>
<tr>
<td align="left"><strong>gFID (geometric)</strong></td>
<td align="left">Geometric variant of FID</td>
<td align="left">0 to 300+</td>
<td align="left">Lower is better</td>
<td align="left">Alternative FID calculation</td>
<td align="left">Research contexts</td>
</tr>
<tr>
<td align="left"><strong>Content-Invariant Metrics</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>LPIPS</strong></td>
<td align="left">Learned perceptual similarity</td>
<td align="left">0 to 1+</td>
<td align="left">Lower is better</td>
<td align="left">LPIPS &lt; 0.1: Very similar, 0.1–0.3: Similar, &gt;0.3: Different</td>
<td align="left">Perceptual quality assessment</td>
</tr>
<tr>
<td align="left"><strong>SSIM</strong></td>
<td align="left">Structural similarity</td>
<td align="left">-1 to 1</td>
<td align="left">Higher is better</td>
<td align="left">&gt; 0.9: Excellent, 0.8–0.9: Good, 0.6–0.8: Fair, &lt;0.6: Poor</td>
<td align="left">Pixel-level structural comparison</td>
</tr>
<tr>
<td align="left"><strong>MSSIM</strong></td>
<td align="left">Multi-scale structural similarity</td>
<td align="left">0 to 1</td>
<td align="left">Higher is better</td>
<td align="left">Similar to SSIM, more robust to scale</td>
<td align="left">Different viewing distances/scales</td>
</tr>
<tr>
<td align="left"><strong>PSNR</strong></td>
<td align="left">Peak signal-to-noise ratio</td>
<td align="left">10 to 50+ dB</td>
<td align="left">Higher is better</td>
<td align="left">&gt; 40dB: Excellent, 30–40dB: Good, 20–30dB: Fair, &lt;20dB: Poor</td>
<td align="left">Traditional pixel-level quality</td>
</tr>
<tr>
<td align="left"><strong>CLIP Score</strong></td>
<td align="left">Semantic similarity / alignment</td>
<td align="left">~0.1 to 0.4+</td>
<td align="left">Higher is better</td>
<td align="left">Higher score means better semantic alignment with text or reference image</td>
<td align="left">Text-to-image, semantic fidelity</td>
</tr>
</tbody>
</table><h2 id="metrics-for-vector-quantization-and-autoregressive-models">1.3 Metrics for Vector Quantization and Autoregressive Models</h2>
<p>For generative models that rely on a discrete codebook (vector quantization) followed by an autoregressive model to generate token sequences, a specialized set of metrics is required to evaluate the intermediate representation and the sequence modeling performance. This two-stage approach was prominently established by models like VQ-VAE (van den Oord &amp; Vinyals, 2017).</p>
<h3 id="codebook-and-token-utilization-metrics">1.3.1 Codebook and Token Utilization Metrics</h3>
<ul>
<li><strong>Codebook Utilization Rate:</strong> This metric measures the percentage of available vectors (codes) in the codebook that are actually used during the encoding of a representative dataset. Low utilization, or “codebook collapse,” suggests that the model is relying on only a small subset of its learned codes, indicating representational inefficiency.</li>
<li><strong>Token Usage Entropy:</strong> This quantifies the uniformity of the token distribution. For a codebook of size <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>K</mi></mrow><annotation encoding="application/x-tex">K</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.07153em;">K</span></span></span></span></span>, the entropy is calculated as <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>H</mi><mo stretchy="false">(</mo><mi>p</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><msubsup><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></msubsup><msub><mi>p</mi><mi>k</mi></msub><msub><mrow><mi>log</mi><mo>⁡</mo></mrow><mn>2</mn></msub><mo stretchy="false">(</mo><msub><mi>p</mi><mi>k</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">H(p) = -\sum_{k=1}^{K} p_k \log_2(p_k)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.08125em;">H</span><span class="mopen">(</span><span class="mord mathnormal">p</span><span class="mclose">)</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.28094em; vertical-align: -0.29971em;"></span><span class="mord">−</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop"><span class="mop op-symbol small-op" style="position: relative; top: -5e-06em;">∑</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.981231em;"><span class="" style="top: -2.40029em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03148em;">k</span><span class="mrel mtight">=</span><span class="mord mtight">1</span></span></span></span><span class="" style="top: -3.2029em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.07153em;">K</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.29971em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03148em;">k</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop"><span class="mop">lo<span style="margin-right: 0.01389em;">g</span></span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.206968em;"><span class="" style="top: -2.45586em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.24414em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03148em;">k</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>, where <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>p</mi><mi>k</mi></msub></mrow><annotation encoding="application/x-tex">p_k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord"><span class="mord mathnormal">p</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.03148em;">k</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span> is the frequency of the <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>k</mi></mrow><annotation encoding="application/x-tex">k</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03148em;">k</span></span></span></span></span>-th token. Higher entropy indicates a more balanced and efficient use of the entire codebook.</li>
<li><strong>Token Frequency Distribution Skew:</strong> This measures the concentration of usage among a few popular tokens. High skew indicates an imbalanced distribution where a few codes dominate, which can be detrimental to learning a rich representation.</li>
</ul>
<h3 id="autoregressive-generation-performance">1.3.2 Autoregressive Generation Performance</h3>
<p>The metrics used to evaluate the second stage of these models are adapted from natural language processing, where autoregressive models are standard. The application of these models to pixel-level generation established their use in the image domain (van den Oord et al., 2016).</p>
<ul>
<li><strong>Bits Per Token (BPT):</strong> This metric measures the compression efficiency of the autoregressive model, calculated as the average negative log-likelihood of the token sequences. A lower BPT indicates that the model is better at predicting the next token, implying a superior understanding of the token distribution.</li>
<li><strong>Negative Log-Likelihood (NLL):</strong> Directly measures how well the autoregressive model predicts the sequence of tokens produced by the encoder. A lower NLL signifies better modeling of the tokenized data’s probability distribution.</li>
<li><strong>Perplexity:</strong> Calculated as <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>exp</mi><mo>⁡</mo><mo stretchy="false">(</mo><mtext>NLL</mtext><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\exp(\text{NLL})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mop">exp</span><span class="mopen">(</span><span class="mord text"><span class="mord">NLL</span></span><span class="mclose">)</span></span></span></span></span>, perplexity is an intuitive measure of how “surprised” the model is by a sequence of tokens. Lower perplexity indicates better sequence modeling and prediction capabilities.</li>
<li><strong>Convergence Curves:</strong> These are plots of the training loss or perplexity over training epochs. While qualitative, they are important for diagnosis. A faster drop in the loss curve suggests that the learned tokens are “more learnable” for the autoregressive model. Conversely, slow or unstable convergence (e.g., an oscillating loss) can be a sign that a particular model design choice (like an unregularized codebook) is making the prediction task too difficult.</li>
</ul>
<h3 id="summary-of-vq-and-ar-metrics">1.3.3 Summary of VQ and AR Metrics</h3>

<table>
<thead>
<tr>
<th align="left"><strong>Metric</strong></th>
<th align="left"><strong>Measures</strong></th>
<th align="left"><strong>Direction</strong></th>
<th align="left"><strong>Interpretation</strong></th>
<th align="left"><strong>Main Use Case</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Codebook &amp; Token Utilization</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>Codebook Utilization</strong></td>
<td align="left">Percentage of codebook used</td>
<td align="left">Higher is better</td>
<td align="left">High rate indicates efficient codebook; low rate suggests collapse</td>
<td align="left">Diagnosing VQ model efficiency</td>
</tr>
<tr>
<td align="left"><strong>Token Usage Entropy</strong></td>
<td align="left">Uniformity of token usage</td>
<td align="left">Higher is better</td>
<td align="left">High entropy means balanced usage; low means unbalanced</td>
<td align="left">Assessing codebook balance</td>
</tr>
<tr>
<td align="left"><strong>Token Frequency Skew</strong></td>
<td align="left">Concentration of token usage</td>
<td align="left">Lower is better</td>
<td align="left">Low skew means balanced usage; high skew means few tokens dominate</td>
<td align="left">Identifying token over-specialization</td>
</tr>
<tr>
<td align="left"><strong>Autoregressive Performance</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>Bits Per Token (BPT)</strong></td>
<td align="left">Compression efficiency</td>
<td align="left">Lower is better</td>
<td align="left">Lower BPT means better prediction and modeling of token distribution</td>
<td align="left">Evaluating AR model compression</td>
</tr>
<tr>
<td align="left"><strong>NLL</strong></td>
<td align="left">Predictive accuracy of model</td>
<td align="left">Lower is better</td>
<td align="left">Lower NLL means the model is less surprised by the data sequence</td>
<td align="left">Fundamental AR model evaluation</td>
</tr>
<tr>
<td align="left"><strong>Perplexity</strong></td>
<td align="left">Model’s surprise at data</td>
<td align="left">Lower is better</td>
<td align="left">Lower perplexity indicates better sequence modeling and prediction</td>
<td align="left">Intuitive measure of AR performance</td>
</tr>
<tr>
<td align="left"><strong>Convergence Speed</strong></td>
<td align="left">Rate of loss reduction</td>
<td align="left">Faster is better</td>
<td align="left">Faster convergence suggests tokens are more “learnable”</td>
<td align="left">Diagnosing token representation</td>
</tr>
</tbody>
</table><hr>
<h3 id="references-1">References</h3>
<p>Gonzalez, R. C., &amp; Woods, R. E. (2018). <em>Digital image processing</em> (4th ed.). Pearson.</p>
<p>Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., &amp; Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In <em>Advances in Neural Information Processing Systems 30</em> (NIPS 2017).</p>
<p>Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., &amp; Chen, X. (2016). Improved Techniques for Training GANs. In <em>Advances in Neural Information Processing Systems 29</em> (NIPS 2016).</p>
<p>van den Oord, A., Kalchbrenner, N., &amp; Kavukcuoglu, K. (2016). Pixel recurrent neural networks. In <em>Proceedings of the 33rd International Conference on Machine Learning</em> (ICML).</p>
<p>van den Oord, A., &amp; Vinyals, O. (2017). Neural discrete representation learning. In <em>Advances in Neural Information Processing Systems 30</em> (NIPS 2017).</p>
<p>Wang, Z., Bovik, A. C., Sheikh, H. R., &amp; Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. <em>IEEE Transactions on Image Processing</em>, 13(4), 600-612.</p>
<p>Zhang, R., Isola, P., Efros, A. A., Shechtman, E., &amp; Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In <em>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</em> (CVPR).</p>
<h1 id="literature-review-the-evolution-of-visual-tokenization">2. Literature Review: The Evolution of Visual Tokenization</h1>
<p>The application of autoregressive (AR) models to image generation is a story of representation. Early attempts to model pixels directly were computationally intractable and failed to capture global structure. The pivotal innovation that unlocked the potential of AR models for vision was the development of vector-quantized (VQ) tokenizers, which transform continuous images into discrete sequences amenable to the powerful “next-token prediction” paradigm that revolutionized natural language processing. This section traces the historical evolution of this “visual language,” from its foundational concepts to the sophisticated, semantically-aware tokenizers of the modern era.</p>
<h2 id="foundational-era-from-discrete-codes-to-high-fidelity-tokens-2017-2021">2.1 Foundational Era: From Discrete Codes to High-Fidelity Tokens (2017-2021)</h2>
<p>The journey began with the <strong>Vector Quantised-Variational AutoEncoder (VQ-VAE)</strong> in 2017 (van den Oord &amp; Vinyals, 2017). This work introduced a discrete latent bottleneck into a VAE, elegantly solving the “posterior collapse” problem where a powerful decoder might otherwise ignore the latent codes. By forcing the representation into a discrete form, VQ-VAE enabled the training of an AR prior (like PixelCNN) on these codes to generate coherent, albeit blurry, images. It was the crucial proof-of-concept that discrete image representations could work.</p>
<p>While groundbreaking, VQ-VAE’s reconstructions suffered from low fidelity due to its reliance on pixel-wise losses. The breakthrough in quality came with <strong>VQGAN</strong> in 2021 (Esser et al., 2021). By incorporating a patch-based adversarial loss and a perceptual loss (LPIPS), VQGAN forced its decoder to generate realistic details and textures. This leap in reconstruction fidelity produced much sharper tokens and made high-resolution AR generation a practical reality, establishing VQGAN as the de facto standard tokenizer for the next several years.</p>
<p>This two-stage VQ-AR approach culminated in OpenAI’s <strong>DALL·E</strong> (Ramesh et al., 2021). A landmark model, DALL·E scaled the paradigm to text-to-image synthesis by training a massive 12-billion parameter transformer to model a unified stream of text tokens followed by image tokens from a discrete VAE. It demonstrated that complex, novel visual concepts could be generated and composed from natural language, proving the power of treating image tokens like words in a foreign language.</p>
<h2 id="the-modern-era-scaling-efficiency-and-semantics-2022-present">2.2 The Modern Era: Scaling, Efficiency, and Semantics (2022-Present)</h2>
<p>Following the success of DALL·E, the field was briefly dominated by diffusion models, which offered superior photorealism. However, a new wave of research sought to re-establish the viability of AR models by addressing their limitations in speed, coherence, and semantic understanding.</p>
<p><strong>Scaling and Unification:</strong> The work on <strong>LlamaGen</strong> (Sun et al., 2024) demonstrated that with sufficient scale and an improved VQGAN-style tokenizer, a standard Llama architecture could achieve state-of-the-art FID scores on ImageNet, surpassing prominent diffusion models and re-igniting interest in AR. Concurrently, Meta’s <strong>Chameleon (CM3leon)</strong> (Li et al., 2023) pushed towards a unified multimodal architecture. It employed an “early-fusion” approach where image and text tokens are processed by a single transformer, enabling the model to natively understand and generate arbitrarily interleaved sequences of images and text.</p>
<p><strong>The Shift to 1D and Flexibility:</strong> Recognizing the inefficiency of rigid 2D token grids, <strong>TiTok</strong> (Yu et al., 2024) introduced a Transformer-based 1D tokenizer. It showed that an image could be represented by a highly compact sequence of just 32 tokens, leading to a dramatic acceleration in generation speed. Building on this, <strong>FlexTok</strong> (Bachmann et al., 2025) introduced variable-length token sequences. Using nested dropout, it learns an ordered representation where initial tokens capture high-level semantics and subsequent tokens add finer details, enabling a coarse-to-fine generation process.</p>
<p><strong>The Rise of Semantics:</strong> A key frontier became bridging the gap between tokens for reconstruction (pixel detail) and for understanding (abstract concepts). <strong>UniTok</strong> (Ma et al., 2025) addressed this by using multi-codebook quantization to expand the representational capacity of tokens, allowing them to capture both semantic and perceptual information. Taking this further, <strong>Semanticist</strong> (Wang et al., 2025) introduced a highly structured latent space with a provable PCA-like hierarchy, explicitly decoupling high-level semantic content from low-level spectral details to create a highly efficient and interpretable visual language.</p>
<p>This progression reveals a clear trajectory. The field moved from making images discrete (VQ-VAE), to making them perceptually accurate (VQGAN), to making them controllable (DALL·E). The latest research is focused on making the token sequences themselves more intelligent—more efficient (TiTok), more adaptive (FlexTok), and more semantically structured (Semanticist). In this new paradigm, the tokenizer is no longer a simple compression utility; it is a powerful inductive bias that pre-organizes visual information, transforming the AR model’s task from raw creation to the logical synthesis of meaningful components.</p>
<h2 id="timeline-of-key-developments-in-visual-tokenization">2.3 Timeline of Key Developments in Visual Tokenization</h2>

<table>
<thead>
<tr>
<th align="left"><strong>Year</strong></th>
<th align="left"><strong>Model / Paper</strong></th>
<th align="left"><strong>Key Contribution</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>2017</strong></td>
<td align="left"><strong>VQ-VAE</strong></td>
<td align="left">Introduced discrete latent bottlenecks for images, enabling AR priors.</td>
</tr>
<tr>
<td align="left"><strong>2021</strong></td>
<td align="left"><strong>VQGAN</strong></td>
<td align="left">Added adversarial and perceptual losses for high-fidelity token reconstruction.</td>
</tr>
<tr>
<td align="left"><strong>2021</strong></td>
<td align="left"><strong>DALL·E</strong></td>
<td align="left">Scaled the VQ-AR approach for large-scale text-to-image generation.</td>
</tr>
<tr>
<td align="left"><strong>2023</strong></td>
<td align="left"><strong>Chameleon (CM3leon)</strong></td>
<td align="left">Unified image and text generation with a single early-fusion transformer.</td>
</tr>
<tr>
<td align="left"><strong>2024</strong></td>
<td align="left"><strong>LlamaGen</strong></td>
<td align="left">Proved that scaled AR models can outperform diffusion models in fidelity.</td>
</tr>
<tr>
<td align="left"><strong>2024</strong></td>
<td align="left"><strong>TiTok</strong></td>
<td align="left">Introduced an extremely compressed 1D token representation (e.g., 32 tokens).</td>
</tr>
<tr>
<td align="left"><strong>2024</strong></td>
<td align="left"><strong>IBQ</strong></td>
<td align="left">Developed a method to train massive codebooks (262k+) without code collapse.</td>
</tr>
<tr>
<td align="left"><strong>2024</strong></td>
<td align="left"><strong>XQ-GAN / SeQ-GAN</strong></td>
<td align="left">Advanced GAN-based tokenizers with flexible frameworks and training strategies.</td>
</tr>
<tr>
<td align="left"><strong>2025</strong></td>
<td align="left"><strong>UniTok</strong></td>
<td align="left">Used multi-codebook quantization to unify tokens for generation and understanding.</td>
</tr>
<tr>
<td align="left"><strong>2025</strong></td>
<td align="left"><strong>FlexTok</strong></td>
<td align="left">Enabled variable-length, coarse-to-fine generation with ordered 1D tokens.</td>
</tr>
<tr>
<td align="left"><strong>2025</strong></td>
<td align="left"><strong>Semanticist</strong></td>
<td align="left">Created a hierarchical, PCA-like token structure for semantic decoupling.</td>
</tr>
<tr>
<td align="left"><strong>2025</strong></td>
<td align="left"><strong>HITA / TokenFlow</strong></td>
<td align="left">Introduced specialized architectures (holistic tokens, dual codebooks) for improved coherence and semantic alignment.</td>
</tr>
</tbody>
</table><hr>
<h3 id="references-2">References</h3>
<p>Esser, P., Rombach, R., &amp; Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In <em>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</em> (CVPR).</p>
<p>Li, Y., Geng, S., Li, B., Liu, F., Yang, F., et al. (2023). CM3leon: A Multi-modal, Causal-masked, Multi-task Model-in-the-Loop. <em>arXiv preprint arXiv:2305.07295</em>.</p>
<p>Ma, Y., Zhang, X., Li, H., &amp; Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. <em>Project Page</em>.</p>
<p>Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., et al. (2021). Zero-Shot Text-to-Image Generation. In <em>Proceedings of the 38th International Conference on Machine Learning</em> (ICML).</p>
<p>Sun, J., Wang, R., Li, Y., Wang, W., &amp; Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. <em>arXiv preprint arXiv:2404.09344</em>.</p>
<p>van den Oord, A., &amp; Vinyals, O. (2017). Neural discrete representation learning. In <em>Advances in Neural Information Processing Systems 30</em> (NIPS 2017).</p>
<p>Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. <em>Project Page</em>.</p>
<p>Yu, J., Wang, C., Wang, X., &amp; Wang, J. (2024). An Image is Worth 32 Tokens: A Transformer-based 1D Tokenizer. <em>arXiv preprint arXiv:2404.05923</em>.</p>
<p>Bachmann, R., et al. (2025). FlexTok: A Trainable Variable-Length Tokenizer for Language-Interfaced Vision. <em>Project Page</em>.</p>
<hr>
<h1 id="a-detailed-analysis-of-key-visual-tokenizers">3. A Detailed Analysis of Key Visual Tokenizers</h1>
<p>This section provides a detailed summary of the key tokenizers and associated models that have defined the field of autoregressive image generation. Each entry outlines the problem it addresses, its core design innovations, key results, and notable limitations.</p>
<h2 id="core-tokenizers-and-models">3.1 Core Tokenizers and Models</h2>
<h3 id="vq-vae-2017-–-learning-discrete-image-codes">3.1.1 VQ-VAE (2017) – Learning Discrete Image Codes</h3>
<ul>
<li><strong>Problem:</strong> Standard VAEs, when paired with powerful autoregressive decoders, often suffer from “posterior collapse,” where the decoder ignores the latent code. The goal was to enable the use of powerful discrete generative models on images.</li>
<li><strong>Design:</strong> A CNN encoder maps an image to a feature map, where each vector is quantized to the nearest entry in a learned codebook. A straight-through estimator and a commitment loss are used for training. This forces the model to learn a discrete representation.</li>
<li><strong>Results:</strong> VQ-VAE proved that discrete bottlenecks could be trained end-to-end, enabling AR models like PixelCNN to generate new images from the learned codes.</li>
<li><strong>Limitations:</strong> Reconstructions were often blurry and lacked fine texture detail due to reliance on MSE loss. The codebook often had underutilized entries (“codebook collapse”).</li>
</ul>
<h3 id="vqgan-2021-–-high-fidelity-image-tokens">3.1.2 VQGAN (2021) – High-Fidelity Image Tokens</h3>
<ul>
<li><strong>Problem:</strong> VQ-VAE reconstructions were overly smooth and missed the high-frequency details necessary for photorealistic generation.</li>
<li><strong>Design:</strong> VQGAN augmented the VQ-VAE framework with a patch-based GAN discriminator and a perceptual loss (LPIPS). These additions encourage the tokenizer to preserve realistic detail and texture.</li>
<li><strong>Results:</strong> Dramatically improved the visual fidelity of reconstructions, producing sharp, photo-realistic images. VQGAN’s high-quality tokens became the basis for the first compelling transformer-based image generators.</li>
<li><strong>Limitations:</strong> Adversarial training could introduce instability. The model still primarily optimized for reconstruction over semantics and could encode visual quirks to fool the discriminator.</li>
</ul>
<h3 id="dall·e-2021-–-large-scale-ar-image-generation">3.1.3 DALL·E (2021) – Large-Scale AR Image Generation</h3>
<ul>
<li><strong>Problem:</strong> To demonstrate that AR models can generate complex, novel images from text descriptions at a very large scale.</li>
<li><strong>Design:</strong> Used a proprietary discrete VAE (dVAE) to encode 256x256 images into a 32×32 grid of 1024 tokens from a large codebook of 8192 entries. A 12-billion parameter Transformer was then trained to model sequences of text and image tokens.</li>
<li><strong>Results:</strong> DALL·E produced an unprecedented diversity of creative and coherent images from text prompts, proving that visual tokens combined with transformers could power a highly capable generative model.</li>
<li><strong>Limitations:</strong> Images were not fully photo-realistic and often contained artifacts. The tokenizer and model struggled to render text coherently within images.</li>
</ul>
<h3 id="llamagen-2024-–-scaling-autoregression-to-beat-diffusion">3.1.4 LlamaGen (2024) – Scaling Autoregression to Beat Diffusion</h3>
<ul>
<li><strong>Problem:</strong> To prove that a “GPT-for-images,” when properly scaled, could outperform leading diffusion models in image generation fidelity and speed.</li>
<li><strong>Design:</strong> A family of AR models (up to 3.1B parameters) based on the LLaMA architecture. It uses an improved VQGAN-style tokenizer with a large codebook (8192) and training optimizations to improve reconstruction and codebook utilization.</li>
<li><strong>Results:</strong> Achieved a state-of-the-art FID of ~2.18 on ImageNet 256x256, outperforming comparable diffusion models. It also demonstrated significantly faster inference speeds.</li>
<li><strong>Limitations:</strong> While strong on benchmarks like ImageNet, external evaluations noted it struggled with some complex compositional prompts (e.g., precise spatial relations) compared to guided diffusion models.</li>
</ul>
<h3 id="chameleon--cm3leon-2023-–-mixed-modality-tokens-for-images--text">3.1.5 Chameleon / CM3leon (2023) – Mixed-Modality Tokens for Images &amp; Text</h3>
<ul>
<li><strong>Problem:</strong> To unify image understanding and generation with text in a single “early-fusion” model, requiring a tokenizer that produces image tokens compatible with text tokens.</li>
<li><strong>Design:</strong> A large transformer that ingests a single, interleaved sequence of text and image tokens. The image tokenizer converts 512x512 images into a 32x32 grid of 1024 tokens from an 8192-entry codebook, with special oversampling on faces during training to improve fidelity.</li>
<li><strong>Results:</strong> Achieved state-of-the-art or competitive results across a breadth of tasks, including image captioning, VQA, and image generation, demonstrating that discrete image tokens can be treated just like word tokens in a unified model.</li>
<li><strong>Limitations:</strong> The tokenizer struggles to reconstruct fine-grained text within images, limiting its OCR capabilities. Image generation quality was competitive but not state-of-the-art compared to specialized models.</li>
</ul>
<h3 id="titok-2024-–-transformer-tokenizer-with-32-tokens-per-image">3.1.6 TiTok (2024) – Transformer Tokenizer with 32 Tokens per Image</h3>
<ul>
<li><strong>Problem:</strong> To drastically reduce the AR sequence length to accelerate generation by removing the redundancy of fixed 2D grids.</li>
<li><strong>Design:</strong> A Transformer-based encoder that uses a small set of learnable query vectors (e.g., 32) to attend over the entire image and produce a highly compressed 1D sequence of discrete tokens.</li>
<li><strong>Results:</strong> Represented a 256x256 image with just 32 tokens while achieving an excellent gFID of 1.97 on ImageNet. This enabled generation speeds hundreds of times faster than diffusion.</li>
<li><strong>Limitations:</strong> The extreme compression can lead to the loss of fine details. Evaluations showed weaker reconstruction of small text and faces compared to less compressed tokenizers.</li>
</ul>
<h3 id="unitok-2025-–-unified-tokenizer-for-generation-and-understanding">3.1.7 UniTok (2025) – Unified Tokenizer for Generation and Understanding</h3>
<ul>
<li><strong>Problem:</strong> To bridge the gap between high-fidelity generative tokens and semantically meaningful tokens for understanding tasks like classification.</li>
<li><strong>Design:</strong> A multi-codebook quantization scheme. A latent feature vector is split into several chunks, and each chunk is quantized with its own independent sub-codebook. This exponentially increases the effective vocabulary size and is trained with a mix of reconstruction and semantic losses.</li>
<li><strong>Results:</strong> Achieved state-of-the-art performance on both tasks simultaneously: a near-perfect reconstruction FID of 0.38 and 78.6% zero-shot classification accuracy on ImageNet, outperforming CLIP.</li>
<li><strong>Limitations:</strong> The design is more complex, requiring the training and management of multiple codebooks.</li>
</ul>
<h3 id="flextok-2025-–-flexible-length-1d-tokens-coarse-to-fine">3.1.8 FlexTok (2025) – Flexible-Length 1D Tokens (Coarse-to-Fine)</h3>
<ul>
<li><strong>Problem:</strong> Fixed-length token sequences are inefficient, forcing a one-size-fits-all generation cost regardless of image complexity.</li>
<li><strong>Design:</strong> A multi-stage pipeline that produces an ordered 1D token sequence. Nested dropout during training forces the earliest tokens to encode the most salient, high-level information. A decoder can then reconstruct a plausible image from any prefix of the sequence.</li>
<li><strong>Results:</strong> Enables an adaptive, coarse-to-fine generation process. A recognizable image can be generated from a few tokens and progressively refined by generating more, providing a “budget knob” for AR generation.</li>
<li><strong>Limitations:</strong> The overall system is complex, involving a VAE, a transformer, and a rectified flow decoder. The notion of “image complexity” that determines token length is implicit.</li>
</ul>
<h3 id="semanticist-2025-–-semantic-first-tokenization-with-diffusion-decoding">3.1.9 Semanticist (2025) – Semantic-First Tokenization with Diffusion Decoding</h3>
<ul>
<li><strong>Problem:</strong> Standard tokenizers entangle high-level semantics with low-level spectral details, making their representations inefficient and hard to interpret.</li>
<li><strong>Design:</strong> Imposes a PCA-like structure on a 1D token sequence using a nested CFG training strategy, ensuring tokens are ordered by importance. It uses a diffusion-based decoder to explicitly decouple semantic content (low-frequency) from spectral detail (high-frequency).</li>
<li><strong>Results:</strong> Achieved state-of-the-art reconstruction FID while creating a highly interpretable latent space. AR models required significantly fewer tokens for high-quality generation, and the tokens showed strong semantic properties for downstream tasks.</li>
<li><strong>Limitations:</strong> The diffusion decoder is slower than a standard VQ decoder, increasing computational load during inference.</li>
</ul>
<h2 id="other-notable-advances-in-tokenization">3.2 Other Notable Advances in Tokenization</h2>
<ul>
<li><strong>HITA (2025):</strong> Introduced a holistic-to-local design where a few “holistic” tokens capture global context before patch tokens are generated. This improved image coherence and accelerated AR model training.</li>
<li><strong>XQ-GAN (2024):</strong> An open-source, extensible tokenizer framework that combines multiple quantization methods (residual, product, multi-scale) and robust training techniques like codeword dropout.</li>
<li><strong>IBQ (2024):</strong> A technical breakthrough enabling the training of massive codebooks (e.g., 262k entries) by backpropagating gradients through the discrete code selection, overcoming “codebook collapse.”</li>
<li><strong>SeQ-GAN (2024):</strong> Focused on a two-phase training objective: first, a semantic-aware loss to ensure tokens capture global content, followed by GAN finetuning to restore detail.</li>
<li><strong>VGQ (2025):</strong> An experimental tokenizer that represents image patches with parametric 2D Gaussians instead of pixel grids, aiming to make tokens more geometrically aware of shape and position.</li>
<li><strong>TokenFlow (2025):</strong> A dual-codebook architecture where one codebook captures semantics (aligned with CLIP) and the other captures pixel details, linked by a shared index for unified understanding and generation.</li>
</ul>
<hr>
<h3 id="references-3">References</h3>
<p>(Note: This list is comprehensive, including all sources mentioned in the detailed analysis.)</p>
<p>Bachmann, R., et al. (2025). FlexTok: A Trainable Variable-Length Tokenizer for Language-Interfaced Vision. <em>Project Page available at <a href="http://flextok.epfl.ch">flextok.epfl.ch</a></em>.</p>
<p>Esser, P., Rombach, R., &amp; Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In <em>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</em> (CVPR).</p>
<p>Gu, J., et al. (2024). SeQ-GAN: Semantic-Quantized GAN for High-Fidelity Image Generation. <em>arXiv preprint arXiv:2404.03043</em>.</p>
<p>Li, Y., Geng, S., Li, B., Liu, F., Yang, F., et al. (2023). CM3leon: A Multi-modal, Causal-masked, Multi-task Model-in-the-Loop. <em>arXiv preprint arXiv:2305.07295</em>.</p>
<p>Ma, Y., Zhang, X., Li, H., &amp; Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. <em>Project Page available at <a href="http://foundationvision.github.io">foundationvision.github.io</a></em>.</p>
<p>Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., et al. (2021). Zero-Shot Text-to-Image Generation. In <em>Proceedings of the 38th International Conference on Machine Learning</em> (ICML).</p>
<p>Shi, T., et al. (2024). Index Backpropagation for High-Performance VQ-VAE. <em>arXiv preprint arXiv:2404.05877</em>.</p>
<p>Sun, J., Wang, R., Li, Y., Wang, W., &amp; Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. <em>arXiv preprint arXiv:2404.09344</em>.</p>
<p>van den Oord, A., &amp; Vinyals, O. (2017). Neural discrete representation learning. In <em>Advances in Neural Information Processing Systems 30</em> (NIPS 2017).</p>
<p>Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. <em>Project Page available at <a href="http://visual-gen.github.io">visual-gen.github.io</a></em>.</p>
<p>Yu, J., Wang, C., Wang, X., &amp; Wang, J. (2024). An Image is Worth 32 Tokens: A Transformer-based 1D Tokenizer. <em>arXiv preprint arXiv:2404.05923</em>.</p>
<p>Zheng, Y., et al. (2025). HITA: A Holistic Image Tokenizer for Autoregressive Generation. <em>arXiv preprint arXiv:2501.07689</em>.</p>
<h1 id="a-taxonomy-of-design-choices-in-visual-tokenization">4. A Taxonomy of Design Choices in Visual Tokenization</h1>
<p>To build a causal understanding of how VQ tokenizers function, we can analyze them through a structured framework: how specific <strong>Design Choices</strong> lead to certain <strong>Observed Properties</strong> in the trained tokenizer, which in turn have downstream <strong>Effects</strong> on the autoregressive (AR) model’s performance and capabilities. This section systematically deconstructs the design space of visual tokenizers, providing a framework to understand why each decision matters.</p>
<h2 id="a-framework-for-tokenizer-design">4.1 A Framework for Tokenizer Design</h2>
<p>We organize this taxonomy by five major design aspects: training data, sequence structure, encoder causality, token flexibility, and semantic guidance.</p>
<h3 id="training-data-mixture--augmentations">4.1.1 Training Data Mixture &amp; Augmentations</h3>
<ul>
<li><strong>Design Choice:</strong> The composition of the dataset used to train the tokenizer, including its domain (e.g., ImageNet vs. broad web data), class distribution, and any oversampling or data augmentation strategies applied.</li>
<li><strong>Why it matters:</strong> The tokenizer will be biased to best represent the kinds of images it sees most frequently. A tokenizer trained only on object-centric datasets like ImageNet may lack the codebook capacity for concepts like legible text or nuanced facial expressions. Conversely, a tokenizer trained on a diverse web-scale dataset will learn a more universal codebook.</li>
<li><strong>Observed Properties:</strong>
<ul>
<li><strong>Codebook Usage:</strong> A tokenizer trained on narrow data may exhibit lower codebook utilization, as the variety of visual patterns is limited.</li>
<li><strong>Subset-Specific Reconstruction Quality:</strong> The tokenizer’s performance will vary across different image categories. For instance, if faces are underrepresented in training, the tokenizer will likely have a higher reconstruction error on faces. Meta’s Chameleon team actively countered this by <strong>doubling the proportion of face images in training</strong>, which resulted in the tokenizer allocating more capacity to fine facial features.</li>
<li><strong>Out-of-Distribution (OOD) Reconstruction:</strong> A tokenizer trained on a specific domain (e.g., natural images) will struggle to reconstruct OOD inputs like medical X-rays, approximating them with its vocabulary of “natural” codes. A broader training set yields a more robust, universal codebook.</li>
</ul>
</li>
<li><strong>Effects on AR and Other Tasks:</strong>
<ul>
<li><strong>AR Generation Quality:</strong> The AR model’s generative capabilities are fundamentally limited by the tokenizer’s vocabulary. If the tokenizer has not learned to encode certain structures (e.g., readable text), the AR model cannot generate them. This is reflected in metrics like low Text Accuracy (T-ACC) for generations.</li>
<li><strong>OOD Generation Robustness:</strong> A tokenizer trained on curated data can cause the AR model to fail on unusual prompts. TokenFlow, aiming for a “unified” tokenizer, was trained on multi-domain data, which had the effect of allowing a discrete-token multimodal model to <strong>surpass a strong vision-language baseline (LLaVA)</strong> on understanding tasks.</li>
<li><strong>Forgetting during Fine-tuning:</strong> Fine-tuning a tokenizer on a new domain can cause it to overwrite codes that were essential for the original domain, a phenomenon known as catastrophic forgetting. Using a larger or multi-codebook architecture (like UniTok) can help mitigate this.</li>
</ul>
</li>
<li><strong>Example &amp; Takeaway:</strong><br>
Chameleon’s tokenizer was trained on a curated mix of 1.4B image-text pairs. The <strong>observed property</strong> was strong performance on faces (due to oversampling) but poor reconstruction of small text. The <strong>effect</strong> was that the multimodal model struggled with OCR-based tasks. In contrast, UniTok’s training on diverse data with semantic supervision led to an <strong>observed property</strong> of surpassing continuous VAEs and CLIP on their respective metrics.<br>
<strong>Takeaway:</strong> <em>The scope and distribution of training data directly shape a tokenizer’s vocabulary and its ability to capture diverse content. Achieving high fidelity on specific categories like faces or text often requires explicit oversampling or data augmentation.</em></li>
</ul>
<h3 id="token-sequence-type-1d-vs.-2d-grid-and-generation-order">4.1.2 Token Sequence Type: 1D vs. 2D (Grid) and Generation Order</h3>
<ul>
<li><strong>Design Choice:</strong> Whether the tokenizer outputs a <strong>2D grid</strong> of tokens corresponding to image patches or a <strong>1D sequence</strong> with no explicit spatial structure. For 2D grids, the serialization order (e.g., raster-scan) is also a design choice.</li>
<li><strong>Why it matters:</strong> This choice critically impacts sequence length and how the AR model learns dependencies. A 2D grid flattened in raster order creates an arbitrary sequence where spatially distant patches can become temporally distant, making it difficult to model global relationships. A 1D approach can encode global information or order tokens by importance, breaking this rigid structure.</li>
<li><strong>Observed Properties:</strong>
<ul>
<li><strong>Token Frequency Distribution:</strong> 2D grid tokenizers are more prone to a <strong>skewed distribution</strong>, where tokens for common backgrounds (e.g., “sky blue”) appear very frequently due to patch-level redundancy. 1D holistic tokenizers may have a more balanced distribution.</li>
<li><strong>Reconstruction Quality:</strong> 2D grids excel at local reconstruction. 1D tokenizers that perform high compression (like TiTok) may achieve good overall FID but exhibit higher pixel-level error on fine details, resulting in lower performance on text and face reconstruction benchmarks like TokBench.</li>
<li><strong>Holistic vs. Local Encoding:</strong> 1D tokens are often <strong>holistic</strong>, with each token encoding a mix of global and local information. 2D tokens are inherently <strong>local</strong>. This is observable in token correlations: in 2D grids, neighboring tokens are highly correlated, whereas 1D holistic tokens may be more independent.</li>
</ul>
</li>
<li><strong>Effects on AR and Other Tasks:</strong>
<ul>
<li><strong>AR Model Sequence Length:</strong> This is the most direct effect. 2D grids produce long sequences (e.g., 256-1024 tokens), which slows generation. 1D tokenizers like TiTok can reduce this to just 32 tokens, enabling massive speedups and better scalability to higher resolutions.</li>
<li><strong>AR Learnability &amp; Coherence:</strong> 2D raster-scan order can hinder the learning of <strong>global consistency</strong>. As observed in HITA’s experiments, a vanilla AR model can generate an image where one half is a fish and the other is a bird due to this lack of global context. 1D or hybrid approaches with holistic tokens can mitigate this.</li>
<li><strong>AR Model Perplexity:</strong> A highly compressed 1D token that encapsulates complex global content likely has a higher conditional entropy, making it harder for an AR model to predict. However, hybrid models like HITA, which provide a global context token first, can actually <strong>reduce perplexity</strong> for the subsequent local tokens.</li>
</ul>
</li>
<li><strong>Example &amp; Takeaway:</strong><br>
<strong>TiTok (1D)</strong> achieved extremely fast generation and state-of-the-art FID, but the <strong>observed property</strong> was weaker reconstruction of fine details. <strong>HITA (1D+2D hybrid)</strong> introduced holistic tokens before the 2D grid, which had the <strong>effect</strong> of improving global coherence and accelerating AR model convergence. <strong>FlexTok (1D hierarchical)</strong> produced tokens ordered by importance, which had the unique <strong>effect</strong> of enabling progressive, coarse-to-fine generation, a capability impossible with fixed-grid tokenizers.<br>
<strong>Takeaway:</strong> <em>The choice between 1D and 2D token sequences represents a trade-off between compression and a built-in structural prior. 2D grids offer a locality prior beneficial for detail, while 1D sequences offer radical compression and flexibility, enabling faster generation and learned hierarchies.</em></li>
</ul>
<h3 id="causal-structure-in-encoder-and-decoder">4.1.3 Causal Structure in Encoder (and Decoder)</h3>
<ul>
<li><strong>Design Choice:</strong> Whether the tokenizer’s encoder is architecturally designed to output tokens in a specific <strong>causal or hierarchical order</strong> (e.g., using a unidirectional transformer or nested dropout), rather than an unordered set.</li>
<li><strong>Why it matters:</strong> Imposing a causal order on the latent codes—such as a PCA-like ranking of importance—can yield more interpretable and semantically organized tokens. This structure aligns naturally with the sequential nature of AR models, potentially simplifying the learning task.</li>
<li><strong>Observed Properties:</strong>
<ul>
<li><strong>Token Importance Variance:</strong> With a causal encoder, tokens exhibit <strong>diminishing information content</strong>. As demonstrated by Semanticist and FlexTok, the first few tokens carry the bulk of the semantic information, and subsequent tokens add finer details. This is observable as a steep drop in reconstruction error with the first few tokens.</li>
<li><strong>Qualitatively Different Tokens:</strong> Hierarchical tokenizers often learn qualitatively different roles for tokens. HITA’s holistic tokens were observed to encode style and overall shape, while its patch tokens handled fine detail.</li>
<li><strong>Graceful Degradation:</strong> Reconstructions from hierarchical encoders <strong>degrade gracefully</strong> as tokens are removed from the end of the sequence, producing a blurry but semantically correct image rather than a partially missing one.</li>
</ul>
</li>
<li><strong>Effects on AR and Other Tasks:</strong>
<ul>
<li><strong>AR Learnability:</strong> A causal structure simplifies the AR model’s task. By generating the “broad strokes” first, the model has a coherent global context on which to condition the generation of details. HITA observed a 2x convergence speedup with this approach.</li>
<li><strong>Global Coherence:</strong> This design choice drastically reduces the chance of generating incoherent images, as demonstrated by HITA’s ability to solve the “fish-bird” problem in inpainting.</li>
<li><strong>Controllability &amp; Editability:</strong> Hierarchical tokens enable novel forms of control. Semanticist demonstrated <strong>style transfer</strong> by swapping only the first few tokens between images. FlexTok allows for <strong>on-demand, progressive generation</strong>, where a user can generate a quick preview and request more detail if needed.</li>
</ul>
</li>
<li><strong>Example &amp; Takeaway:</strong><br>
<strong>Semanticist</strong>’s design creates a PCA-like token hierarchy. The <strong>observed property</strong> is that dropping later tokens preserves semantic meaning. The <strong>effect</strong> is that AR models require fewer tokens, and the latent space is more interpretable. <strong>FlexTok</strong>’s use of nested dropout yields an ordered token sequence, with the <strong>effect</strong> of enabling AR models to generate a recognizable class object with as few as 8 tokens.<br>
<strong>Takeaway:</strong> <em>Imposing a causal or hierarchical order in the tokenizer’s latent space creates a powerful synergy with AR models. It leads to more coherent generation, faster training, and novel capabilities for interactive control and editing, aligning the generation process more closely with human artistic creation.</em></li>
</ul>
<h3 id="flexible-number-of-tokens-dynamic-length">4.1.4 Flexible Number of Tokens (Dynamic Length)</h3>
<ul>
<li><strong>Design Choice:</strong> Allowing the tokenizer to produce a <strong>variable number of tokens per image</strong>, rather than a fixed-length sequence, adapting to the image’s complexity.</li>
<li><strong>Why it matters:</strong> Not all images are equally complex. A fixed-length tokenizer is inefficient, wasting capacity on simple images while potentially lacking enough for complex ones. A flexible token count allows for a more efficient allocation of resources, similar to variable bitrate encoding in video.</li>
<li><strong>Observed Properties:</strong>
<ul>
<li><strong>Rate-Distortion Behavior:</strong> A flexible tokenizer can be evaluated along a rate-distortion curve, showing how reconstruction quality improves as the token budget increases. This allows for a continuous trade-off, unlike the single operating point of a fixed tokenizer.</li>
<li><strong>Adaptive Token Allocation:</strong> An adaptive tokenizer will naturally use fewer tokens for simple images and more for complex ones. This distribution of token counts can be observed and correlated with image complexity metrics.</li>
</ul>
</li>
<li><strong>Effects on AR and Other Tasks:</strong>
<ul>
<li><strong>Efficiency of AR Generation:</strong> The AR model can terminate generation early for simple prompts or images, saving significant computation. This is a core feature enabled by FlexTok’s design.</li>
<li><strong>User-Controlled Quality vs. Speed Trade-off:</strong> This design enables interactive applications where a user can generate a fast preview and then request a higher-fidelity version by having the model generate more tokens.</li>
<li><strong>Improved Average Performance:</strong> For a given average token budget, an adaptive tokenizer can allocate its resources more intelligently across a dataset, leading to better overall FID and LPIPS scores compared to a fixed-length baseline.</li>
<li><strong>Robustness to Complexity Extremes:</strong> An adaptive tokenizer can handle extremely detailed images by simply generating more tokens, avoiding the “quality cliff” where a fixed tokenizer runs out of capacity and fails.</li>
</ul>
</li>
<li><strong>Example &amp; Takeaway:</strong><br>
<strong>FlexTok</strong> is the prime example, where its design explicitly allows for a variable-length output. The <strong>observed property</strong> is that images transition from coarse outlines to full detail as the number of tokens increases from ~20 to ~300. The <strong>effect</strong> is the ability to perform on-demand, progressive generation with an AR model.<br>
<strong>Takeaway:</strong> <em>Variable token counts make a tokenizer more data-efficient and versatile. This design leads to dynamic control over the speed-quality trade-off, better utilization of modeling capacity, and more robust performance across images of varying complexity, making AR models more efficient and adaptive.</em></li>
</ul>
<h3 id="semantic-forcing-and-external-guidance">4.1.5 Semantic Forcing and External Guidance</h3>
<ul>
<li><strong>Design Choice:</strong> Incorporating <strong>semantic knowledge</strong> into the tokenizer’s training, typically by aligning its representations with features from a powerful pretrained foundation model like CLIP or DINOv2.</li>
<li><strong>Why it matters:</strong> A standard VQ tokenizer is only optimized for reconstruction and does not guarantee that its tokens will be semantically meaningful. Adding a semantic loss encourages the tokens to encode human-relevant concepts, which improves their utility for downstream tasks and can aid in generating more coherent compositions.</li>
<li><strong>Observed Properties:</strong>
<ul>
<li><strong>Semantic Clustering:</strong> The token embeddings from a semantically-guided tokenizer will exhibit stronger clustering by semantic class.</li>
<li><strong>Improved Understanding Metrics:</strong> These tokenizers show higher <strong>zero-shot classification accuracy</strong> or text-image retrieval scores. UniTok, for example, achieved 78.6% accuracy on ImageNet, outperforming CLIP itself.</li>
<li><strong>Linearly Separable Representations:</strong> The token representations become more linearly separable, as demonstrated by the strong performance of simple linear probes on the embeddings from models like Semanticist and UniTok.</li>
</ul>
</li>
<li><strong>Effects on AR and Other Tasks:</strong>
<ul>
<li><strong>Bridging Generation and Understanding:</strong> The most significant effect is <strong>closing the gap between discrete and continuous representations for understanding tasks</strong>. As shown by TokenFlow, a multimodal model using its discrete tokens could outperform LLaVA, which uses continuous features.</li>
<li><strong>Improved Generative Coherence:</strong> By encoding clear semantic concepts, these tokens may help AR models with complex compositional tasks, such as correctly binding attributes to objects.</li>
<li><strong>Faster Convergence:</strong> Using a semantic loss, such as the REPA loss in FlexTok, can provide a strong, high-level learning signal that <strong>greatly accelerates the convergence</strong> of the tokenizer’s training.</li>
</ul>
</li>
<li><strong>Example &amp; Takeaway:</strong><br>
<strong>UniTok</strong>’s use of multi-codebooks and CLIP supervision resulted in tokens that excelled at both reconstruction and classification. <strong>TokenFlow</strong>’s dual-codebook design, with one codebook for semantics and one for detail, enabled its discrete tokens to be used in a model that surpassed a strong continuous-feature baseline in VQA.<br>
<strong>Takeaway:</strong> <em>Semantic guidance ensures that discrete representations carry meaningful, human-interpretable information. This enhances AR generation, enables the direct use of tokens for multimodal understanding tasks, and effectively unifies generative and discriminative representations within a single, powerful tokenizer.</em></li>
</ul>
<h2 id="application-of-the-framework-to-key-tokenizers">4.2 Application of the Framework to Key Tokenizers</h2>
<h3 id="vq-vae">4.2.1 VQ-VAE</h3>
<ul>
<li><strong>Design Choices:</strong> 2D grid sequence; standard convolutional encoder/decoder; key innovation was the discrete bottleneck with a straight-through estimator.</li>
<li><strong>Observed Properties:</strong> Small codebook size (K=512); prone to <strong>codebook collapse</strong>; blurry reconstructions due to MSE loss.</li>
<li><strong>Effects:</strong> Successfully enabled AR priors by avoiding posterior collapse, but final generation quality was limited by the poor reconstruction baseline.</li>
</ul>
<h3 id="vqgan">4.2.2 VQGAN</h3>
<ul>
<li><strong>Design Choices:</strong> Maintained the 2D grid; key innovation was <strong>semantic forcing</strong> via a PatchGAN adversarial loss and a perceptual (LPIPS) loss.</li>
<li><strong>Observed Properties:</strong> Supported larger codebooks (up to 16,384); dramatically improved reconstruction quality (rFID), resulting in sharp, textured images.</li>
<li><strong>Effects:</strong> Provided a high-fidelity foundation that made megapixel-scale AR generation practical; adversarial training improved decoder robustness.</li>
</ul>
<h3 id="llamagen">4.2.3 LlamaGen</h3>
<ul>
<li><strong>Design Choices:</strong> A scaling achievement using a 2D grid and a standard Llama architecture with 2D positional embeddings.</li>
<li><strong>Observed Properties:</strong> High codebook utilization (97%); strong reconstruction quality (rFID of 0.94) from its improved VQGAN-style tokenizer.</li>
<li><strong>Effects:</strong> Proved that standard LLM architectures scale effectively for visual token modeling, achieving a gFID of 2.18 that surpassed leading diffusion models.</li>
</ul>
<h3 id="titok">4.2.4 TiTok</h3>
<ul>
<li><strong>Design Choices:</strong> Key innovation was shifting to a highly compressed <strong>1D sequence</strong> using a Transformer-based encoder with learnable queries.</li>
<li><strong>Observed Properties:</strong> Extremely low, fixed token count (e.g., 32); good overall reconstruction (rFID 2.21) but traded some fine-detail fidelity for compression.</li>
<li><strong>Effects:</strong> A massive acceleration in AR inference speed (hundreds of times faster); simplified the AR learning task; achieved excellent gFID (1.97).</li>
</ul>
<h3 id="flextok">4.2.5 FlexTok</h3>
<ul>
<li><strong>Design Choices:</strong> A <strong>1D ordered sequence</strong> with <strong>variable length</strong>, enabled by nested dropout; employed <strong>semantic forcing</strong> via a rectified flow decoder and a REPA (DINOv2 alignment) loss.</li>
<li><strong>Observed Properties:</strong> An emergent <strong>hierarchical token structure</strong> (coarse-to-fine); demonstrated adaptive compression based on image complexity.</li>
<li><strong>Effects:</strong> Enabled <strong>adaptive and progressive generation</strong>, allowing for a dynamic trade-off between speed and quality.</li>
</ul>
<h3 id="semanticist">4.2.6 Semanticist</h3>
<ul>
<li><strong>Design Choices:</strong> A <strong>1D causal sequence</strong> with a mathematically guaranteed <strong>PCA-like structure</strong>, enforced by a nested CFG strategy and a diffusion-based decoder.</li>
<li><strong>Observed Properties:</strong> A highly structured latent space with <strong>semantic-spectrum decoupling</strong>; tokens are ordered by importance, contributing orthogonal information.</li>
<li><strong>Effects:</strong> The semantically pure “language” is very easy for an AR model to learn; tokens are highly effective for downstream classification (63.5% accuracy from a linear probe) and provide a more interpretable representation.</li>
</ul>
<h2 id="comparison-of-key-visual-tokenizers">4.3 Comparison of Key Visual Tokenizers</h2>
<p>This table provides a comparative overview of the key VQ tokenizers discussed, summarizing their architectural designs, observed performance properties, and their downstream effects on autoregressive models and related tasks.</p>

<table>
<thead>
<tr>
<th align="left">Tokenizer (Year)</th>
<th align="left">Sequence &amp; Tokens (256px)</th>
<th align="left">Key Innovation</th>
<th align="left">Semantic Guidance / Loss</th>
<th align="left">rFID (Recon.)</th>
<th align="left">Key Properties</th>
<th align="left">Key Effects &amp; Gen. FID</th>
<th align="left">Availability</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Core Tokenizers</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>VQ-VAE</strong> (2017)</td>
<td align="left">2D Grid (e.g., 32x32)</td>
<td align="left">Foundational discrete VAE bottleneck.</td>
<td align="left">Pixel-wise MSE</td>
<td align="left">High (~15-20), blurry</td>
<td align="left">Small codebook (512-1024), prone to <strong>codebook collapse</strong>.</td>
<td align="left">Enabled AR models on discrete codes but with limited generation quality.</td>
<td align="left"><a href="https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py">Multiple Implementations</a></td>
</tr>
<tr>
<td align="left"><strong>VQGAN</strong> (2021)</td>
<td align="left">2D Grid (e.g., 16x16)</td>
<td align="left">Augments VQ-VAE with GAN + perceptual losses.</td>
<td align="left">Adversarial + LPIPS Loss</td>
<td align="left">Low (~5-8), sharp</td>
<td align="left">Larger codebook (1024-16,384), captures high-frequency patterns.</td>
<td align="left">Enabled high-fidelity, photorealistic AR generation. Became foundational.</td>
<td align="left"><a href="https://github.com/CompVis/taming-transformers">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>DALL·E dVAE</strong> (2021)</td>
<td align="left">2D Grid (32x32) / 1024 Tokens</td>
<td align="left">Scaled for massive text-to-image training.</td>
<td align="left">Perceptual Loss (inferred)</td>
<td align="left">Excellent (reported ~0.13)</td>
<td align="left">Large codebook (8,192), high code usage due to massive dataset.</td>
<td align="left">Powered DALL·E 1, proving AR models can handle complex prompts.</td>
<td align="left"><a href="https://github.com/openai/dalle-2">Tokenizer Weights Available</a></td>
</tr>
<tr>
<td align="left"><strong>LlamaGen VQ</strong> (2024)</td>
<td align="left">2D Grid (e.g., 24x24)</td>
<td align="left">Careful scaling of VQGAN for a large AR model.</td>
<td align="left">Perceptual Loss</td>
<td align="left">Very low (0.94)</td>
<td align="left">Large codebook (16,384), high usage (~97%).</td>
<td align="left">Enabled SOTA <strong>gFID (~2.18)</strong>, proving scaled AR can outperform diffusion.</td>
<td align="left"><a href="https://github.com/FoundationVision/LlamaGen">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>Chameleon</strong> (2023)</td>
<td align="left">2D Grid (32x32) / 1024 Tokens</td>
<td align="left">Designed for early-fusion multimodal models.</td>
<td align="left">Perceptual Loss + Biased training data (oversampled faces).</td>
<td align="left">N/A</td>
<td align="left">Improved face reconstruction; struggled with small text/OCR.</td>
<td align="left">Enabled a single model for interleaved image/text generation &amp; understanding.</td>
<td align="left"><a href="https://huggingface.co/facebook/chameleon-7b">Model on HuggingFace</a></td>
</tr>
<tr>
<td align="left"><strong>TiTok</strong> (2024)</td>
<td align="left">1D Sequence / 32 Tokens</td>
<td align="left">Ultra-compact representation via Transformer encoder.</td>
<td align="left">Perceptual Loss</td>
<td align="left">Good (2.21)</td>
<td align="left">Sacrifices fine detail (text/faces) for extreme compression.</td>
<td align="left">Drastically accelerates AR gen. (&gt;400x). Achieved SOTA <strong>gFID (1.97)</strong>.</td>
<td align="left"><a href="https://yucornetto.github.io/TiTok/">Official Code &amp; Demo</a></td>
</tr>
<tr>
<td align="left"><strong>SoftVQ</strong> (2024)</td>
<td align="left">1D Continuous Sequence</td>
<td align="left">“Soft” quantization (weighted mixture of codewords).</td>
<td align="left">DINO alignment loss</td>
<td align="left">FID 1.78 (with DiT)</td>
<td align="left">High representational capacity per token; fully differentiable.</td>
<td align="left">Accelerates training and inference (&gt;18x faster). Loses discrete indexing. <strong>gFID 1.78</strong> (with DiT).</td>
<td align="left"><a href="https://github.com/kakaobrain/soft-vq">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>XQ-GAN</strong> (2024)</td>
<td align="left">2D Grid</td>
<td align="left">Extensible framework combining multiple quantization methods (RQ, PQ, etc.).</td>
<td align="left">Supports CLIP/DINOv2 alignment.</td>
<td align="left">Excellent (0.64)</td>
<td align="left">Highly flexible, allowing trade-offs between different objectives.</td>
<td align="left">Strong open-source baseline. Used with VAR model to achieve <strong>gFID of 2.60</strong>.</td>
<td align="left"><a href="https://github.com/kent-lcc/XQ-GAN">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>UniTok</strong> (2025)</td>
<td align="left">2D Grid with 1D codes</td>
<td align="left">Multi-codebook quantization to expand vocabulary.</td>
<td align="left">Combined reconstruction + CLIP objectives.</td>
<td align="left">SOTA (0.38)</td>
<td align="left">Semantically rich (78.6% zero-shot acc.), preserves fine details.</td>
<td align="left">Unified generation and understanding; tokens are powerful features.</td>
<td align="left"><a href="https://github.com/FoundationVision/UniTok">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>FlexTok</strong> (2025)</td>
<td align="left">1D Ordered Seq. / Variable (8-256)</td>
<td align="left">Variable-length tokens via nested dropout.</td>
<td align="left">REPA (DINOv2) loss on rectified flow decoder.</td>
<td align="left">Variable</td>
<td align="left">Hierarchical tokens (coarse-to-fine).</td>
<td align="left">Enables adaptive, progressive AR generation; user can trade speed vs. quality.</td>
<td align="left"><a href="https://flextok.epfl.ch/">Official Code &amp; Project Page</a></td>
</tr>
<tr>
<td align="left"><strong>Semanticist</strong> (2025)</td>
<td align="left">1D Causal Seq.</td>
<td align="left">Enforces a PCA-like structure on tokens; diffusion decoder.</td>
<td align="left">Nested CFG training for semantic-spectrum decoupling.</td>
<td align="left">SOTA (0.72)</td>
<td align="left">Highly structured, interpretable latent space. Strong linear separability (63.5% acc.).</td>
<td align="left">Simplifies AR learning (<strong>gFID 2.57</strong> with 32 tokens). Enables semantic editing.</td>
<td align="left"><a href="https://visual-gen.github.io/semanticist/">Project Page</a></td>
</tr>
<tr>
<td align="left"><strong>Extended Tokenizers</strong></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
<td align="left"></td>
</tr>
<tr>
<td align="left"><strong>HITA</strong> (2025)</td>
<td align="left">Hybrid 1D+2D (Holistic + Patch)</td>
<td align="left">Generates global “holistic” tokens first for context.</td>
<td align="left">Injects DINOv2 features into holistic tokens.</td>
<td align="left">N/A</td>
<td align="left">Holistic tokens capture global semantics. AR model trains ~2x faster.</td>
<td align="left">Improves global coherence; enables zero-shot inpainting/style transfer.</td>
<td align="left"><a href="https://github.com/CVMI-Lab/Hita">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>TokenFlow</strong> (2025)</td>
<td align="left">2D Grid</td>
<td align="left">Dual-codebook architecture (semantic + pixel) with a shared index.</td>
<td align="left">Semantic codebook aligned with CLIP.</td>
<td align="left">Excellent (0.63 @ 384px)</td>
<td align="left">Decoupled representation for semantics and detail.</td>
<td align="left">First discrete tokenizer to enable a model to surpass LLaVA on understanding tasks.</td>
<td align="left"><a href="https://github.com/ByteFlow-AI/TokenFlow">Official Code (Announced)</a></td>
</tr>
<tr>
<td align="left"><strong>IBQ</strong> (2024)</td>
<td align="left">2D Grid</td>
<td align="left">Index Backpropagation for training massive codebooks without collapse.</td>
<td align="left">Perceptual Loss</td>
<td align="left">Extremely high quality</td>
<td align="left">Near-perfect codebook utilization (~96%) on huge codebooks (up to 262k).</td>
<td align="left">Removes codebook size bottleneck. AR models achieved <strong>gFID of ~2.05</strong>.</td>
<td align="left"><a href="https://github.com/TencentARC/SEED-Voken">Official Code &amp; Models</a></td>
</tr>
<tr>
<td align="left"><strong>GigaTok</strong> (2025)</td>
<td align="left">1D Sequence</td>
<td align="left">A 2.9B parameter tokenizer, proving scalability.</td>
<td align="left">Semantic regularization with DINOv2.</td>
<td align="left">SOTA</td>
<td align="left">Semantically organized latent space even at massive scale.</td>
<td align="left">SOTA AR generation (<strong>gFID ~1.7-2.0</strong>). Scaling tokenizer improves both gen &amp; understanding.</td>
<td align="left"><a href="https://github.com/silentview/GigaTok">Official Code &amp; Models</a></td>
</tr>
</tbody>
</table><h3 id="references-4">References</h3>
<p>Esser, P., Rombach, R., &amp; Ommer, B. (2021). Taming Transformers for High-Resolution Image Synthesis. In <em>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</em> (CVPR).</p>
<p>Gu, J., et al. (2024). SeQ-GAN: Semantic-Quantized GAN for High-Fidelity Image Generation. <em>arXiv preprint arXiv:2404.03043</em>.</p>
<p>Li, Y., et al. (2024). TokenFlow: A Unified Tokenizer for Vision and Language. <em>arXiv preprint arXiv:2412.03069</em>.</p>
<p>Ma, Y., Zhang, X., Li, H., &amp; Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. <em>Project Page available at <a href="http://foundationvision.github.io">foundationvision.github.io</a></em>.</p>
<p>Razavi, A., van den Oord, A., &amp; Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. In <em>Advances in Neural Information Processing Systems 32</em> (NeurIPS 2019).</p>
<p>Sun, J., Wang, R., Li, Y., Wang, W., &amp; Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. <em>arXiv preprint arXiv:2404.09344</em>.</p>
<p>Tian, Y., et al. (2024). VAR: Visual Autoregressive Modeling with Scalable Tokenization. In <em>International Conference on Learning Representations</em> (ICLR 2025).</p>
<p>TokBench Collaborators. (2025). TokBench: A Comprehensive Benchmark for Visual Tokenizers. <em>arXiv preprint arXiv:2505.18142v2</em>.</p>
<p>Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. <em>Project Page available at <a href="http://visual-gen.github.io">visual-gen.github.io</a></em>.</p>
<p>Zheng, Y., et al. (2025). HITA: A Holistic Image Tokenizer for Autoregressive Generation. <em>arXiv preprint arXiv:2507.02358v1</em>.</p>
<h1 id="frontiers-tensions-and-emerging-themes">5. Frontiers, Tensions, and Emerging Themes</h1>
<p>This final section outlines the key challenges, emerging themes, and research gaps that will likely shape the next generation of models.</p>
<h2 id="key-tensions-and-foundational-trade-offs">5.1 Key Tensions and Foundational Trade-offs</h2>
<p>The design of visual tokenizers is governed by a set of competing priorities. These core tensions represent the fundamental trade-offs that researchers must navigate.</p>
<ul>
<li>
<p><strong>The Reconstruction vs. Generation Dilemma:</strong> This remains the most critical tension. As highlighted by GigaTok, naively scaling a tokenizer to achieve better reconstruction fidelity (lower rFID) can produce an exponentially more complex latent space. This makes the distribution of token sequences harder for the AR model to learn, leading to higher perplexity and, paradoxically, worse final generation quality (higher gFID). Managing this trade-off is a central design challenge.</p>
</li>
<li>
<p><strong>Sequence Length vs. Information Richness:</strong> There is an ongoing debate between using a short sequence of highly information-dense tokens (TiTok) versus a longer sequence of simpler tokens. While short sequences offer massive efficiency gains, longer sequences may be necessary for complex prompts that require fine-grained detail (FlexTok). The optimal balance is unclear and may be task-dependent.</p>
</li>
<li>
<p><strong>Unified vs. Specialized Tokenizers:</strong> The push towards unified models (UniTok, TokenFlow) that serve both understanding and generation is a compelling research direction. However, this introduces complexity (e.g., dual codebooks, multi-objective losses) and raises the question of a <strong>generalization vs. specialization tension</strong>. It remains an open question whether a single “jack-of-all-trades” tokenizer can consistently outperform specialized models. For instance, a tokenizer optimized purely for artistic generation might prioritize texture fidelity over semantic accuracy, a trade-off a unified model might not make.</p>
</li>
</ul>
<h2 id="open-research-questions-and-future-directions">5.2 Open Research Questions and Future Directions</h2>
<p>Several open questions and research gaps remain, pointing toward exciting avenues for future work.</p>
<h3 id="out-of-distribution-ood-robustness">5.2.1 Out-of-Distribution (OOD) Robustness</h3>
<p>The vast majority of research evaluates tokenizers on in-distribution datasets. There is a significant lack of investigation into how these models perform on OOD images (e.g., medical imagery, abstract art, sketches), their robustness to common corruptions, or their vulnerability to adversarial attacks. Future work could introduce adaptive codebooks or fine-tuning methods to adapt tokenizers to new domains without catastrophic forgetting.</p>
<h3 id="compositionality-disentanglement-and-spatial-reasoning">5.2.2 Compositionality, Disentanglement, and Spatial Reasoning</h3>
<p>AR models still falter on prompts requiring complex spatial arrangements or precise counting. This may stem from tokenizing by arbitrary patches rather than semantic entities. An emerging research direction is <strong>object-centric and disentangled tokenization</strong>. Instead of a single token trying to represent shape, texture, and semantics all at once, future tokenizers might produce multiple, parallel token streams—one for geometry, one for appearance, one for high-level concepts. This could involve channel-wise VQ or hybrid VQ-segmentation models to align tokens more closely with how humans describe scenes—by objects and their relations.</p>
<h3 id="hybrid-autoregressive-and-diffusion-models">5.2.3 Hybrid Autoregressive and Diffusion Models</h3>
<p>The line between AR and diffusion paradigms is blurring. Semanticist’s use of a diffusion decoder hints at powerful hybrid models: AR for the high-level, structured sequence of concepts, and diffusion for the low-level pixel synthesis. This opens up possibilities for multi-stage generation: an AR model could generate a coarse token sequence, which is then refined by a diffusion model or even a second “error correction” AR model. Understanding the optimal way to combine the speed and structure of AR with the refinement capabilities of diffusion is a key frontier.</p>
<h3 id="multi-scale-and-hierarchical-generation">5.2.4 Multi-Scale and Hierarchical Generation</h3>
<p>Beyond a single sequence, <strong>multi-scale tokenization</strong> offers a promising path. Models like VAR have explored tokenizing an image at multiple resolutions, allowing an AR model to predict a sequence of scale-wise tokens. This could enable more efficient handling of both global layouts (via coarse tokens) and fine details (via fine-grained tokens), potentially improving both quality and generation speed for high-resolution images.</p>
<h3 id="multi-modality-and-unified-models">5.2.5 Multi-Modality and Unified Models</h3>
<p>While models like TokenFlow have begun to unify vision and text, the next frontier is incorporating other modalities like audio, video, and 3D. An open challenge is creating a single tokenizer and AR model that can process and generate arbitrarily interleaved sequences of image, text, and audio tokens, as explored by early work like AToken. This raises fundamental questions about whether a universal codebook is feasible or if modality-specific vocabularies are necessary.</p>
<h3 id="tokenization-for-fine-tuning-and-personalization">5.2.6 Tokenization for Fine-Tuning and Personalization</h3>
<p>Diffusion models have powerful personalization techniques like DreamBooth. Developing analogous methods for VQ-AR models is a key open challenge. This might involve learning new codebook entries for a specific object or style and fine-tuning the AR model to use these new “visual words.” Research into few-shot learning with discrete visual vocabularies is needed to enable efficient personalization without catastrophic forgetting.</p>
<h2 id="emerging-architectural-and-conceptual-themes">5.3 Emerging Architectural and Conceptual Themes</h2>
<p>Several powerful concepts are emerging that redefine the role of the tokenizer and its components.</p>
<ul>
<li>
<p><strong>Structured Latent Spaces:</strong> There is a clear trend away from an unstructured “bag of tokens” toward sequences with inherent order and meaning. The hierarchical structures in FlexTok and the mathematically-guaranteed, PCA-like orthogonality in Semanticist are prime examples. Future research will likely explore more sophisticated structures, such as graph-based representations, to make the latent space even more interpretable and efficient.</p>
</li>
<li>
<p><strong>The Evolving Role of the Decoder:</strong> The tokenizer’s decoder is no longer just a simple upsampling network. In VQGAN, it became an adversarial player. In FlexTok, it is a rectified flow model capable of decoding partial sequences. In Semanticist, it is a diffusion model used to enforce semantic-spectral decoupling. The decoder is increasingly being used as an active tool during training to impose desirable structural and semantic properties onto the latent space itself.</p>
</li>
<li>
<p><strong>Continuous vs. Discrete Revisited:</strong> While discrete VQ tokenizers have dominated, models like SoftVQ are re-introducing continuous representations. By allowing a token to be a “soft” mixture of multiple codebook entries, these models aim to increase representational capacity and avoid information loss from hard quantization. The future may lie in hybrid approaches that combine the structured nature of discrete tokens with the expressive power of continuous representations.</p>
</li>
</ul>
<h2 id="gaps-in-methodology-and-evaluation">5.4 Gaps in Methodology and Evaluation</h2>
<p>Progress in the field is hampered by several gaps in how models are developed and evaluated.</p>
<ul>
<li><strong>Standardizing Evaluation:</strong> There is a critical need for more comprehensive and standardized benchmarks. This includes <strong>OOD benchmarks</strong> that test generalization beyond natural images, <strong>compositional metrics</strong> to evaluate spatial accuracy and object counting, and a more rigorous protocol for <strong>FID reporting</strong> to ensure comparability across publications.</li>
<li><strong>AR Learnability Metrics:</strong> The field lacks a standard metric for comparing the “language complexity” each tokenizer produces. The “AR probing” approach from GigaTok, which evaluates a fixed, lightweight AR model on different tokenizers, could provide a much-needed, apples-to-apples comparison of AR learnability.</li>
<li><strong>Theoretical Understanding:</strong> The theoretical frameworks for predicting tokenizer performance from design choices remain limited. A deeper theoretical understanding could accelerate progress and reduce reliance on empirical trial-and-error.</li>
<li><strong>Practical and Methodological Gaps:</strong> Key factors like <strong>training stability</strong>, <strong>computational efficiency</strong>, and <strong>architectural compatibility</strong> with different downstream models are often under-reported but are crucial for practical adoption. Furthermore, <strong>human perceptual alignment</strong> remains an underexplored evaluation dimension that could reveal biases in our current automated metrics.</li>
</ul>
<h2 id="the-future-of-visual-language">5.5 The Future of Visual Language</h2>
<p>The journey of VQ tokenizers—from VQ-VAE’s blurry patches to the semantic-rich, flexible sequences of today—has been remarkable. The field is converging on a powerful “visual sentence” metaphor. The progression from 2D grids (a “page” of tokens) to 1D sequences (a simple “sentence”) and now to ordered, hierarchical sequences (a “grammatically structured sentence”) indicates a drive to create a true, compositional language of images.</p>
<p>As the tokenizer becomes more adept at creating this structured language, the role of the AR model may shift from that of a raw creator to a logical synthesizer, tasked with arranging these meaningful visual components into a coherent narrative. Answering the open questions outlined above will likely involve an interplay of ideas: merging paradigms (AR + diffusion hybrids), drawing from NLP techniques for images, and rethinking the very notion of visual “tokens” to be more aligned with human perception. The coming years will determine if autoregressive image generation, armed with these advanced tokenizers, can truly become the dominant approach for generative AI.</p>
<hr>
<h3 id="references-5">References</h3>
<p>(Note: This list combines all unique sources cited in the provided text for this section.)</p>
<p>Chang, H., et al. (2023). MaskGIT: Masked Generative Image Transformer. In <em>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition</em> (CVPR).</p>
<p>Gao, P., et al. (2024). ACDiT: Autoregressive Conditional Diffusion Transformers for High-Fidelity Image Synthesis. <em>arXiv preprint arXiv:2404.09559</em>.</p>
<p>Li, Y., et al. (2024). TokenFlow: A Unified Tokenizer for Vision and Language. <em>arXiv preprint arXiv:2412.03069</em>.</p>
<p>Ma, Y., Zhang, X., Li, H., &amp; Wang, L. (2025). UniTok: A Unified Tokenizer for Generation and Understanding. <em>Project Page available at <a href="http://foundationvision.github.io">foundationvision.github.io</a></em>.</p>
<p>Sun, J., Wang, R., Li, Y., Wang, W., &amp; Li, H. (2024). LlamaGen: An Autoregressive Model for High-Performance Image Generation. <em>arXiv preprint arXiv:2404.09344</em>.</p>
<p>Tian, Y., et al. (2024). VAR: Visual Autoregressive Modeling with Scalable Tokenization. In <em>International Conference on Learning Representations</em> (ICLR 2025).</p>
<p>TokBench Collaborators. (2025). TokBench: A Comprehensive Benchmark for Visual Tokenizers. <em>arXiv preprint arXiv:2505.18142v2</em>.</p>
<p>Wang, Z., Zhao, Y., et al. (2025). Semanticist: Hierarchical Semantic-Spectral Quantization for Generative Modeling. <em>Project Page available at <a href="http://visual-gen.github.io">visual-gen.github.io</a></em>.</p>
<p>Yu, Z., et al. (2025). GigaTok: A Billion-Parameter Visual Tokenizer. <em>Project Page available at <a href="http://silentview.github.io">silentview.github.io</a></em>.</p>

