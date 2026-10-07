# Dependencies and attribution

The package holds the project's own runners and analysis, not upstream architectures or training pipelines. External implementations and their licenses remain in their respective repositories:

- LlamaGen: https://github.com/FoundationVision/LlamaGen ; recorded working fork https://github.com/farzinnasiri/LlamaGen
- VQGAN / taming-transformers: https://github.com/CompVis/taming-transformers ; recorded working fork https://github.com/farzinnasiri/taming-transformers
- FlexTok: https://github.com/apple/ml-flextok
- One-D-Piece: https://github.com/turingmotors/One-D-Piece
- OpenAI feature-space evaluation: https://github.com/openai/guided-diffusion/tree/main/evaluations
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- Alternative torch-fidelity evaluation: https://github.com/toshas/torch-fidelity

`configs/upstreams.lock.json` records source commits actually found in the project/server checkouts. The runner files were written or modified for these experiments; helper routines invoke upstream model and preprocessing APIs. Preserve upstream license notices if extracting or redistributing upstream code separately. Dataset access remains subject to each dataset's terms. No dataset images or pretrained checkpoint weights are committed here.
