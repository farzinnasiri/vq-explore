# Discrete image tokenizer properties

Experiment runners and analysis notebooks for **An Analysis of Discrete Image Tokenizers under Perturbations and Distribution Shifts**, and the later ordered-prefix reconstruction study.

This repository contains Farzin Nasiri's evaluation/export scripts, notebooks, small result tables, and a supervisor report. It does **not** vendor LlamaGen, VQGAN, FlexTok, One-D-Piece, pretrained weights, or datasets. Model implementations are fetched separately at recorded revisions. Heavy inputs and executed originals remain on the AIRLab NAS.

## Start here

- [Merged supervisor report](reports/ordered-prefix/ordered-prefix-ood.html): figures are embedded, so the HTML can be shared alone. No server or internet connection is needed to read it. Original figure files are also retained in `assets/`.
- [Environment and external models](docs/setup.md)
- [Run the thesis experiments](docs/thesis-experiments.md)
- [Run the ordered-prefix experiments](experiments/ordered_prefix_ood/README.md)
- [Execute the analysis notebooks](docs/notebooks.md)
- [NAS layout and saved inputs](docs/nas.md)
- [Provenance and verification](docs/provenance.md)

## Layout

| Path | Contents |
| --- | --- |
| `scripts/llamagen`, `scripts/vqgan` | Reconstruction, code usage, codebook relations, global noise, encoder locality and decoder locality exporters |
| `scripts/evaluation` | Reference NPZ creation, external OpenAI evaluator launcher, alternative torch-fidelity wrapper |
| `scripts/analysis` | Decoder-locality cache builder and historical notebook assembly helpers |
| `scripts/datasets` | ImageNet-V2 class-directory layout helper |
| `scripts/thesis_figures` | Thesis plot and supporting-results generators |
| `notebooks` | Latest RQ1–RQ5 notebooks and ordered-prefix analysis, with outputs cleared |
| `notebooks/exploratory` | Earlier exploratory analyses, retained separately |
| `experiments/ordered_prefix_ood` | Main runners, Docker build recipes, exact cohort manifests and clearly labelled historical scripts |
| `results` | Small saved tables for inspecting results without the full NAS data |
| `reports` | Offline supervisor HTML and figures |
| `configs`, `tools` | External revision lock, NAS index, workspace preparation and notebook execution |

The personal and AIRLab repositories publish the same working tree. The personal repository retains its older history; the AIRLab repository starts with the curated package so that previous image exports are not carried into its history. The original working directories have not been cleaned or reset.

## What reproduction means here

**Reanalyse saved exports:** prepare a writable workspace and run the notebooks using NAS inputs. This does not invoke the model exporters.

**Generate new exports:** install each model in its own environment, select the documented checkpoint and dataset layout, then run the exporters and analysis. Full GPU inference was not repeated during packaging. Historical stochastic interventions cannot be made retrospectively paired by rerunning the analysis; preserve recorded batch membership and settings when comparing results.

Datasets retain their original access/licensing conditions. Upstream repositories retain their own licenses; see [attribution](docs/attribution.md). No general redistribution license has been invented for this handover.
