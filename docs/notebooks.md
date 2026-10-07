# Notebook reproduction

The canonical notebooks are latest **local** RQ1–RQ5 sources (including the September thesis-review corrections), plus the **15 September server** ordered-prefix notebook. The latter supersedes the stale local copy. Code outputs are cleared in Git, not in the original notebooks.

Always run from a separate prepared `notebooks/` working directory. The top-level path assumptions in the analyses depend on that directory name.

```bash
export VQ_NAS_ROOT=/megaverse/storage/nasiri
python tools/prepare_workspace.py --nas-root "$VQ_NAS_ROOT" --workspace work/analysis
python tools/run_notebook.py work/analysis/notebooks/ch04_rq1_reconstruction_codebook_usage.ipynb
```

Repeat the final command with the desired filename. `tools/run_notebook.py` executes top to bottom, fails on cell errors, and writes a separate `.executed.ipynb`. For interactive work, launch Jupyter from `work/analysis/notebooks`. Do not execute a canonical notebook inside the read-only archive or point output/cache directories there.

| Notebook | Inputs | Outputs / notes |
| --- | --- | --- |
| `ch04_rq1_reconstruction_codebook_usage` | Code-usage exports, reconstruction metrics CSV, RQ2 clean per-image PSNR rows | `chapter4_outputs`; the preparer copies existing RQ2 rows. If absent, run RQ2 first. |
| `ch05_rq2_global_robustness` | Both dated global-noise image/metadata exports | `chapter5_outputs`; recomputes per-image summaries and writes the RQ1 PSNR input. Latest source includes the direct-RMSE audit. |
| `ch06_rq3_local_image_perturbations` | `jupyter/data/h1_encoder_locality`, codebooks, saved reconstructions | `chapter6_outputs`; image and LPIPS calculations can be expensive without existing caches. |
| `ch07_rq4_token_space_perturbations` | `jupyter/data/h2_decoder_locality/{raw,analysis_cache,relations}` | `chapter7_outputs`; requires the saved merged decoder-locality metrics and variants. |
| `ch08_rq5_distribution_shift` | Code-usage counts, reconstruction CSV, global clean per-image token metadata | `chapter8_outputs`; sample-matched entropy check validates that token grids reconstruct archived counts. |
| `ch09_ordered_prefix_distribution_shift` | Four ordered-prefix CSVs and saved qualitative images under archived `chapter9_outputs/data/raw` | `chapter9_outputs/{data/derived,figures}/focused_v3`; jointly sampled 5,000 class-bootstrap draws, seed 0. |

`prepare_workspace.py` links raw inputs and copies notebooks. It does not execute anything or populate every analysis cache. RQ2 can precede RQ1; RQ3–RQ5 are otherwise independent once their raw exports/caches exist. Ordered-prefix analysis is independent of the thesis RQ1–RQ5 runs. Set `CH08_METADATA_BASE` if the clean global exports are mounted somewhere other than the configured Jupyter input root.

`notebooks/exploratory/` retains seven earlier analyses. They are **not** the canonical reproduction path and can rely on old mount names, cached state or manual cells. Run an individual exploratory notebook from the prepared `notebooks/` directory if it assumes that working directory, and inspect its configuration before executing. No claim of clean execution is made for every historical notebook.

The notebook preparation script changes path constructors/defaults only, adds a storage setup cell and clears outputs. Statistical formulas and original cell order remain unchanged. A subsequent ch08 plotting-only fix gives the six-domain spatial plot its own x-axis variables, avoiding state left by the preceding three-domain check. Saved executed originals and source hashes are retained in the NAS handover snapshot; do not mistake output clearing for new results.
