# Package provenance

Prepared 7 October 2026 from local project files and a direct Elysium inventory. No new dataset experiments were commissioned during packaging. Original dirty working trees, original notebooks and heavy NAS data were preserved.

## Source selection

- LlamaGen/VQGAN exporters: local runner source agrees byte-for-byte with the corresponding server scripts before portability edits. `eval_fidelity.py` was recovered from the recorded server fork commit; no upstream evaluator source is vendored.
- Canonical RQ1–RQ5 notebooks: latest local `vq-explore/notebooks`, including later direct-RMSE and sample-matched entropy checks. Older server copies were not used to replace these sources.
- Ordered-prefix notebook: server `jupyter/vq-explore/notebooks/ch09_ordered_prefix_distribution_shift.ipynb`, dated 15 September, `focused_v3` analysis. The server-only batched runner and shard manifests were included.
- Historical extended/pilot/benchmark scripts are separated from canonical runners rather than silently discarded or relabelled.
- Thesis figure and appendix helpers plus their full input Assets tree are retained (code in Git; assets on NAS).
- Report source decisions and the resolution of earlier conflicting numbers/settings are in `reports/ordered-prefix/SOURCES.md`.

## Original canonical notebook hashes (SHA-256)

| Notebook prefix | Original hash before output/path preparation |
| --- | --- |
| ch04 | `9545705e929b42182937b5037d40269d8f7dfb78f3625301728e2164af058be1` |
| ch05 | `7d27d0c92d3318b1f0bf506843188b2d138a9bb644fd440caa8489bb7dad86ac` |
| ch06 | `542431d453cbef16891e93ff3bfe029140080814309272f50598440ab852c8f6` |
| ch07 | `5703c5334a643d06410ccd1345a5fc97dc8620f082ca3f3f6cdc589bebed4993` |
| ch08 | `0fb419cae2e0aacabacc3e4c08864fb57cafb125e0d7e187bb4fbc9cb1a36f7c` |
| ch09 | `c30bb960852ed3e0cebd304142cc53d9b04fe5b525574e4f8e83f253bcbfd347` |

The hashes identify executed originals stored under `original_notebooks/2026-10-07` on NAS. The stale local ch09 copy is also retained with an explicit stale name. Prepared Git notebooks have different hashes because outputs were cleared and path setup added.

## Portability edits, not scientific reanalysis

- Added configurable model/checkpoint/dataset/output paths and noise-level/sample-limit environment variables to grid robustness/reference exporters. Original defaults and formulas remain; the documented RQ2 mid noise is explicitly 0.2 while H1 uses 0.25.
- Notebook preparation maps old `/exp` and Chapter 8 `/mnt/jupyter` defaults to configurable NAS roots, preserving original formulas/cell order.
- Main ordered-prefix loaders now pin the HF snapshots already found in the server cache, and write revision/batch metadata for new exports.
- The portable launcher mounts manifests correctly even in subdirectories and separates read-only source/inputs from writable outputs. Original launchers remain under `legacy/`.
- Added external source revision lock, safe NAS navigation index, notebook workspace/execution helpers and evaluator/export launchers.
- Thesis presentation helpers accept input/output roots without changing calculations.
- Source whitespace was tidied mechanically. Archived CSV line endings are explicitly preserved by Git because manifest/result provenance depends on their byte hashes.

No run is presented as bitwise reproducible solely because source commits are pinned. Historical transitive environments were not completely frozen, and stochastic interventions/GPU arithmetic remain relevant. See `VERIFICATION.md` for tests actually completed, rather than inferring validation from a README command.
