# Handover verification — 7 October 2026

## Completed

- Ten automated tests pass: source syntax, cleared notebook outputs, path-preparation preservation, exact 20,000-image manifest coverage and disjoint shard union, deterministic small-cohort selection, synthetic nine-budget summary execution, metric sign/budget endpoints, lightweight payload policy, offline image references and the RQ5 plotting-state regression.
- All five external source revisions were fetched successfully into a separate scratch directory. Both grid exporter launchers passed dry-run checks against those checkouts; model code was not added to the package.
- NAS catalog: all **26** dataset/checkpoint/result links resolve. Original data was not moved or replaced. Original executed notebooks and thesis input assets were copied into a dated snapshot.
- Canonical **RQ1 / Chapter 4** notebook executed top to bottom using the packaged notebook runner in an isolated writable workspace. Saved input directories were mounted read-only. It generated reconstruction/codebook tables, plots and the paired-clean-PSNR check.
- Canonical **ordered-prefix / ch09** notebook executed top to bottom with its exact saved input CSVs and qualitative assets. All **eleven** generated quantitative CSVs have identical SHA-256 hashes to the archived `focused_v3` originals. Exact-image matching, nine-budget coverage and qualitative original-image hash assertions passed. No new model inference was run.
- Canonical **RQ5 / Chapter 8** execution completed input/count validation, global code-usage summaries, sample-matched entropy calculations, and per-code spatial summaries, then exposed a plotting-state bug (three x positions reused for six datasets). That plotting cell was corrected to use local axis variables and covered by a regression test with deliberately stale three-position state. Its raw inputs and statistical calculations were not changed. Full post-fix top-to-bottom RQ5 execution is not claimed here.
- Helper `--help` checks pass for workspace preparation, notebook execution, export/evaluation launchers, manifest builders, decoder-locality analysis and the historical summarizer. Both main ordered-prefix runners also load and show help in the recorded FlexTok image without GPU access. The portable shell launcher passes `bash -n`.
- Supervisor HTML: original assembly passed desktop/mobile visual inspection and offline reload with no remote rendering requests. Its four scientific PNGs were unchanged. Local image sources were subsequently embedded mechanically for single-file sharing; decoded bytes match the adjacent original PNGs. Package documentation links were checked.
- The two existing ordered-prefix container runtimes were inspected without GPU access, and their installed package inventories were recorded. No container rebuild was claimed.

## Scope and limits

This is a **saved-data analysis and packaging check**, not a rerun of all thesis inference. RQ2/RQ3/RQ4 and exploratory notebooks retain their latest code but have not all been freshly executed during this handover. GPU inference, the TensorFlow feature evaluator, the original pilot/extension jobs and container rebuilds have not been rerun. Their input locations and run recipes are documented; historical dependency and stochastic limitations are explicit.

Verification artifacts are on the NAS at `vq-tokenizer-properties/verification/2026-10-07/`. They are not mixed into the published clean notebooks. Tests can be rerun with:

```bash
python -m unittest discover -s tests -v
```

Install the analysis dependencies for the numeric/plotting tests; missing dependencies are explicitly reported as skips. All ten tests passed without skips in the server's recorded Python 3.11 analysis image. This document records outcomes, not a guarantee of identical new GPU outputs on another hardware/runtime stack.
