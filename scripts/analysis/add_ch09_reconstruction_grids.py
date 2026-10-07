"""Snapshot saved run previews and add reproducible qualitative grids to Chapter 9.

Run on Elysium. This script copies existing PNGs; it never loads a model.
"""

import argparse
import csv
import hashlib
import json
import shutil
import textwrap
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output-notebook", type=Path, required=True)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    assert notebook["metadata"].get("ch09_revision") == "matched_v2_2026-09-06"
    assert not any(c.get("id", "").startswith("ch09-visual-") for c in notebook["cells"])
    raw = args.notebook.parent / "chapter9_outputs" / "data" / "raw"
    destination = raw / "saved_reconstructions_v1"
    destination.mkdir(parents=True, exist_ok=True)
    prefixes = [8, 16, 32, 64, 96, 128, 160, 208, 256]
    groups = {"part1": ("flextok_matched10k_part1", "matched_10k_part1.csv")}
    for gpu in range(4):
        groups[f"gpu{gpu}"] = (f"flextok_matched10k_part2_shards/gpu{gpu}", f"part2_shards/gpu{gpu}.csv")
    records = []
    meta_records = []

    def snapshot(model, group, dataset, row, ordinal, run):
        stem = f"{ordinal:03d}_{row['wnid']}"
        folder = destination / model / group / dataset
        folder.mkdir(parents=True, exist_ok=True)
        meta_path = run / "metadata.json"
        shutil.copy2(meta_path, folder / "metadata.json")
        meta_records.append({"model": model, "group": group, "dataset": dataset,
                             "source": str(meta_path), "sha256": sha256(meta_path)})
        for prefix in [0] + prefixes:
            filename = f"{stem}_original.png" if prefix == 0 else f"{stem}_{prefix:03d}_tokens.png"
            source = run / "previews" / filename
            assert source.is_file(), source
            target = folder / filename
            shutil.copy2(source, target)
            records.append({"model": model, "group": group, "dataset": dataset,
                "wnid": row["wnid"], "image": row["image"], "ordinal": ordinal,
                "prefix_tokens": prefix, "asset": str(target.relative_to(destination)),
                "source": str(source), "sha256": sha256(target)})

    for group, (run_path, manifest_path) in groups.items():
        manifest = args.experiment / "manifests" / manifest_path
        with manifest.open() as handle:
            rows = list(csv.DictReader(handle))
        for dataset in ["imagenet", "imagenet_r"]:
            domain_rows = [r for r in rows if r["dataset"] == dataset]
            run = args.experiment / "outputs" / run_path / dataset
            meta = json.loads((run / "metadata.json").read_text())
            assert meta["manifest_sha256"] == sha256(manifest), (group, dataset)
            ordinals = [0, 1, 2] if group == "part1" and dataset == "imagenet_r" else [0]
            for ordinal in ordinals:
                row = domain_rows[ordinal]
                snapshot("flextok", group, dataset, row, ordinal, run)
                if group == "part1" and dataset == "imagenet_r":
                    other = args.experiment / "outputs" / "one_d_piece" / dataset
                    stem = f"{ordinal:03d}_{row['wnid']}"
                    # Identical originals establish a same-input comparison, not just matching labels.
                    assert sha256(run / "previews" / f"{stem}_original.png") == sha256(
                        other / "previews" / f"{stem}_original.png")
                    snapshot("one_d_piece", "full_run", dataset, row, ordinal, other)
    with (destination / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (destination / "source_metadata.json").write_text(json.dumps(meta_records, indent=2) + "\n")

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"ch09-visual-{len(notebook['cells'])}",
                "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        notebook["cells"].append(cell)

    add("markdown", """
    ## 12. Saved Reconstructions: Same Inputs, Different Budgets

    The grids below use PNGs saved during the completed quantitative runs, not new
    model samples. The original is the actual 256 x 256 preprocessed model input.
    Columns show 8, 32, 64, 128, and 256 tokens; all nine measured budgets for these
    examples are available in the copied assets. LPIPS labels come from the original
    run CSVs (computed before PNG quantization), not a new evaluation of the PNGs.

    **Availability limits the comparison.** The scripts saved only the first five
    previews of each run or shard, not every reconstruction. Both models have
    identical saved inputs for the first ImageNet-R class, `n01443537`. We show the
    first three of those five inputs, selected by saved index rather than visual
    quality or reconstruction score. The input PNG hashes match exactly across
    models. This is a same-input model comparison, but only for one class.

    FlexTok also has saved examples from five classes in each domain. Separate grids
    show index zero from each completed source group, in WNID order. Those are
    convenience examples, not a random or representative class sample. The paired
    rows across domains share a class, not an underlying scene. We do not substitute
    unrelated One-D-Piece images into these rows. A multi-class, same-input comparison
    of both models would require saving additional One-D-Piece reconstructions.

    Assets and their source paths/hashes are preserved in
    `data/raw/saved_reconstructions_v1/`. Figures are exported as PNG and PDF in
    `figures/matched_v2/` for thesis use. These illustrations cannot establish an
    individual token's semantics or decoder stability across random seeds.
    """)
    add("code", """
    from PIL import Image

    RECON_DIR = RAW_DIR / "saved_reconstructions_v1"
    recon_manifest = pd.read_csv(RECON_DIR / "manifest.csv")
    RECON_MODELS = {"flextok": "FlexTok", "one_d_piece": "One-D-Piece"}
    SHOW_BUDGETS = [8, 32, 64, 128, 256]
    for asset in recon_manifest.itertuples():
        path = RECON_DIR / asset.asset
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset.sha256
        with Image.open(path) as image:
            assert image.size == (256, 256), path

    recon_examples = recon_manifest[recon_manifest.prefix_tokens == 0].copy()
    assert not recon_manifest.duplicated(["model", "dataset", "image", "prefix_tokens"]).any()
    metric_lookup = analysis_df.set_index(["model_label", "dataset_key", "wnid", "image", "prefix_tokens"])
    for example in recon_examples.itertuples():
        for budget in PREFIX_ORDER:
            assert (RECON_MODELS[example.model], example.dataset, example.wnid, example.image, budget) in metric_lookup.index
    save_table(recon_examples[["model", "group", "dataset", "wnid", "image", "ordinal", "sha256"]],
               "ch09_visual_example_selection.csv")
    display(recon_examples.groupby(["model", "dataset"]).agg(
        saved_examples_used=("image", "nunique"), classes=("wnid", "nunique")).reset_index())

    def reconstruction_grid(examples, title, filename):
        fig, axes = plt.subplots(len(examples), len(SHOW_BUDGETS) + 1,
                                 figsize=(14, 2.15 * len(examples)), squeeze=False)
        for r, example in enumerate(examples.to_dict("records")):
            assets = recon_manifest[(recon_manifest.model == example["model"]) &
                (recon_manifest.dataset == example["dataset"]) &
                (recon_manifest.image == example["image"])].set_index("prefix_tokens")
            for c, budget in enumerate([0] + SHOW_BUDGETS):
                ax = axes[r, c]
                with Image.open(RECON_DIR / assets.loc[budget, "asset"]) as img:
                    ax.imshow(np.asarray(img), interpolation="nearest")
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if r == 0:
                    ax.set_title("Original input" if budget == 0 else f"{budget} tokens", fontsize=12)
                if c == 0:
                    ax.set_ylabel(example["row_label"], fontsize=10, rotation=0, ha="right", va="center", labelpad=12)
                else:
                    value = metric_lookup.loc[(RECON_MODELS[example["model"]], example["dataset"],
                        example["wnid"], example["image"], budget), "lpips"]
                    ax.set_xlabel(f"LPIPS {value:.3f}", fontsize=9, labelpad=3)
        fig.suptitle(title, fontsize=14, y=1.01)
        fig.subplots_adjust(wspace=.06, hspace=.2, left=.13, right=.99, top=.96, bottom=.025)
        save_figure(fig, filename + ".png")
        save_figure(fig, filename + ".pdf")
        plt.show()
        return fig

    pair_rows = []
    for ordinal in [0, 1, 2]:
        pair = recon_examples[(recon_examples.dataset == "imagenet_r") &
            (recon_examples.wnid == "n01443537") & (recon_examples.ordinal == ordinal)]
        assert len(pair) == 2 and pair.sha256.nunique() == 1
        for model in ["flextok", "one_d_piece"]:
            row = pair[pair.model == model].iloc[0].to_dict()
            row["row_label"] = f"Example {ordinal + 1}\\n{RECON_MODELS[model]}"
            pair_rows.append(row)
    reconstruction_grid(pd.DataFrame(pair_rows),
        "Same ImageNet-R inputs: FlexTok and One-D-Piece (class n01443537)",
        "ch09_fig_reconstructions_matched_models")
    """)
    add("markdown", """
    ### Reading the Paired Grid

    Read across a row to follow one reconstruction as its prefix grows, then compare
    the two rows for the same input. LPIPS is lower when the reconstruction is closer
    to the original under that metric. Apparent realism and fidelity to the original
    are different properties: a plausible output may introduce details not present
    in the input. These are single saved outputs, especially relevant for FlexTok's
    stochastic decoder, rather than guarantees about every decoding realization.

    ### Available FlexTok Examples Across Classes

    Each figure below uses the same five WNIDs. Only the first saved input of each
    class/source group is shown. The export manifest supplies the exact input paths
    for captions and traceability; the WNID labels avoid guessing class names.
    """)
    add("code", """
    for dataset in DATASET_ORDER:
        examples = recon_examples[(recon_examples.model == "flextok") &
            (recon_examples.dataset == dataset) & (recon_examples.ordinal == 0)].sort_values("wnid").copy()
        assert len(examples) == 5 and examples.wnid.nunique() == 5
        examples["row_label"] = examples.wnid
        reconstruction_grid(examples,
            f"FlexTok: saved examples across five {DATASET_LABELS[dataset]} classes",
            f"ch09_fig_reconstructions_flextok_{dataset}")

    captions = [
        "# Reconstruction figure captions", "",
        "## Same-input comparison", "",
        "Saved reconstructions from FlexTok d12-d12 and One-D-Piece-S-256 for three "
        "identical ImageNet-R inputs in WNID n01443537. Columns show the preprocessed "
        "original and 8-, 32-, 64-, 128-, and 256-token prefixes. The first three saved "
        "examples are shown; selection was not based on quality. LPIPS values are "
        "taken from the original inference exports. This single-class illustration "
        "does not establish a general model ranking.", "",
        "## FlexTok across classes", "",
        "Saved FlexTok reconstructions for the first available preview in each of "
        "five source-group classes, displayed separately for ImageNet and ImageNet-R. "
        "Classes are matched across the figures, but the underlying images are not "
        "scene-paired. These convenience examples illustrate progressive reconstruction "
        "and are not a random sample of the evaluated classes.",
    ]
    (DERIVED_DIR / "ch09_reconstruction_captions.md").write_text("\\n".join(captions) + "\\n")
    print("Saved figure PNGs, PDFs, selection manifest, and captions. No new inference was used.")
    """)
    args.output_notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Copied {len(records)} existing PNGs and added four notebook cells.")


if __name__ == "__main__":
    main()
