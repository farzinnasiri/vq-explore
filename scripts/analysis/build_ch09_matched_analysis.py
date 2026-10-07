"""Revise the existing Chapter 9 notebook without changing inference or raw exports."""

import argparse
import copy
import json
import textwrap
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    original = json.loads(args.source.read_text())
    cells = []

    def add(kind, source):
        cell = {
            "cell_type": kind,
            "id": f"ch09-matched-{len(cells):02d}",
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def reuse(index):
        cell = copy.deepcopy(original["cells"][index])
        cell["id"] = f"ch09-matched-{len(cells):02d}"
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", r"""
    # Chapter 9 - Reconstruction From Ordered Prefixes Under Distribution Shift

    This chapter compares FlexTok d12-d12 and One-D-Piece-S-256 on ImageNet and
    ImageNet-R using completed inference outputs. Each image was encoded once,
    then decoded at 8, 16, 32, 64, 96, 128, 160, 208, and 256 tokens.

    We ask three questions:

    1. What reconstruction quality is obtained at each budget, and how does it differ between datasets?
    2. How much does each additional block of tokens improve reconstruction?
    3. Are domain differences consistent across classes, and do the two systems behave differently?

    The primary comparison uses **the same 10,000 images per domain for both models**,
    with 50 images from each of 200 shared classes. ImageNet and ImageNet-R are
    class-matched, not paired photographs and renditions of the same scene.
    These measurements concern reconstruction under two complete decoding systems;
    they do not identify the information carried by individual tokens or prove that
    a learned ordering fails under shift. Equal token counts are not an equal-bit
    or equal-compute comparison.
    """)
    add("markdown", """
    ## 1. Inputs, Settings, and Reproducibility

    Raw CSVs and metadata remain unchanged in `chapter9_outputs/data/raw/`.
    Revised tables are written to `data/derived/matched_v2/` and figures to
    `figures/matched_v2/`; older analyses are retained outside those folders.
    The full One-D-Piece exports (50,000 ImageNet and 30,000 ImageNet-R images)
    are used only in the explicitly labelled supplementary section.

    The recorded FlexTok configuration is 25 denoising steps, guidance 15, and APG
    enabled, matching the d12-d12 column in
    [the paper's Appendix E.5, Table 8](https://arxiv.org/html/2502.13967v1).
    FlexTok decoding is stochastic; One-D-Piece decoding is deterministic.
    Shared preprocessing is RGB conversion, resize of the shorter edge to 256,
    and a 256-pixel center crop. LPIPS uses AlexNet on inputs scaled to [-1, 1].
    PSNR uses RGB mean squared error with data range 1; SSIM uses the recorded
    scikit-image channel-wise implementation with data range 1. These are our
    measured metrics, not a claim of reproducing every paper evaluation detail.
    """)
    reuse(3)
    add("code", """
    DERIVED_DIR = OUTPUT_DIR / "data" / "derived" / "matched_v2"
    FIGURE_DIR = OUTPUT_DIR / "figures" / "matched_v2"
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.max_open_warning"] = 40
    print("Revised tables:", DERIVED_DIR)
    print("Revised figures:", FIGURE_DIR)
    """)
    reuse(5)
    cells[-1]["source"] = "".join(cells[-1]["source"]).split("\ndef class_bootstrap")[0]
    reuse(7)
    add("code", """
    import hashlib

    provenance = []
    for key, filename in RUN_FILES.items():
        path = RAW_DIR / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        meta = metadata[key]
        provenance.append({
            "model": key[0], "dataset": key[1], "file": filename,
            "sha256": digest.hexdigest(), "checkpoint": meta["model"],
            "decoder_steps": meta.get("decoder_timesteps"),
            "guidance": meta.get("guidance_scale"),
            "apg": meta.get("perform_norm_guidance"),
        })
    save_table(pd.DataFrame(provenance), "ch09_input_provenance.csv")
    display(pd.DataFrame(provenance).drop(columns=["sha256", "file"]))
    """)
    add("markdown", """
    ## 2. Exact Image Matching

    We use the FlexTok selection as a fixed manifest. Each selected dataset/WNID/path
    must occur in both models at all nine budgets. Missing images, duplicate rows,
    unexpected class counts, or incomplete schedules stop the analysis rather than
    silently shrinking the sample. Paths are matched exactly, not by basename.
    """)
    add("code", """
    IMAGE_KEY = ["dataset_key", "wnid", "image"]
    IMAGE_MODEL_KEY = ["model_label"] + IMAGE_KEY
    selected = metrics_df.loc[metrics_df.model_label == "FlexTok", IMAGE_KEY].drop_duplicates()
    counts = selected.groupby(["dataset_key", "wnid"]).size()
    assert len(selected) == 20_000 and len(counts) == 400 and counts.eq(50).all()
    WNIDS = sorted(selected.loc[selected.dataset_key == "imagenet", "wnid"].unique())
    assert len(WNIDS) == 200
    assert WNIDS == sorted(selected.loc[selected.dataset_key == "imagenet_r", "wnid"].unique())

    matched_frames = []
    for model in MODEL_ORDER:
        available = metrics_df[metrics_df.model_label == model]
        coverage = selected.merge(available[IMAGE_KEY].drop_duplicates(), on=IMAGE_KEY,
                                  how="left", indicator=True, validate="one_to_one")
        assert coverage["_merge"].eq("both").all(), f"Missing selected images in {model}"
        matched = available.merge(selected, on=IMAGE_KEY, how="inner", validate="many_to_one")
        assert len(matched) == 20_000 * len(PREFIX_ORDER)
        assert not matched.duplicated(IMAGE_KEY + ["prefix_tokens"]).any()
        matched_frames.append(matched)
    analysis_df = pd.concat(matched_frames, ignore_index=True)
    scope_df = analysis_df.groupby(["model_label", "dataset_key"]).agg(
        images=("image", "nunique"), classes=("wnid", "nunique"), rows=("image", "size")
    ).reset_index()
    save_table(selected.sort_values(IMAGE_KEY), "ch09_matched_image_manifest.csv")
    save_table(scope_df, "ch09_comparison_scope.csv")
    display(scope_df)

    # One set of class draws is shared across every model, domain, metric, and budget.
    BOOT_INDEX = np.random.default_rng(SEED).integers(
        0, len(WNIDS), size=(BOOTSTRAP_REPLICATES, len(WNIDS)))

    def paired_intervals(matrix):
        matrix = np.asarray(matrix, dtype=float)
        assert matrix.shape[0] == len(WNIDS) and np.isfinite(matrix).all()
        flat = matrix.reshape(len(WNIDS), -1)
        draws = np.empty((BOOTSTRAP_REPLICATES, flat.shape[1]))
        for start in range(0, BOOTSTRAP_REPLICATES, 64):
            indices = BOOT_INDEX[start:start + 64]
            draws[start:start + len(indices)] = flat[indices].mean(axis=1)
        return flat.mean(axis=0), *np.quantile(draws, [0.025, 0.975], axis=0)
    """)
    add("markdown", """
    ## 3. Absolute Reconstruction Quality

    Each point averages images within a class and then gives each of the 200
    classes equal weight. All plots here use identical image sets across models.
    Lower LPIPS and higher PSNR/SSIM indicate better agreement with the original
    preprocessed image. Domain differences remain descriptive: content and the
    behavior of each metric can differ between photographs and renditions.
    """)
    reuse(12)
    reuse(13)
    add("code", """
    save_table(class_absolute_df, "ch09_class_absolute_quality.csv")
    display(absolute_summary[absolute_summary.prefix_tokens.isin([8, 32, 128, 256])]
            [["model_label", "dataset_key", "prefix_tokens", "lpips_mean", "psnr_mean", "ssim_mean"]])
    """)
    add("markdown", """
    ## 4. Gains From Additional Tokens

    For each image, the gain from budget a to b is LPIPS(a) minus LPIPS(b), or
    PSNR(b) minus PSNR(a) / SSIM(b) minus SSIM(a). Positive values always mean
    improvement. We report total gain and gain divided by b-a because the blocks
    have different lengths. These are finite differences of reconstruction scores,
    not estimates of information content per token or compression in bits.

    Curves are class-balanced. The accompanying table also reports the fraction
    of images that improve at each step: mean improvement is not a guarantee for
    every image. Small negative gains are retained, not clipped away.
    """)
    add("code", """
    ordered = analysis_df.sort_values(IMAGE_MODEL_KEY + ["prefix_tokens"]).copy()
    image_groups = ordered.groupby(IMAGE_MODEL_KEY, sort=False)
    ordered["from_tokens"] = image_groups["prefix_tokens"].shift()
    for metric in METRICS:
        change = image_groups[metric].diff()
        ordered[f"{metric}_gain"] = -change if metric == "lpips" else change
    gains = ordered.dropna(subset=["from_tokens"]).copy()
    gains["from_tokens"] = gains["from_tokens"].astype(int)
    gains["added_tokens"] = gains.prefix_tokens - gains.from_tokens
    GAIN_COLS = []
    for metric in METRICS:
        gains[f"{metric}_gain_per_token"] = gains[f"{metric}_gain"] / gains.added_tokens
        gains[f"{metric}_improved_fraction"] = (gains[f"{metric}_gain"] > 0).astype(float)
        GAIN_COLS += [f"{metric}_gain", f"{metric}_gain_per_token", f"{metric}_improved_fraction"]
    class_gains = gains.groupby(
        ["model_label", "dataset_key", "wnid", "from_tokens", "prefix_tokens", "added_tokens"],
        as_index=False)[GAIN_COLS].mean()
    gain_summary = class_gains.groupby(
        ["model_label", "dataset_key", "from_tokens", "prefix_tokens", "added_tokens"],
        as_index=False)[GAIN_COLS].mean()
    save_table(class_gains, "ch09_class_block_gains.csv")
    save_table(gain_summary, "ch09_block_gain_summary.csv")
    for suffix, label in [("gain", "total block gain"), ("gain_per_token", "gain per added token")]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        for r, model in enumerate(MODEL_ORDER):
            for col, metric in enumerate(METRICS):
                ax = axes[r, col]
                for dataset in DATASET_ORDER:
                    sub = gain_summary[(gain_summary.model_label == model) &
                                       (gain_summary.dataset_key == dataset)].sort_values("prefix_tokens")
                    ax.plot(range(len(sub)), sub[f"{metric}_{suffix}"], marker="o",
                            color=DATASET_COLORS[dataset], label=DATASET_LABELS[dataset])
                ax.axhline(0, color="black", linewidth=.7)
                ax.set_title(f"{model} | {metric.upper()}")
                ax.set_ylabel(label + (" (dB)" if metric == "psnr" and suffix == "gain" else ""))
                ax.set_xticks(range(8), [f"{a}-{b}" for a, b in zip(PREFIX_ORDER[:-1], PREFIX_ORDER[1:])], rotation=45)
                ax.set_xlabel("prefix block")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, f"ch09_fig_block_{suffix}.png")
        plt.show()
    display(gain_summary[["model_label", "dataset_key", "from_tokens", "prefix_tokens"] +
                         [f"{m}_improved_fraction" for m in METRICS]])
    """)
    add("markdown", r"""
    ## 5. Secondary View: Endpoint-Adjusted Truncation Cost

    We retain the earlier delta as a secondary analysis, not as a substitute for
    absolute quality. For LPIPS, the penalty is L(k)-L(256); for PSNR and SSIM it
    is M(256)-M(k). The class-level delta is the ImageNet-R penalty minus the
    ImageNet penalty. Negative individual penalties are possible.

    For LPIPS, the identity is:

    $$\Delta(k)=[L_R(k)-L_I(k)]-[L_R(256)-L_I(256)].$$

    A positive delta means a larger cost relative to the full-length endpoint.
    It does not necessarily mean worse absolute ImageNet-R reconstruction, more
    tokens required for a fixed absolute target, or less semantic information.
    Every delta is zero at 256 by construction. The intervals below resample
    classes jointly across all budgets and models. They are **pointwise**
    class-bootstrap intervals, not simultaneous bands or uncertainty over new
    shifts, new images within a class, or decoder seeds.
    """)
    reuse(17)
    add("code", """
    bootstrap_rows = []
    for model in MODEL_ORDER:
        for metric in METRICS:
            matrix = class_delta[class_delta.model_label == model].pivot(
                index="wnid", columns="prefix_tokens", values=f"{metric}_delta"
            ).reindex(index=WNIDS, columns=PREFIX_ORDER)
            mean, lower, upper = paired_intervals(matrix.to_numpy())
            for j, prefix in enumerate(PREFIX_ORDER):
                bootstrap_rows.append({"model": model, "metric": metric, "prefix_tokens": prefix,
                    "mean_ood_minus_imagenet": mean[j], "ci95_lower": lower[j], "ci95_upper": upper[j],
                    "n_classes": len(WNIDS), "positive_class_fraction": (matrix[prefix] > 0).mean()})
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    save_table(bootstrap_df, "ch09_bootstrap_intervals.csv")
    display(bootstrap_df[bootstrap_df.prefix_tokens.isin([8, 32, 128])])
    """)
    reuse(19)
    add("markdown", """
    ## 6. Direct Class-Paired Model Comparisons

    We compare FlexTok minus One-D-Piece within each shared class before
    bootstrapping. The first contrast is the difference in endpoint-adjusted
    domain deltas; positive means a larger adjusted domain gap for FlexTok.
    The second is the difference in reconstruction gain per added token, evaluated
    separately on each domain; positive means a larger score improvement for
    FlexTok over that block. Neither contrast establishes an intrinsic tokenizer
    ranking: checkpoints, decoders, and bits per token differ.

    The same class draws are used for every column. We do not infer model
    differences from overlap or non-overlap of separate model confidence intervals.
    """)
    add("code", """
    contrasts = []
    class_contrasts = []
    for metric in METRICS:
        wide = class_delta.pivot(index=["wnid", "prefix_tokens"], columns="model_label",
                                 values=f"{metric}_delta")
        diff = (wide["FlexTok"] - wide["One-D-Piece"]).unstack("prefix_tokens").reindex(
            index=WNIDS, columns=PREFIX_ORDER)
        mean, low, high = paired_intervals(diff.to_numpy())
        for j, p in enumerate(PREFIX_ORDER):
            contrasts.append({"kind": "endpoint_adjusted_domain_delta", "dataset": "R-minus-I",
                "metric": metric, "prefix_tokens": p, "from_tokens": np.nan,
                "flex_minus_one_d": mean[j], "ci95_lower": low[j], "ci95_upper": high[j]})
            class_contrasts.extend({"wnid": w, "metric": metric, "prefix_tokens": p,
                "flex_minus_one_d_delta": diff.loc[w, p]} for w in WNIDS)
        for dataset in DATASET_ORDER:
            wide = class_gains[class_gains.dataset_key == dataset].pivot(
                index=["wnid", "prefix_tokens"], columns="model_label", values=f"{metric}_gain_per_token")
            diff = (wide["FlexTok"] - wide["One-D-Piece"]).unstack("prefix_tokens").reindex(
                index=WNIDS, columns=PREFIX_ORDER[1:])
            mean, low, high = paired_intervals(diff.to_numpy())
            for j, p in enumerate(PREFIX_ORDER[1:]):
                contrasts.append({"kind": "block_gain_per_token", "dataset": dataset,
                    "metric": metric, "prefix_tokens": p, "from_tokens": PREFIX_ORDER[j],
                    "flex_minus_one_d": mean[j], "ci95_lower": low[j], "ci95_upper": high[j]})
    contrast_df = pd.DataFrame(contrasts)
    save_table(contrast_df, "ch09_paired_model_contrasts.csv")
    save_table(pd.DataFrame(class_contrasts), "ch09_class_model_delta_contrasts.csv")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for r, (kind, dataset) in enumerate([("endpoint_adjusted_domain_delta", "R-minus-I"),
                                       ("block_gain_per_token", "imagenet"),
                                       ("block_gain_per_token", "imagenet_r")]):
        for c, metric in enumerate(METRICS):
            sub = contrast_df[(contrast_df.kind == kind) & (contrast_df.dataset == dataset) &
                              (contrast_df.metric == metric)].sort_values("prefix_tokens")
            ax = axes[r, c]
            x = sub.prefix_tokens.to_numpy()
            ax.axhline(0, color="black", linestyle="--", linewidth=.8)
            ax.plot(x, sub.flex_minus_one_d, marker="o", color="#695A85")
            ax.fill_between(x, sub.ci95_lower.to_numpy(), sub.ci95_upper.to_numpy(), color="#695A85", alpha=.2)
            ax.set_title(f"{metric.upper()} | {dataset}")
            ax.set_ylabel("Flex minus One-D: " + ("adjusted gap" if r == 0 else "gain/token"))
            ax.set_xlabel("retained tokens" if r == 0 else "end of added block")
    fig.tight_layout()
    save_figure(fig, "ch09_fig_paired_model_contrasts.png")
    plt.show()
    """)
    add("markdown", """
    ## 7. Variation Across Classes and Budgets

    Each row in the heatmaps is a class; both models use the same fixed WNID order,
    without sorting classes to emphasize one result. The shared color scale shows
    endpoint-adjusted LPIPS deltas. The table includes mean, median, quartiles,
    and positive-class fraction for every metric and budget. It describes the
    distribution of the observed class means, not per-class statistical significance.
    """)
    add("code", """
    consistency_rows = []
    for (model, prefix), sub in class_delta.groupby(["model_label", "prefix_tokens"]):
        for metric in METRICS:
            v = sub[f"{metric}_delta"]
            consistency_rows.append({"model": model, "prefix_tokens": prefix, "metric": metric,
                "mean": v.mean(), "median": v.median(), "q25": v.quantile(.25), "q75": v.quantile(.75),
                "positive_classes": int((v > 0).sum()), "classes": len(v), "positive_fraction": (v > 0).mean()})
    consistency_df = pd.DataFrame(consistency_rows)
    save_table(consistency_df, "ch09_class_consistency_all_budgets.csv")
    vmax = class_delta.lpips_delta.abs().max()
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, layout="constrained")
    for ax, model in zip(axes, MODEL_ORDER):
        matrix = class_delta[class_delta.model_label == model].pivot(
            index="wnid", columns="prefix_tokens", values="lpips_delta").reindex(index=WNIDS, columns=PREFIX_ORDER)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(model)
        ax.set_xticks(range(len(PREFIX_ORDER)), PREFIX_ORDER, rotation=45)
        ax.set_yticks(range(0, 200, 20), [WNIDS[j] for j in range(0, 200, 20)])
        ax.set_xlabel("retained tokens")
    axes[0].set_ylabel("WNID (fixed order)")
    fig.colorbar(im, ax=axes, label="class-level endpoint-adjusted LPIPS delta")
    save_figure(fig, "ch09_fig_class_budget_heatmaps.png")
    plt.show()
    display(consistency_df[consistency_df.prefix_tokens.isin([8, 32, 128])])
    """)
    add("markdown", """
    ## 8. Smallest Tested Budget Reaching an Absolute Target

    For each image and metric, we record the first measured budget reaching a
    threshold. We also record the first budget after which **all remaining measured
    budgets** satisfy it, because individual curves can be nonmonotone. Neither
    quantity interpolates between observations or guarantees behavior at unmeasured
    budgets. An image that never reaches the threshold remains missing in the budget
    columns and is explicitly counted as not reached; it is never assigned 256.

    Thresholds are illustrative sensitivity choices, not validated definitions of
    acceptable quality: LPIPS <= 0.2, 0.3, 0.4; PSNR >= 16, 18, 20 dB; SSIM >= 0.4,
    0.5, 0.6. We do not choose the threshold giving the largest domain effect.
    Curves report the class-balanced fraction first reaching the target by a budget.
    Median budgets among successful images are conditional and should not be compared
    without also considering the failure fraction. These metrics do not establish
    semantic recognition quality.
    """)
    add("code", """
    THRESHOLDS = {"lpips": [.2, .3, .4], "psnr": [16., 18., 20.], "ssim": [.4, .5, .6]}
    budgets = np.asarray(PREFIX_ORDER)
    threshold_frames = []
    for metric in METRICS:
        wide = analysis_df.pivot(index=IMAGE_MODEL_KEY, columns="prefix_tokens", values=metric).reindex(columns=PREFIX_ORDER)
        values = wide.to_numpy()
        assert np.isfinite(values).all()
        for threshold in THRESHOLDS[metric]:
            hit = values <= threshold if metric == "lpips" else values >= threshold
            reached = hit.any(axis=1)
            sustained = np.logical_and.accumulate(hit[:, ::-1], axis=1)[:, ::-1]
            frame = wide.index.to_frame(index=False)
            frame["metric"] = metric
            frame["threshold"] = threshold
            frame["reached"] = reached
            frame["first_budget"] = np.where(reached, budgets[hit.argmax(axis=1)], np.nan)
            frame["sustained_budget"] = np.where(sustained.any(axis=1), budgets[sustained.argmax(axis=1)], np.nan)
            threshold_frames.append(frame)
    threshold_images = pd.concat(threshold_frames, ignore_index=True)
    save_table(threshold_images, "ch09_image_threshold_budgets.csv")
    threshold_rows = []
    reach_curves = []
    for (model, dataset, metric, threshold), sub in threshold_images.groupby(
            ["model_label", "dataset_key", "metric", "threshold"]):
        class_reach = sub.groupby("wnid").reached.mean().reindex(WNIDS)
        threshold_rows.append({"model": model, "dataset": dataset, "metric": metric, "threshold": threshold,
            "reach_fraction": class_reach.mean(), "never_reached_fraction": 1 - class_reach.mean(),
            "images_never_reached": int((~sub.reached).sum()),
            "median_first_budget_among_reached": sub.first_budget.median(),
            "sustained_reach_fraction": sub.sustained_budget.notna().mean()})
        for p in PREFIX_ORDER:
            by_class = sub.assign(first_by=sub.first_budget.le(p), sustained_by=sub.sustained_budget.le(p)).groupby("wnid")
            reach_curves.append({"model": model, "dataset": dataset, "metric": metric,
                "threshold": threshold, "prefix_tokens": p, "first_reached_fraction": by_class.first_by.mean().mean(),
                "sustained_reached_fraction": by_class.sustained_by.mean().mean()})
    threshold_summary = pd.DataFrame(threshold_rows)
    threshold_curve = pd.DataFrame(reach_curves)
    save_table(threshold_summary, "ch09_threshold_summary.csv")
    save_table(threshold_curve, "ch09_threshold_reach_curves.csv")
    for metric in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
        for r, model in enumerate(MODEL_ORDER):
            for col, threshold in enumerate(THRESHOLDS[metric]):
                ax = axes[r, col]
                for dataset in DATASET_ORDER:
                    sub = threshold_curve[(threshold_curve.model == model) & (threshold_curve.dataset == dataset) &
                        (threshold_curve.metric == metric) & (threshold_curve.threshold == threshold)].sort_values("prefix_tokens")
                    ax.plot(sub.prefix_tokens, sub.first_reached_fraction, marker="o", color=DATASET_COLORS[dataset], label=DATASET_LABELS[dataset])
                    ax.plot(sub.prefix_tokens, sub.sustained_reached_fraction, linestyle="--", color=DATASET_COLORS[dataset], alpha=.7)
                ax.set_title(f"{model} | {metric.upper()} {'<=' if metric == 'lpips' else '>='} {threshold:g}")
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("retained tokens")
                ax.set_ylabel("fraction reaching target")
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Solid: first reached; dashed: maintained at all later measured budgets", y=1.02)
        fig.tight_layout()
        save_figure(fig, f"ch09_fig_thresholds_{metric}.png")
        plt.show()
    display(threshold_summary)
    """)
    add("markdown", """
    ## 9. Supplementary Scope and Sampling Check

    The primary results above never mix image sets across models. Here we retain
    the larger One-D-Piece exports for two separate purposes: documenting absolute
    performance on all 1,000 ImageNet classes, and checking the influence of the
    selected ImageNet-R images on the 200-class analysis. The 1,000-class average
    is not a class-matched control for ImageNet-R. The plot compares absolute
    One-D-Piece curves on the fixed selected set with all available images from
    the same 200 classes; the full 1,000-class summary is exported separately.
    """)
    add("code", """
    all_class = metrics_df.groupby(["model_label", "dataset_key", "wnid", "prefix_tokens"], as_index=False)[METRICS].mean()
    full_inventory_quality = all_class.groupby(["model_label", "dataset_key", "prefix_tokens"], as_index=False)[METRICS].mean()
    full_inventory_quality["scope"] = "all exported classes; not a matched domain comparison"
    save_table(full_inventory_quality, "ch09_supplement_all_exported_absolute.csv")
    large_class = all_class[(all_class.model_label == "One-D-Piece") & all_class.wnid.isin(WNIDS)]
    large_summary = large_class.groupby(["dataset_key", "prefix_tokens"], as_index=False)[METRICS].mean()
    save_table(large_summary, "ch09_supplement_one_d_all_images_shared_classes.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, METRICS):
        for dataset in DATASET_ORDER:
            selected_sub = absolute_summary[(absolute_summary.model_label == "One-D-Piece") &
                                            (absolute_summary.dataset_key == dataset)].sort_values("prefix_tokens")
            large_sub = large_summary[large_summary.dataset_key == dataset].sort_values("prefix_tokens")
            ax.plot(selected_sub.prefix_tokens, selected_sub[f"{metric}_mean"], color=DATASET_COLORS[dataset],
                    label=f"{DATASET_LABELS[dataset]} selected")
            ax.plot(large_sub.prefix_tokens, large_sub[metric], color=DATASET_COLORS[dataset], linestyle="--",
                    label=f"{DATASET_LABELS[dataset]} all shared-class images")
        ax.set_title(metric.upper())
        ax.set_xlabel("retained tokens")
        ax.set_ylabel(METRIC_LABELS[metric])
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_figure(fig, "ch09_fig_one_d_sampling_sensitivity.png")
    plt.show()
    display(full_inventory_quality[(full_inventory_quality.model_label == "One-D-Piece") &
        (full_inventory_quality.dataset_key == "imagenet") & full_inventory_quality.prefix_tokens.isin([8, 32, 128, 256])])
    """)
    add("markdown", """
    ## 10. Source Groups and Interpretation Limits

    FlexTok source groups have different classes, batch sizes, GPU architectures,
    and batch-dependent random draws. The following summary is descriptive; it is
    not a controlled hardware or seed-sensitivity experiment. The class-bootstrap
    intervals do not establish stability across new decoder realizations.

    Matching images across models removes one sampling difference, not decoder or
    training differences. Matching classes across domains does not remove differences
    in image content or metric calibration. This chapter therefore describes the
    recorded systems and this shift. Causal explanations, semantic retention, and
    generalization to other shifts would need different evidence.
    """)
    reuse(29)
    add("markdown", """
    ## 11. Internal Checks and Results Notes

    These checks verify that the revised tables use the intended sample, block gains
    telescope to the endpoint difference, and adjusted deltas agree with absolute
    domain-gap subtraction. Notes are generated from the revised tables, without
    Supported/Not-resolved labels or claims of an ordering mechanism.
    """)
    add("code", """
    assert scope_df.images.eq(10_000).all() and scope_df.classes.eq(200).all()
    assert len(analysis_df) == 360_000
    assert np.allclose(class_delta.loc[class_delta.prefix_tokens == 256,
                                      [f"{m}_delta" for m in METRICS]], 0)
    for metric in METRICS:
        wide = analysis_df.pivot(index=IMAGE_MODEL_KEY, columns="prefix_tokens", values=metric).reindex(columns=PREFIX_ORDER)
        total = gains.groupby(IMAGE_MODEL_KEY)[f"{metric}_gain"].sum().reindex(wide.index)
        expected = wide[8] - wide[256] if metric == "lpips" else wide[256] - wide[8]
        assert np.allclose(total, expected), metric
        for model in MODEL_ORDER:
            curves = class_absolute_df[class_absolute_df.model_label == model].pivot(
                index=["wnid", "prefix_tokens"], columns="dataset_key", values=metric)
            gap = (curves.imagenet_r - curves.imagenet).unstack("prefix_tokens").reindex(index=WNIDS, columns=PREFIX_ORDER)
            expected_delta = gap.subtract(gap[256], axis=0)
            if metric != "lpips":
                expected_delta = -expected_delta
            actual = class_delta[class_delta.model_label == model].pivot(index="wnid", columns="prefix_tokens",
                values=f"{metric}_delta").reindex(index=WNIDS, columns=PREFIX_ORDER)
            assert np.allclose(expected_delta, actual), (model, metric)
    assert threshold_images.loc[~threshold_images.reached, "first_budget"].isna().all()
    assert threshold_images.first_budget.dropna().isin(PREFIX_ORDER).all()
    assert (threshold_curve.sustained_reached_fraction <= threshold_curve.first_reached_fraction + 1e-12).all()
    print("Passed: sample coverage, unique rows, endpoint identity, telescoping gains, and threshold checks.")

    from IPython.display import Markdown
    notes = ["# Chapter 9: results on identical image sets", "",
             "Both models use 10,000 images per domain, 50 per class across the same 200 WNIDs.", ""]
    evidence = []
    for model in MODEL_ORDER:
        notes.append(f"## {model}")
        for metric in METRICS:
            sub = bootstrap_df[(bootstrap_df.model == model) & (bootstrap_df.metric == metric) &
                               (bootstrap_df.prefix_tokens == 32)].iloc[0]
            absolute = absolute_summary[(absolute_summary.model_label == model) &
                (absolute_summary.prefix_tokens == 32)].set_index("dataset_key")
            sentence = (f"{metric.upper()} at 32 tokens: ImageNet {absolute.loc['imagenet', metric + '_mean']:.4f}; "
                f"ImageNet-R {absolute.loc['imagenet_r', metric + '_mean']:.4f}. "
                f"Endpoint-adjusted domain delta {sub.mean_ood_minus_imagenet:+.4f} "
                f"(pointwise 95% class-bootstrap interval {sub.ci95_lower:+.4f} to {sub.ci95_upper:+.4f}); "
                f"{round(sub.positive_class_fraction * 200)}/200 classes have positive observed deltas.")
            notes.append("- " + sentence)
            evidence.append({"model": model, "metric": metric, "budget": 32, "evidence": sentence})
        notes.append("")
        for dataset in DATASET_ORDER:
            sub = gain_summary[(gain_summary.model_label == model) &
                               (gain_summary.dataset_key == dataset)].sort_values("prefix_tokens")
            peak = sub.loc[sub.lpips_gain_per_token.idxmax()]
            late = sub[sub.from_tokens >= 128].lpips_gain.sum()
            target = threshold_curve[(threshold_curve.model == model) &
                (threshold_curve.dataset == dataset) & (threshold_curve.metric == "lpips") &
                (threshold_curve.threshold == .3) & (threshold_curve.prefix_tokens == 128)].iloc[0]
            failure = threshold_summary[(threshold_summary.model == model) &
                (threshold_summary.dataset == dataset) & (threshold_summary.metric == "lpips") &
                (threshold_summary.threshold == .3)].iloc[0]
            notes.append(f"- {DATASET_LABELS[dataset]}: the largest observed mean LPIPS gain per added token "
                f"is in {int(peak.from_tokens)}-{int(peak.prefix_tokens)} "
                f"({peak.lpips_gain_per_token:.5f}). Adding tokens from 128 to 256 reduces mean LPIPS "
                f"by {late:.4f}. For the illustrative LPIPS <= 0.3 target, "
                f"{target.first_reached_fraction:.1%} first reach it by 128 tokens; "
                f"{failure.never_reached_fraction:.1%} never reach it at any tested budget.")
        notes.append("")
    notes.extend(["## Effect of matching the image sets", ""])
    selected_r32 = absolute_summary[(absolute_summary.model_label == "One-D-Piece") &
        (absolute_summary.dataset_key == "imagenet_r") & (absolute_summary.prefix_tokens == 32)].iloc[0]
    all_r32 = large_summary[(large_summary.dataset_key == "imagenet_r") &
                           (large_summary.prefix_tokens == 32)].iloc[0]
    notes.append(f"One-D-Piece's class-balanced ImageNet-R LPIPS at 32 tokens is "
        f"{selected_r32.lpips_mean:.4f} on the fixed selected set, versus {all_r32.lpips:.4f} "
        "using all available images from those classes. The primary model comparison uses the selected "
        "set; conclusions from the previous unequal-sample analysis should not be mixed with these values.")
    notes.append("")
    notes.extend(["## What this answers", "",
        "Absolute curves, block gains, and threshold reach rates answer different reconstruction questions. "
        "They should be read together: a positive adjusted delta need not imply worse absolute reconstruction.", "",
        "The class-paired contrasts describe budget-dependent differences between the complete systems. "
        "They do not isolate encoder information or demonstrate failure of token ordering. "
        "Intervals describe class resampling of these outputs, not repeated seeds or new shifts."])
    save_table(pd.DataFrame(evidence), "ch09_results_evidence.csv")
    (DERIVED_DIR / "ch09_chapter_notes.md").write_text("\\n".join(notes) + "\\n")
    display(Markdown("\\n".join(notes)))
    """)

    original["cells"] = cells
    original["metadata"]["ch09_revision"] = "matched_v2_2026-09-06"
    args.destination.write_text(json.dumps(original, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cells)} cells to {args.destination}")


if __name__ == "__main__":
    main()
