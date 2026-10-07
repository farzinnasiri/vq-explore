import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PREFIXES = (8, 16, 32, 64, 128, 256)
TRUNCATED_PREFIXES = PREFIXES[:-1]
METRICS = ("lpips", "psnr", "ssim")
REFERENCE_BUDGETS = (32, 64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--id-dataset", default="imagenet")
    parser.add_argument("--ood-dataset", default="imagenet_r")
    return parser.parse_args()


def load_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["prefix_tokens"] = int(row["prefix_tokens"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def metric_loss(prefix_value, full_value, metric):
    return prefix_value - full_value if metric == "lpips" else full_value - prefix_value


def class_curves(rows):
    by_image = defaultdict(dict)
    for row in rows:
        by_image[(row["wnid"], row["image"])][row["prefix_tokens"]] = row

    by_class = defaultdict(lambda: defaultdict(list))
    for (wnid, _), prefix_rows in by_image.items():
        if set(prefix_rows) != set(PREFIXES):
            continue
        full = prefix_rows[256]
        for prefix in PREFIXES:
            for metric in METRICS:
                value = metric_loss(prefix_rows[prefix][metric], full[metric], metric)
                by_class[wnid][(prefix, metric)].append(value)

    return {
        wnid: {
            key: float(np.mean(values))
            for key, values in measurements.items()
        }
        for wnid, measurements in by_class.items()
    }


def class_raw_values(rows, prefix, metric):
    by_class = defaultdict(list)
    for row in rows:
        if row["prefix_tokens"] == prefix:
            by_class[row["wnid"]].append(row[metric])
    return {wnid: float(np.mean(values)) for wnid, values in by_class.items()}


def bootstrap_mean(values, samples, seed):
    values = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(values, size=(samples, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "n_classes": len(values),
    }


def paired_difference(left, right):
    shared = sorted(left.keys() & right.keys())
    return {key: right[key] - left[key] for key in shared}


def curve_area(curve, metric):
    losses = np.asarray([curve[(prefix, metric)] for prefix in PREFIXES], dtype=float)
    x = np.log2(np.asarray(PREFIXES, dtype=float))
    return float(np.trapz(losses, x) / (x[-1] - x[0]))


def required_budget(curve, metric, target_loss):
    losses = np.asarray([curve[(prefix, metric)] for prefix in PREFIXES], dtype=float)
    losses = np.minimum.accumulate(losses)
    if losses[0] <= target_loss:
        return float(PREFIXES[0])

    x = np.log2(np.asarray(PREFIXES, dtype=float))
    for index in range(1, len(PREFIXES)):
        if losses[index] <= target_loss:
            upper = losses[index - 1]
            lower = losses[index]
            fraction = 1.0 if upper == lower else (upper - target_loss) / (upper - lower)
            return float(2 ** (x[index - 1] + fraction * (x[index] - x[index - 1])))
    return float(PREFIXES[-1])


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    grouped = {}
    for path in args.inputs:
        metadata = json.loads((path.parent / "metadata.json").read_text())
        grouped[(metadata["model"], metadata["dataset"])] = load_rows(path)

    models = sorted({model for model, _ in grouped})
    datasets = (args.id_dataset, args.ood_dataset)
    expected = {(model, dataset) for model in models for dataset in datasets}
    missing = expected - grouped.keys()
    if missing:
        raise ValueError(f"Missing completed inputs: {sorted(missing)}")

    curves = {
        key: class_curves(rows)
        for key, rows in grouped.items()
    }
    result = {
        "primary_metric": "lpips",
        "primary_prefix": 32,
        "models": {},
        "between_models": {},
    }
    penalties = defaultdict(dict)
    class_rows = []
    marginal_rows = []
    area_rows = []
    inflation_rows = []

    for model_index, model in enumerate(models):
        result["models"][model] = {
            "full_reconstruction": {},
            "prefixes": {},
            "curve_area_ood_minus_id": {},
            "budget_inflation": {},
        }
        for dataset in datasets:
            rows = grouped[(model, dataset)]
            result["models"][model]["full_reconstruction"][dataset] = {
                metric: bootstrap_mean(
                    class_raw_values(rows, 256, metric).values(),
                    args.bootstrap_samples,
                    args.bootstrap_seed + model_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
            for wnid, curve in curves[(model, dataset)].items():
                for prefix in PREFIXES:
                    class_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "wnid": wnid,
                            "prefix_tokens": prefix,
                            **{
                                f"excess_{metric}_loss": curve[(prefix, metric)]
                                for metric in METRICS
                            },
                        }
                    )
                for previous, current in zip(PREFIXES, PREFIXES[1:]):
                    marginal_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "wnid": wnid,
                            "token_block": f"{previous + 1}-{current}",
                            "tokens_added": current - previous,
                            **{
                                f"{metric}_gain": curve[(previous, metric)] - curve[(current, metric)]
                                for metric in METRICS
                            },
                        }
                    )
                area_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "wnid": wnid,
                        **{
                            f"{metric}_log_budget_area": curve_area(curve, metric)
                            for metric in METRICS
                        },
                    }
                )

        for prefix in TRUNCATED_PREFIXES:
            prefix_result = {"domains": {}, "ood_minus_id_loss": {}}
            for metric_index, metric in enumerate(METRICS):
                losses = {
                    dataset: {
                        wnid: curve[(prefix, metric)]
                        for wnid, curve in curves[(model, dataset)].items()
                    }
                    for dataset in datasets
                }
                for dataset, values in losses.items():
                    prefix_result["domains"].setdefault(dataset, {})[
                        f"excess_{metric}_loss"
                    ] = bootstrap_mean(
                        values.values(),
                        args.bootstrap_samples,
                        args.bootstrap_seed + model_index * 1000 + prefix + metric_index,
                    )
                penalty = paired_difference(
                    losses[args.id_dataset], losses[args.ood_dataset]
                )
                prefix_result["ood_minus_id_loss"][metric] = bootstrap_mean(
                    penalty.values(),
                    args.bootstrap_samples,
                    args.bootstrap_seed + model_index * 1000 + prefix + metric_index + 10,
                )
                penalties[(model, prefix)][metric] = penalty
            result["models"][model]["prefixes"][str(prefix)] = prefix_result

        for metric_index, metric in enumerate(METRICS):
            id_areas = {
                wnid: curve_area(curve, metric)
                for wnid, curve in curves[(model, args.id_dataset)].items()
            }
            ood_areas = {
                wnid: curve_area(curve, metric)
                for wnid, curve in curves[(model, args.ood_dataset)].items()
            }
            result["models"][model]["curve_area_ood_minus_id"][metric] = bootstrap_mean(
                paired_difference(id_areas, ood_areas).values(),
                args.bootstrap_samples,
                args.bootstrap_seed + model_index * 10000 + metric_index,
            )

            for reference in REFERENCE_BUDGETS:
                inflation = {}
                shared = sorted(
                    curves[(model, args.id_dataset)].keys()
                    & curves[(model, args.ood_dataset)].keys()
                )
                for wnid in shared:
                    target = curves[(model, args.id_dataset)][wnid][(reference, metric)]
                    budget = required_budget(
                        curves[(model, args.ood_dataset)][wnid], metric, target
                    )
                    inflation[wnid] = budget - reference
                    inflation_rows.append(
                        {
                            "model": model,
                            "wnid": wnid,
                            "metric": metric,
                            "id_reference_tokens": reference,
                            "ood_required_tokens": budget,
                            "extra_tokens": budget - reference,
                            "budget_ratio": budget / reference,
                        }
                    )
                result["models"][model]["budget_inflation"].setdefault(str(reference), {})[
                    metric
                ] = bootstrap_mean(
                    inflation.values(),
                    args.bootstrap_samples,
                    args.bootstrap_seed + model_index * 10000 + reference + metric_index,
                )

    if len(models) == 2:
        first, second = models
        result["between_models"] = {
            "contrast": f"({second} OOD-ID loss) - ({first} OOD-ID loss)",
            "prefixes": {},
        }
        for prefix in TRUNCATED_PREFIXES:
            result["between_models"]["prefixes"][str(prefix)] = {
                metric: bootstrap_mean(
                    paired_difference(
                        penalties[(first, prefix)][metric],
                        penalties[(second, prefix)][metric],
                    ).values(),
                    args.bootstrap_samples,
                    args.bootstrap_seed + prefix + metric_index + 100,
                )
                for metric_index, metric in enumerate(METRICS)
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(
        args.output.parent / "class_prefix_losses.csv",
        ["model", "dataset", "wnid", "prefix_tokens"]
        + [f"excess_{metric}_loss" for metric in METRICS],
        class_rows,
    )
    write_csv(
        args.output.parent / "class_marginal_gains.csv",
        ["model", "dataset", "wnid", "token_block", "tokens_added"]
        + [f"{metric}_gain" for metric in METRICS],
        marginal_rows,
    )
    write_csv(
        args.output.parent / "class_curve_areas.csv",
        ["model", "dataset", "wnid"]
        + [f"{metric}_log_budget_area" for metric in METRICS],
        area_rows,
    )
    write_csv(
        args.output.parent / "class_budget_inflation.csv",
        [
            "model",
            "wnid",
            "metric",
            "id_reference_tokens",
            "ood_required_tokens",
            "extra_tokens",
            "budget_ratio",
        ],
        inflation_rows,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
