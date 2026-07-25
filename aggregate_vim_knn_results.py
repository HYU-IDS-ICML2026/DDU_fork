"""Aggregate seed 0/1/2 ViM and kNN metrics with sample standard deviation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_SEEDS = (0, 1, 2)


def sample_mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two scalar values are required")
    if not np.isfinite(values).all():
        raise ValueError("Cannot aggregate NaN or Inf")
    return float(values.mean()), float(values.std(ddof=1))


def _load_seed_summary(seed_dir):
    summary_path = seed_dir / "summary.json"
    metadata_path = seed_dir / "metadata.json"
    if not summary_path.is_file() or not metadata_path.is_file():
        return None
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if metadata.get("status") != "complete":
        return None
    with summary_path.open() as handle:
        summary = json.load(handle)
    metrics = {}
    for row in summary.get("metrics", []):
        key = (row["dataset"], row["detector"])
        if key in metrics:
            raise ValueError(f"Duplicate metric row {key} in {summary_path}")
        metrics[key] = {
            "auroc": float(row["auroc"]),
            "fpr95": float(row["fpr95"]),
        }
    if not metrics:
        raise ValueError(f"No metrics in {summary_path}")
    return {"metadata": metadata, "metrics": metrics}


def aggregate_group(group_dir, required_seeds=REQUIRED_SEEDS, write=True):
    group_dir = Path(group_dir)
    seed_results = {}
    missing_seeds = []
    for seed in required_seeds:
        result = _load_seed_summary(group_dir / f"seed_{seed}")
        if result is None:
            missing_seeds.append(seed)
        else:
            seed_results[seed] = result

    aggregate = {
        "experiment_group": group_dir.name,
        "status": "complete" if not missing_seeds else "incomplete",
        "required_seeds": list(required_seeds),
        "n_success": len(seed_results),
        "missing_seeds": missing_seeds,
        "standard_deviation": "sample",
        "ddof": 1,
        "metrics": [],
    }

    if not missing_seeds:
        key_sets = [set(result["metrics"]) for result in seed_results.values()]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError(f"Seed metric schemas differ in {group_dir}")
        for dataset, detector in sorted(key_sets[0]):
            row = {
                "dataset": dataset,
                "detector": detector,
                "n_success": len(required_seeds),
                "missing_seeds": [],
            }
            for metric_name in ("auroc", "fpr95"):
                values = [
                    seed_results[seed]["metrics"][(dataset, detector)][metric_name]
                    for seed in required_seeds
                ]
                mean, std = sample_mean_std(values)
                for seed, value in zip(required_seeds, values):
                    row[f"seed{seed}_{metric_name}"] = value
                row[f"mean_{metric_name}"] = mean
                row[f"sample_std_{metric_name}"] = std
            aggregate["metrics"].append(row)

    if write:
        group_dir.mkdir(parents=True, exist_ok=True)
        with (group_dir / "aggregate.json").open("w") as handle:
            json.dump(aggregate, handle, indent=2)
        fieldnames = [
            "dataset",
            "detector",
            "seed0_auroc",
            "seed1_auroc",
            "seed2_auroc",
            "mean_auroc",
            "sample_std_auroc",
            "seed0_fpr95",
            "seed1_fpr95",
            "seed2_fpr95",
            "mean_fpr95",
            "sample_std_fpr95",
            "n_success",
            "missing_seeds",
        ]
        with (group_dir / "aggregate.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate["metrics"])
    return aggregate


def aggregate_root(output_root):
    output_root = Path(output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    aggregates = []
    for group_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if any((group_dir / f"seed_{seed}").exists() for seed in REQUIRED_SEEDS):
            aggregates.append(aggregate_group(group_dir))
    index = {
        "output_root": str(output_root),
        "standard_deviation": "sample",
        "ddof": 1,
        "groups": aggregates,
    }
    with (output_root / "aggregate_index.json").open("w") as handle:
        json.dump(index, handle, indent=2)
    return index


def get_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output_root",
        default="/home/ghjin/DDU_fork_0724VIM_results/vim_knn",
    )
    return parser.parse_args(argv)


def main():
    args = get_args()
    index = aggregate_root(args.output_root)
    complete = sum(group["status"] == "complete" for group in index["groups"])
    print(f"Aggregated {complete}/{len(index['groups'])} complete groups")


if __name__ == "__main__":
    main()
