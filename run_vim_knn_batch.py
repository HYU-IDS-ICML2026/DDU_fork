"""Run the SN-only ViM and deep kNN evaluator over all seed triplets."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from aggregate_vim_knn_results import aggregate_root
from evaluate_vim_knn import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OOD_DATASETS,
    parse_checkpoint_name,
)


DEFAULT_CHECKPOINT_ROOT = (
    "/home/ghjin/all_models_260328/models_260328/CIFAR-10/WideResNet"
)
DEFAULT_OUTPUT_ROOT = "/home/ghjin/DDU_fork_0724VIM_results/vim_knn"


def discover_checkpoints(checkpoint_root):
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(checkpoint_root)
    entries = []
    parse_errors = []
    for path in sorted(checkpoint_root.glob("*.model")):
        if "wide_resnet_sn_" not in path.name:
            continue
        try:
            parsed = parse_checkpoint_name(path)
        except ValueError as error:
            parse_errors.append({"checkpoint_path": str(path), "error": str(error)})
            continue
        entries.append({"checkpoint_path": str(path), **parsed})
    if parse_errors:
        raise ValueError(f"SN checkpoint parse errors: {parse_errors}")
    if not entries:
        raise ValueError(f"No eligible SN checkpoints under {checkpoint_root}")

    seen = set()
    duplicates = []
    for entry in entries:
        key = (entry["experiment_group"], entry["seed"])
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate experiment-group seeds: {duplicates}")
    return entries


def _is_complete(seed_dir):
    metadata_path = seed_dir / "metadata.json"
    summary_path = seed_dir / "summary.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        return False
    try:
        with metadata_path.open() as handle:
            return json.load(handle).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _append_failure(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def run_batch(args):
    entries = discover_checkpoints(args.checkpoint_root)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(Path(args.checkpoint_root).expanduser().resolve()),
        "output_root": str(output_root),
        "selection": "SN=on, coeff=3.0, mod=on, epoch=350",
        "evaluation_mode": "vim_only" if args.vim_only else "vim_knn",
        "vim_dims": args.vim_dims,
        "knn_k": None if args.vim_only else args.knn_k,
        "checkpoint_count": len(entries),
        "experiment_group_count": len({entry["experiment_group"] for entry in entries}),
        "entries": entries,
    }
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    if args.dry_run:
        print(
            f"Dry run: {manifest['checkpoint_count']} checkpoints, "
            f"{manifest['experiment_group_count']} groups"
        )
        return manifest

    failures_path = output_root / "failures.jsonl"
    evaluator = Path(__file__).with_name("evaluate_vim_knn.py")
    for index, entry in enumerate(entries, start=1):
        seed_dir = (
            output_root / entry["experiment_group"] / f"seed_{entry['seed']}"
        )
        if _is_complete(seed_dir):
            print(f"[{index}/{len(entries)}] skip complete: {entry['checkpoint_name']}")
            continue
        command = [
            sys.executable,
            str(evaluator),
            "--checkpoint_path",
            entry["checkpoint_path"],
            "--data_root",
            args.data_root,
            "--ood_datasets",
            *args.ood_datasets,
            "--vim_dims",
            *(str(dim) for dim in args.vim_dims),
            "--knn_k",
            str(args.knn_k),
            "--seed",
            "auto",
            "--output_dir",
            str(seed_dir),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--knn_chunk_size",
            str(args.knn_chunk_size),
        ]
        if args.gpu:
            command.append("--gpu")
        if args.vim_only:
            command.append("--vim_only")
        print(f"[{index}/{len(entries)}] evaluate: {entry['checkpoint_name']}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failure = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": entry["checkpoint_path"],
                "experiment_group": entry["experiment_group"],
                "seed": entry["seed"],
                "returncode": result.returncode,
            }
            _append_failure(failures_path, failure)
            if args.stop_on_error:
                raise RuntimeError(f"Evaluator failed: {failure}")

    aggregate_root(output_root)
    return manifest


def get_args(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--ood_datasets",
        nargs="+",
        default=list(DEFAULT_OOD_DATASETS),
        choices=list(DEFAULT_OOD_DATASETS),
    )
    parser.add_argument("--vim_dims", nargs="+", type=int, default=[256, 320])
    parser.add_argument("--vim_only", action="store_true")
    parser.add_argument("--knn_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--knn_chunk_size", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args(argv)


def main():
    run_batch(get_args())


if __name__ == "__main__":
    main()
