"""Evaluate ViM and deep kNN OOD scores for one CIFAR-10 WRN checkpoint."""

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.mnist_ood as mnist_ood
import data.ood_detection.svhn as svhn
import data.ood_detection.tiny_imagenet as tiny_imagenet
from net.wide_resnet import wrn
import net.spectral_normalization.spectral_norm_conv_inplace as sn_lib


FEATURE_DIM = 640
NUM_CLASSES = 10
EXPECTED_COEFF = 3.0
EXPECTED_EPOCH = 350
DEFAULT_DATA_ROOT = "/home/ghjin/ICML/DDU_fork/data"
DEFAULT_OOD_DATASETS = ("cifar100", "tiny_imagenet", "svhn", "mnist")
CHECKPOINT_RE = re.compile(
    r"^cifar10_(?P<optimizer>sam|sgd)_"
    r"(?P<hyperparameter>\d+(?:\.\d+)?)wide_resnet_"
    r"sn_(?P<coeff>\d+(?:\.\d+)?)_mod_"
    r"(?P<seed>[012])_(?P<epoch>\d+)\.model$"
)


_original_load_hook = sn_lib.SpectralNormConvLoadStateDictPreHook.__call__


def _patched_load_hook(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    fn = self.fn
    version = local_metadata.get("spectral_norm_conv", {}).get(
        fn.name + ".version", None
    )
    if (version is None or version < 1) and (prefix + fn.name) not in state_dict:
        if (prefix + fn.name + "_orig") in state_dict:
            return
    return _original_load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )


sn_lib.SpectralNormConvLoadStateDictPreHook.__call__ = _patched_load_hook


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)


def parse_checkpoint_name(checkpoint_path):
    name = Path(checkpoint_path).name
    match = CHECKPOINT_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "Checkpoint name must be an SN=on CIFAR-10 WRN checkpoint with an "
            "unambiguous seed: " + name
        )

    parsed = match.groupdict()
    parsed["seed"] = int(parsed["seed"])
    parsed["epoch"] = int(parsed["epoch"])
    parsed["coeff"] = float(parsed["coeff"])
    parsed["sn"] = True
    parsed["mod"] = True
    parsed["checkpoint_name"] = name
    parsed["experiment_group"] = (
        f"cifar10_{parsed['optimizer']}_{parsed['hyperparameter']}_"
        f"wide_resnet_sn_{match.group('coeff')}_mod_epoch_{parsed['epoch']}"
    )

    if parsed["coeff"] != EXPECTED_COEFF:
        raise ValueError(
            f"Expected SN coeff {EXPECTED_COEFF}, got {parsed['coeff']} in {name}"
        )
    if parsed["epoch"] != EXPECTED_EPOCH:
        raise ValueError(
            f"Expected epoch {EXPECTED_EPOCH}, got {parsed['epoch']} in {name}"
        )
    return parsed


def resolve_seed(seed_arg, parsed_seed):
    if seed_arg == "auto":
        return parsed_seed
    try:
        seed = int(seed_arg)
    except ValueError as error:
        raise ValueError("--seed must be 'auto' or one of 0, 1, 2") from error
    if seed not in (0, 1, 2):
        raise ValueError("--seed must be one of 0, 1, 2")
    if seed != parsed_seed:
        raise ValueError(
            f"Explicit seed {seed} does not match checkpoint seed {parsed_seed}"
        )
    return seed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_auroc(id_scores, ood_scores):
    id_scores = _validated_scores(id_scores, "ID")
    ood_scores = _validated_scores(ood_scores, "OOD")
    labels = np.concatenate(
        [np.ones(id_scores.size, dtype=np.int8), np.zeros(ood_scores.size, dtype=np.int8)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def compute_fpr95(id_scores, ood_scores):
    """Match the submitted evaluator: ID fifth percentile, higher score is ID."""
    id_scores = _validated_scores(id_scores, "ID")
    ood_scores = _validated_scores(ood_scores, "OOD")
    threshold = float(np.percentile(id_scores, 5, method="linear"))
    return float(np.mean(ood_scores >= threshold))


def compute_metrics(id_scores, ood_scores):
    return {
        "auroc": compute_auroc(id_scores, ood_scores),
        "fpr95": compute_fpr95(id_scores, ood_scores),
    }


def _validated_scores(scores, name):
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} scores are empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} scores contain NaN or Inf")
    return array


def l2_normalize(features, eps=1e-10):
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    features = features.float()
    if features.ndim != 2:
        raise ValueError("Features must be a two-dimensional tensor")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain NaN or Inf")
    norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)
    if torch.any(norms <= eps):
        raise ValueError("Cannot L2-normalize a zero feature vector")
    return features / norms


class ExactKNNScorer:
    """Sun et al. deep kNN: L2-normalized penultimate features and kth L2 distance."""

    def __init__(self, k=50, chunk_size=128, device="cpu"):
        if k < 1:
            raise ValueError("k must be at least 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.k = int(k)
        self.chunk_size = int(chunk_size)
        self.device = torch.device(device)
        self.train_features = None

    def fit(self, train_features):
        normalized = l2_normalize(train_features)
        if self.k > normalized.shape[0]:
            raise ValueError(
                f"k={self.k} exceeds fitting sample count {normalized.shape[0]}"
            )
        self.train_features = normalized.to(self.device)
        return self

    def score(self, features):
        if self.train_features is None:
            raise RuntimeError("ExactKNNScorer.fit must be called before score")
        normalized = l2_normalize(features)
        scores = []
        with torch.no_grad():
            for start in range(0, normalized.shape[0], self.chunk_size):
                query = normalized[start : start + self.chunk_size].to(self.device)
                distances = torch.cdist(
                    query,
                    self.train_features,
                    p=2,
                    compute_mode="use_mm_for_euclid_dist",
                )
                kth_distance = torch.kthvalue(distances, self.k, dim=1).values
                scores.append((-kth_distance).cpu())
        return torch.cat(scores).numpy()


class ViMDetector:
    """Wang et al. ViM using raw penultimate features and raw classifier logits."""

    def __init__(self, dims):
        dims = tuple(dict.fromkeys(int(dim) for dim in dims))
        if not dims:
            raise ValueError("At least one ViM dimension is required")
        self.dims = dims
        self.feature_dim = None
        self.weight = None
        self.bias = None
        self.u = None
        self.null_spaces = {}
        self.alphas = {}
        self.fit_diagnostics = {}
        self.numerical_rank = None

    def fit(self, train_features, classifier_weight, classifier_bias):
        features = self._as_float64_features(train_features)
        weight = np.asarray(classifier_weight, dtype=np.float64)
        bias = np.asarray(classifier_bias, dtype=np.float64).reshape(-1)
        if weight.ndim != 2 or weight.shape[1] != features.shape[1]:
            raise ValueError("Classifier weight and feature dimensions do not match")
        if bias.shape != (weight.shape[0],):
            raise ValueError("Classifier bias shape does not match classifier weight")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError("Classifier parameters contain NaN or Inf")

        self.feature_dim = features.shape[1]
        for dim in self.dims:
            if not 0 < dim < self.feature_dim:
                raise ValueError(
                    f"ViM D must satisfy 0 < D < {self.feature_dim}; got {dim}"
                )

        self.weight = weight
        self.bias = bias
        self.u = -np.linalg.pinv(weight) @ bias
        shifted = features - self.u
        second_moment = shifted.T @ shifted / shifted.shape[0]
        second_moment = (second_moment + second_moment.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.numerical_rank = int(np.linalg.matrix_rank(second_moment))

        train_logits = features @ weight.T + bias
        mean_max_logit = float(np.max(train_logits, axis=1).mean())
        if not np.isfinite(mean_max_logit):
            raise ValueError("Mean maximum training logit is not finite")

        for dim in self.dims:
            null_space = np.ascontiguousarray(eigenvectors[:, dim:])
            residual = np.linalg.norm(shifted @ null_space, axis=1)
            mean_residual = float(residual.mean())
            if not np.isfinite(mean_residual) or mean_residual <= 0:
                raise ValueError(f"Invalid mean residual for ViM D={dim}")
            alpha = mean_max_logit / mean_residual
            if not np.isfinite(alpha):
                raise ValueError(f"Invalid alpha for ViM D={dim}")
            self.null_spaces[dim] = null_space
            self.alphas[dim] = float(alpha)
            self.fit_diagnostics[dim] = {
                "mean_max_train_logit": mean_max_logit,
                "mean_train_residual": mean_residual,
                "principal_min_eigenvalue": float(eigenvalues[dim - 1]),
                "residual_max_eigenvalue": float(eigenvalues[dim]),
            }
        return self

    def components(self, features, dim):
        if dim not in self.null_spaces:
            raise RuntimeError(f"ViM D={dim} was not fitted")
        features = self._as_float64_features(features)
        if features.shape[1] != self.feature_dim:
            raise ValueError("Feature dimension changed after ViM fitting")
        logits = features @ self.weight.T + self.bias
        energy = logsumexp(logits, axis=1)
        residual = np.linalg.norm(
            (features - self.u) @ self.null_spaces[dim], axis=1
        )
        virtual_logit = self.alphas[dim] * residual
        vim_score = energy - virtual_logit
        for name, values in (
            ("energy", energy),
            ("residual", residual),
            ("virtual_logit", virtual_logit),
            ("vim_score", vim_score),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or Inf for ViM D={dim}")
        return {
            "energy": energy,
            "residual": residual,
            "virtual_logit": virtual_logit,
            "vim_score": vim_score,
        }

    @staticmethod
    def _as_float64_features(features):
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Features must be a non-empty two-dimensional array")
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or Inf")
        return features


def _load_state_dict(checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    if not any("weight_orig" in key for key in state_dict):
        raise ValueError("Checkpoint filename says SN=on but SN state keys are absent")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def build_model(checkpoint_path, device):
    model = wrn(
        spectral_normalization=True,
        mod=True,
        coeff=EXPECTED_COEFF,
        num_classes=NUM_CLASSES,
        temp=1.0,
    )
    model.load_state_dict(_load_state_dict(checkpoint_path, device), strict=True)
    model.to(device)
    model.eval()
    return model


def extract_features(model, loader, device, verify_logits=False):
    sample_count = len(loader.dataset)
    features = torch.empty((sample_count, FEATURE_DIM), dtype=torch.float32)
    labels = torch.empty(sample_count, dtype=torch.int64)
    max_logit_diff = 0.0
    start = 0
    with torch.no_grad():
        for data, target in tqdm(loader, leave=False, disable=None):
            data = data.to(device, non_blocking=True)
            output = model(data)
            batch_features = model.feature
            if batch_features.shape[1] != FEATURE_DIM:
                raise ValueError(
                    f"Expected feature dimension {FEATURE_DIM}, got {batch_features.shape[1]}"
                )
            if verify_logits:
                reconstructed = model.linear(batch_features)
                max_logit_diff = max(
                    max_logit_diff,
                    float(torch.max(torch.abs(output - reconstructed)).item()),
                )
            end = start + batch_features.shape[0]
            features[start:end].copy_(batch_features.float().cpu())
            labels[start:end].copy_(target.long().cpu())
            start = end
    if start != sample_count:
        raise RuntimeError(f"Extracted {start} features for dataset of size {sample_count}")
    return features, labels.numpy(), max_logit_diff


def build_loaders(data_root, batch_size, num_workers, pin_memory, seed):
    train_loader, _ = cifar10.get_train_valid_loader(
        batch_size=batch_size,
        augment=False,
        val_seed=seed,
        val_size=0.1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        root=data_root,
        download=False,
    )
    id_loader = cifar10.get_test_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        root=data_root,
        download=False,
    )
    return train_loader, id_loader


def build_ood_loader(name, data_root, batch_size, num_workers, pin_memory):
    common = {
        "batch_size": batch_size,
        "root": data_root,
        "download": False,
        "pin_memory": pin_memory,
    }
    if name == "cifar100":
        return cifar100.get_test_loader(num_workers=num_workers, **common)
    if name == "tiny_imagenet":
        return tiny_imagenet.get_test_loader(num_workers=num_workers, **common)
    if name == "svhn":
        return svhn.get_test_loader(num_workers=num_workers, **common)
    if name == "mnist":
        return mnist_ood.get_test_loader(**common)
    raise ValueError(f"Unknown OOD dataset: {name}")


def _summary_rows(dataset_name, id_vim, ood_vim, id_knn, ood_knn, dims, k):
    rows = []
    energy_added = False
    for dim in dims:
        detector_scores = {
            f"vim_D{dim}": (id_vim[dim]["vim_score"], ood_vim[dim]["vim_score"]),
            f"residual_D{dim}": (-id_vim[dim]["residual"], -ood_vim[dim]["residual"]),
            f"virtual_logit_D{dim}": (
                -id_vim[dim]["virtual_logit"],
                -ood_vim[dim]["virtual_logit"],
            ),
        }
        if not energy_added:
            detector_scores["energy"] = (
                id_vim[dim]["energy"],
                ood_vim[dim]["energy"],
            )
            energy_added = True
        for detector, (id_scores, ood_scores) in detector_scores.items():
            metrics = compute_metrics(id_scores, ood_scores)
            rows.append({"dataset": dataset_name, "detector": detector, **metrics})
    if (id_knn is None) != (ood_knn is None):
        raise ValueError("ID and OOD kNN scores must both be present or absent")
    if id_knn is not None:
        metrics = compute_metrics(id_knn, ood_knn)
        rows.append(
            {"dataset": dataset_name, "detector": f"knn_k{k}", **metrics}
        )
    return rows


def _save_vim_components(
    path, components, labels, dataset_name, checkpoint_path, seed, dim, alpha, feature_dim
):
    payload = {
        **components,
        "sample_index": np.arange(len(components["vim_score"]), dtype=np.int64),
        "D": np.asarray(dim, dtype=np.int64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "feature_dim": np.asarray(feature_dim, dtype=np.int64),
        "residual_dim": np.asarray(feature_dim - dim, dtype=np.int64),
        "checkpoint": np.asarray(str(checkpoint_path)),
        "seed": np.asarray(seed, dtype=np.int64),
        "dataset": np.asarray(dataset_name),
    }
    if labels is not None:
        payload["label"] = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(path, **payload)


def _save_knn_scores(
    path, scores, labels, dataset_name, checkpoint_path, seed, k
):
    payload = {
        "knn_score": np.asarray(scores),
        "sample_index": np.arange(len(scores), dtype=np.int64),
        "k": np.asarray(k, dtype=np.int64),
        "checkpoint": np.asarray(str(checkpoint_path)),
        "seed": np.asarray(seed, dtype=np.int64),
        "dataset": np.asarray(dataset_name),
    }
    if labels is not None:
        payload["label"] = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(path, **payload)


def _write_summary(output_dir, summary):
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, cls=NumpyEncoder)
    rows = summary["metrics"]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "detector", "auroc", "fpr95"]
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(args):
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    parsed = parse_checkpoint_name(checkpoint_path)
    seed = resolve_seed(args.seed, parsed["seed"])
    seed_everything(seed)

    cuda = bool(args.gpu and torch.cuda.is_available())
    if args.gpu and not cuda:
        raise RuntimeError("--gpu was requested but CUDA is unavailable")
    device = torch.device("cuda" if cuda else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(checkpoint_path, device)
    train_loader, id_loader = build_loaders(
        str(data_root), args.batch_size, args.num_workers, cuda, seed
    )
    train_features, train_labels, _ = extract_features(
        model, train_loader, device
    )
    id_features, id_labels, logit_diff = extract_features(
        model, id_loader, device, verify_logits=True
    )
    if logit_diff >= 1e-4:
        raise ValueError(f"Raw logit parity failed: max_abs_diff={logit_diff}")

    classifier_weight = model.linear.weight.detach().cpu().numpy()
    classifier_bias = model.linear.bias.detach().cpu().numpy()
    vim = ViMDetector(args.vim_dims).fit(
        train_features, classifier_weight, classifier_bias
    )
    knn = None
    if not args.vim_only:
        knn = ExactKNNScorer(
            k=args.knn_k,
            chunk_size=args.knn_chunk_size,
            device=device,
        ).fit(train_features)

    id_vim = {dim: vim.components(id_features, dim) for dim in vim.dims}
    id_knn = None if knn is None else knn.score(id_features)
    id_raw_dir = output_dir / "raw" / "id"
    id_raw_dir.mkdir(parents=True, exist_ok=True)
    for dim in vim.dims:
        _save_vim_components(
            id_raw_dir / f"vim_D{dim}_components.npz",
            id_vim[dim],
            id_labels,
            "cifar10",
            checkpoint_path,
            seed,
            dim,
            vim.alphas[dim],
            vim.feature_dim,
        )
    if id_knn is not None:
        _save_knn_scores(
            id_raw_dir / f"knn_k{args.knn_k}_scores.npz",
            id_knn,
            id_labels,
            "cifar10",
            checkpoint_path,
            seed,
            args.knn_k,
        )

    summary_rows = []
    for dataset_name in args.ood_datasets:
        loader = build_ood_loader(
            dataset_name,
            str(data_root),
            args.batch_size,
            args.num_workers,
            cuda,
        )
        ood_features, _, _ = extract_features(model, loader, device)
        ood_vim = {dim: vim.components(ood_features, dim) for dim in vim.dims}
        ood_knn = None if knn is None else knn.score(ood_features)
        ood_raw_dir = output_dir / "raw" / "ood" / dataset_name
        ood_raw_dir.mkdir(parents=True, exist_ok=True)
        for dim in vim.dims:
            _save_vim_components(
                ood_raw_dir / f"vim_D{dim}_components.npz",
                ood_vim[dim],
                None,
                dataset_name,
                checkpoint_path,
                seed,
                dim,
                vim.alphas[dim],
                vim.feature_dim,
            )
        if ood_knn is not None:
            _save_knn_scores(
                ood_raw_dir / f"knn_k{args.knn_k}_scores.npz",
                ood_knn,
                None,
                dataset_name,
                checkpoint_path,
                seed,
                args.knn_k,
            )
        summary_rows.extend(
            _summary_rows(
                dataset_name,
                id_vim,
                ood_vim,
                id_knn,
                ood_knn,
                vim.dims,
                args.knn_k,
            )
        )
        del ood_features, ood_vim, ood_knn

    metadata = {
        "status": "complete",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": parsed["checkpoint_name"],
        "experiment_group": parsed["experiment_group"],
        "optimizer": parsed["optimizer"],
        "hyperparameter": parsed["hyperparameter"],
        "sn": True,
        "coeff": EXPECTED_COEFF,
        "mod": True,
        "epoch": parsed["epoch"],
        "parsed_training_seed": parsed["seed"],
        "evaluation_seed": seed,
        "split_seed": seed,
        "fitting_sample_count": int(train_features.shape[0]),
        "feature_dim": FEATURE_DIM,
        "vim_dims": list(vim.dims),
        "vim_residual_dims": {str(dim): FEATURE_DIM - dim for dim in vim.dims},
        "vim_alphas": vim.alphas,
        "vim_covariance_numerical_rank": vim.numerical_rank,
        "vim_fit_diagnostics": vim.fit_diagnostics,
        "evaluation_mode": "vim_only" if args.vim_only else "vim_knn",
        "data_root": str(data_root),
        "ood_datasets": list(args.ood_datasets),
        "score_convention": "higher_is_id",
        "fpr95_definition": "ID fifth percentile; fraction of OOD scores >= threshold",
        "logit_parity_max_abs_diff": logit_diff,
        "device": str(device),
    }
    if not args.vim_only:
        metadata.update(
            {
                "knn_k": args.knn_k,
                "knn_feature_normalization": "sample_l2",
                "knn_distance": "euclidean_kth_neighbor",
            }
        )
    summary = {
        "metadata": metadata,
        "metrics": summary_rows,
    }
    _write_summary(output_dir, summary)
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, cls=NumpyEncoder)
    return summary


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate ViM and L2-normalized deep kNN for one checkpoint",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint_path", required=True)
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
    parser.add_argument("--seed", default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--knn_chunk_size", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args(argv)


def main():
    args = get_args()
    summary = evaluate_checkpoint(args)
    print(json.dumps(summary["metadata"], indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
