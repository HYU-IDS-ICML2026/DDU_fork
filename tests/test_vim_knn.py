import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aggregate_vim_knn_results import aggregate_group, sample_mean_std
from evaluate_vim_knn import (
    ExactKNNScorer,
    ViMDetector,
    compute_auroc,
    compute_fpr95,
    l2_normalize,
    parse_checkpoint_name,
    resolve_seed,
)


class CheckpointParserTest(unittest.TestCase):
    def test_parse_sn_checkpoint(self):
        parsed = parse_checkpoint_name(
            "cifar10_sam_0.5wide_resnet_sn_3.0_mod_2_350.model"
        )
        self.assertEqual(parsed["optimizer"], "sam")
        self.assertEqual(parsed["hyperparameter"], "0.5")
        self.assertEqual(parsed["seed"], 2)
        self.assertEqual(parsed["epoch"], 350)
        self.assertTrue(parsed["sn"])
        self.assertTrue(parsed["mod"])

    def test_reject_non_sn_checkpoint(self):
        with self.assertRaises(ValueError):
            parse_checkpoint_name(
                "cifar10_sam_0.5wide_resnet_mod_2_350.model"
            )

    def test_seed_mismatch_fails(self):
        self.assertEqual(resolve_seed("auto", 1), 1)
        with self.assertRaises(ValueError):
            resolve_seed("0", 1)


class MetricTest(unittest.TestCase):
    def test_higher_is_id_auroc(self):
        self.assertEqual(compute_auroc([2.0, 3.0], [-2.0, -1.0]), 1.0)

    def test_submitted_fpr95_definition(self):
        id_scores = np.arange(100, dtype=np.float64)
        ood_scores = np.array([-1.0, 4.9, 5.0, 100.0])
        self.assertAlmostEqual(compute_fpr95(id_scores, ood_scores), 0.5)


class KNNTest(unittest.TestCase):
    def test_l2_normalization(self):
        features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        normalized = l2_normalize(features)
        torch.testing.assert_close(
            torch.linalg.vector_norm(normalized, dim=1), torch.ones(2)
        )

    def test_zero_feature_rejected(self):
        with self.assertRaises(ValueError):
            l2_normalize(torch.zeros(1, 3))

    def test_exact_kth_distance_and_scale_invariance(self):
        train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        query = torch.tensor([[1.0, 1.0], [2.0, -1.0]])
        scorer = ExactKNNScorer(k=2, chunk_size=1).fit(train)
        actual = scorer.score(query)

        normalized_train = l2_normalize(train)
        normalized_query = l2_normalize(query)
        distances = torch.cdist(normalized_query, normalized_train)
        expected = -torch.kthvalue(distances, 2, dim=1).values.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual, scorer.score(query * 7.0), rtol=1e-6, atol=1e-6
        )

    def test_k_boundary(self):
        with self.assertRaises(ValueError):
            ExactKNNScorer(k=3).fit(torch.eye(2))


class ViMTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.features = rng.normal(size=(30, 6))
        self.weight = rng.normal(size=(3, 6))
        self.bias = rng.normal(size=3)

    def test_reference_calculation(self):
        detector = ViMDetector([2, 3]).fit(
            self.features, self.weight, self.bias
        )
        u = -np.linalg.pinv(self.weight) @ self.bias
        shifted = self.features - u
        second_moment = shifted.T @ shifted / len(shifted)
        eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        logits = self.features @ self.weight.T + self.bias
        mean_max_logit = np.max(logits, axis=1).mean()

        np.testing.assert_allclose(detector.u, u, rtol=1e-12, atol=1e-12)
        for dim in (2, 3):
            null_space = eigenvectors[:, dim:]
            residual = np.linalg.norm(shifted @ null_space, axis=1)
            alpha = mean_max_logit / residual.mean()
            expected_energy = np.log(np.exp(logits).sum(axis=1))
            expected_vim = expected_energy - alpha * residual
            components = detector.components(self.features, dim)
            self.assertAlmostEqual(detector.alphas[dim], alpha, places=12)
            np.testing.assert_allclose(
                components["residual"], residual, rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                components["vim_score"], expected_vim, rtol=1e-10, atol=1e-10
            )

    def test_dimension_boundary(self):
        with self.assertRaises(ValueError):
            ViMDetector([6]).fit(self.features, self.weight, self.bias)

    def test_score_before_fit_fails(self):
        with self.assertRaises(RuntimeError):
            ViMDetector([2]).components(self.features, 2)


class AggregationTest(unittest.TestCase):
    def test_sample_standard_deviation(self):
        mean, std = sample_mean_std([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)

    def test_three_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir) / "group"
            values = [0.7, 0.8, 0.9]
            for seed, value in enumerate(values):
                seed_dir = group_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / "metadata.json").write_text(
                    json.dumps({"status": "complete"})
                )
                (seed_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "metrics": [
                                {
                                    "dataset": "svhn",
                                    "detector": "vim_D320",
                                    "auroc": value,
                                    "fpr95": 1.0 - value,
                                }
                            ]
                        }
                    )
                )
            aggregate = aggregate_group(group_dir, write=False)
            self.assertEqual(aggregate["status"], "complete")
            self.assertEqual(aggregate["ddof"], 1)
            row = aggregate["metrics"][0]
            self.assertAlmostEqual(row["mean_auroc"], 0.8)
            self.assertAlmostEqual(row["sample_std_auroc"], 0.1)

    def test_missing_seed_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            aggregate = aggregate_group(Path(temp_dir), write=False)
            self.assertEqual(aggregate["status"], "incomplete")
            self.assertEqual(aggregate["missing_seeds"], [0, 1, 2])
            self.assertEqual(aggregate["metrics"], [])


if __name__ == "__main__":
    unittest.main()
