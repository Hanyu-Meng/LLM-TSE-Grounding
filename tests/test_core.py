from __future__ import annotations

import unittest

import numpy as np

from llm_tse_grounding.candidate_selection import (
    CDCS5_CANDIDATES,
    select_by_enrollment_similarity,
)
from llm_tse_grounding.csg import csg_penalize_logits, select_csg_tokens
from llm_tse_grounding.difficulty import residual_difficulty, source_threshold_lambda
from llm_tse_grounding.fsq import VOCAB_SIZE, digits_to_ids, hamming_distance, ids_to_digits
from llm_tse_grounding.gnr import refine_gnr
from llm_tse_grounding.statistics import paired_bootstrap_difference, wilson_interval


class FSQTests(unittest.TestCase):
    def test_exhaustive_round_trip(self):
        ids = np.arange(VOCAB_SIZE, dtype=np.int64)
        np.testing.assert_array_equal(digits_to_ids(ids_to_digits(ids)), ids)

    def test_hamming_geometry(self):
        self.assertEqual(int(hamming_distance(0, 0)), 0)
        self.assertEqual(int(hamming_distance(0, VOCAB_SIZE - 1)), 8)
        self.assertEqual(int(hamming_distance(0, 1)), 1)


class CandidateSelectionTests(unittest.TestCase):
    def test_highest_similarity_wins(self):
        scores = {
            name: 0.1 + index / 10
            for index, name in enumerate(CDCS5_CANDIDATES)
        }
        self.assertEqual(select_by_enrollment_similarity(scores).candidate, "tfmap_context_full")

    def test_tie_uses_declared_order(self):
        scores = {name: 0.5 for name in CDCS5_CANDIDATES}
        self.assertEqual(select_by_enrollment_similarity(scores).candidate, "full")


class GroundingTests(unittest.TestCase):
    def test_lambda_zero_preserves_argmax(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(4, VOCAB_SIZE)).astype(np.float32)
        evidence = np.asarray([0, 1, 2, 3], dtype=np.int64)
        grounded = csg_penalize_logits(logits, evidence, 0.0)
        np.testing.assert_array_equal(grounded, logits)
        np.testing.assert_array_equal(select_csg_tokens(logits, evidence, 0.0), logits.argmax(1))

    def test_csg_can_prefer_evidence_neighbor(self):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        logits[0, VOCAB_SIZE - 1] = 4.0
        evidence = np.asarray([0], dtype=np.int64)
        self.assertEqual(int(select_csg_tokens(logits, evidence, 1.0)[0]), 0)

    def test_gnr_respects_radius_and_anchor_union(self):
        anchor = np.asarray([0, 0], dtype=np.int64)
        logits = np.zeros((2, VOCAB_SIZE), dtype=np.float32)
        logits[0, 1] = 5.0  # Hamming distance 1: accepted.
        logits[1, VOCAB_SIZE - 1] = 9.0  # Hamming distance 8: rejected.
        result = refine_gnr(logits, anchor, top_k=20, radius=2)
        np.testing.assert_array_equal(result.tokens, np.asarray([1, 0]))
        self.assertLessEqual(result.max_hamming_edit_distance, 2)
        self.assertFalse(result.refined_tokens_fed_back)


class DifficultyAndStatisticsTests(unittest.TestCase):
    def test_residual_feature_is_finite(self):
        evidence = np.asarray([1.0, -1.0, 0.5, -0.5])
        mixture = evidence + np.asarray([0.1, 0.1, -0.1, -0.1])
        result = residual_difficulty(mixture, evidence)
        self.assertTrue(np.isfinite(result["difficulty_value_db"]))
        self.assertGreaterEqual(source_threshold_lambda(result["difficulty_value_db"]), 0.0)

    def test_intervals(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        result = paired_bootstrap_difference(
            [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], resamples=200, seed=1
        )
        self.assertAlmostEqual(result["mean_difference"], -1.0)


if __name__ == "__main__":
    unittest.main()
