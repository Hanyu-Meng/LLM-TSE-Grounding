"""Core reference operators for Repair Before Grounding."""

from .candidate_selection import POOL_D, Selection, select_by_enrollment_similarity
from .csg import csg_penalize_logits, select_csg_tokens
from .content_gate import (
    STRONG_GUARD_DEV_POLICY,
    TOKEN_ONLY_POLICY,
    ContentGateDecision,
    ContentGatePolicy,
    choose_content_preserving_output,
)
from .difficulty import residual_difficulty, source_threshold_lambda
from .fsq import BASE, N_DIGITS, VOCAB_SIZE, digits_to_ids, hamming_distance, ids_to_digits
from .gnr import GNRResult, refine_gnr
from .statistics import paired_bootstrap_difference, wilson_interval

__all__ = [
    "BASE",
    "ContentGateDecision",
    "ContentGatePolicy",
    "GNRResult",
    "N_DIGITS",
    "POOL_D",
    "Selection",
    "STRONG_GUARD_DEV_POLICY",
    "TOKEN_ONLY_POLICY",
    "VOCAB_SIZE",
    "csg_penalize_logits",
    "choose_content_preserving_output",
    "digits_to_ids",
    "hamming_distance",
    "ids_to_digits",
    "paired_bootstrap_difference",
    "refine_gnr",
    "residual_difficulty",
    "select_by_enrollment_similarity",
    "select_csg_tokens",
    "source_threshold_lambda",
    "wilson_interval",
]
