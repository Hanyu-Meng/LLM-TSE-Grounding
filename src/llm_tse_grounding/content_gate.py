"""Inference-only content-preserving gate for grounded TSE outputs.

The gate keeps the direct repaired evidence unless the candidate evidence has
low target-enrollment confidence and the grounded output passes inexpensive
content-drift checks.  It intentionally consumes no clean target waveform,
interferer reference, transcript reference, WER, or SI-SDR label.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal


OutputChoice = Literal["direct", "grounded"]


@dataclass(frozen=True)
class ContentGatePolicy:
    """Frozen thresholds for a content-preserving grounding gate.

    ``max_words_per_second``, ``max_bigram_repetition`` and
    ``min_dnsmos_p808`` are optional post-generation guards.  Set them to
    ``None`` for the token-only operating point.
    """

    enrollment_cosine_threshold: float = 0.375
    max_token_flip_rate: float = 0.20
    max_words_per_second: float | None = None
    max_bigram_repetition: float | None = None
    min_dnsmos_p808: float | None = None

    def __post_init__(self) -> None:
        finite = {
            "enrollment_cosine_threshold": self.enrollment_cosine_threshold,
            "max_token_flip_rate": self.max_token_flip_rate,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.max_token_flip_rate <= 1.0:
            raise ValueError("max_token_flip_rate must be in [0, 1]")
        if self.max_words_per_second is not None:
            if not math.isfinite(self.max_words_per_second) or self.max_words_per_second <= 0:
                raise ValueError("max_words_per_second must be finite and positive")
        if self.max_bigram_repetition is not None:
            if not 0.0 <= self.max_bigram_repetition <= 1.0:
                raise ValueError("max_bigram_repetition must be in [0, 1]")
        if self.min_dnsmos_p808 is not None:
            if not math.isfinite(self.min_dnsmos_p808):
                raise ValueError("min_dnsmos_p808 must be finite")


@dataclass(frozen=True)
class ContentGateDecision:
    choice: OutputChoice
    reasons: tuple[str, ...]
    words_per_second: float | None
    bigram_repetition: float | None


TOKEN_ONLY_POLICY = ContentGatePolicy()

# Exploratory DEV-only operating point.  It must be independently validated
# before it is used for a confirmatory TEST claim.
STRONG_GUARD_DEV_POLICY = ContentGatePolicy(
    enrollment_cosine_threshold=0.35,
    max_token_flip_rate=0.20,
    max_words_per_second=6.0,
    max_bigram_repetition=0.40,
    min_dnsmos_p808=2.50,
)


def normalized_words(text: str) -> tuple[str, ...]:
    """Return lowercase ASCII word tokens for output-only diagnostics."""

    return tuple(re.findall(r"[a-z]+", text.lower()))


def bigram_repetition_fraction(words: tuple[str, ...]) -> float:
    """Fraction of repeated word bigrams; zero for fewer than two words."""

    bigrams = tuple(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


def choose_content_preserving_output(
    *,
    evidence_enrollment_cosine: float,
    token_flip_rate: float,
    output_duration_seconds: float,
    grounded_transcript: str | None = None,
    grounded_dnsmos_p808: float | None = None,
    policy: ContentGatePolicy = TOKEN_ONLY_POLICY,
) -> ContentGateDecision:
    """Choose between the direct repaired evidence and grounded output.

    The grounded arm is selected only when speaker evidence is uncertain and
    every enabled content guard passes.  Missing optional measurements fail
    closed to the direct arm when their corresponding guard is enabled.
    """

    for name, value in {
        "evidence_enrollment_cosine": evidence_enrollment_cosine,
        "token_flip_rate": token_flip_rate,
        "output_duration_seconds": output_duration_seconds,
    }.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= token_flip_rate <= 1.0:
        raise ValueError("token_flip_rate must be in [0, 1]")
    if output_duration_seconds <= 0:
        raise ValueError("output_duration_seconds must be positive")

    reasons: list[str] = []
    if evidence_enrollment_cosine >= policy.enrollment_cosine_threshold:
        reasons.append("evidence-speaker-confidence-sufficient")
    if token_flip_rate > policy.max_token_flip_rate:
        reasons.append("grounded-token-drift-too-large")

    words_per_second: float | None = None
    repetition: float | None = None
    needs_transcript = (
        policy.max_words_per_second is not None
        or policy.max_bigram_repetition is not None
    )
    if needs_transcript:
        if grounded_transcript is None:
            reasons.append("grounded-transcript-missing")
        else:
            words = normalized_words(grounded_transcript)
            words_per_second = len(words) / output_duration_seconds
            repetition = bigram_repetition_fraction(words)
            if (
                policy.max_words_per_second is not None
                and words_per_second > policy.max_words_per_second
            ):
                reasons.append("grounded-word-rate-too-high")
            if (
                policy.max_bigram_repetition is not None
                and repetition > policy.max_bigram_repetition
            ):
                reasons.append("grounded-repetition-too-high")

    if policy.min_dnsmos_p808 is not None:
        if grounded_dnsmos_p808 is None:
            reasons.append("grounded-dnsmos-missing")
        elif not math.isfinite(grounded_dnsmos_p808):
            raise ValueError("grounded_dnsmos_p808 must be finite")
        elif grounded_dnsmos_p808 < policy.min_dnsmos_p808:
            reasons.append("grounded-quality-too-low")

    choice: OutputChoice = "direct" if reasons else "grounded"
    return ContentGateDecision(
        choice=choice,
        reasons=tuple(reasons),
        words_per_second=words_per_second,
        bigram_repetition=repetition,
    )
