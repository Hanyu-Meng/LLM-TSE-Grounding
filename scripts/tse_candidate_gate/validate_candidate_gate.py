#!/usr/bin/env python3
"""Fail-closed validation for the completed DEV-only candidate gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "analysis/candidate_gate"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    candidate_rows = read_jsonl(GATE / "per_trial_candidates.jsonl")
    acoustic_rows = read_jsonl(GATE / "all_acoustic_metrics.jsonl")
    speaker_rows = read_jsonl(GATE / "speaker_embeddings/candidate_speaker_metrics.jsonl")
    asr_rows = read_jsonl(GATE / "all_asr_metrics.jsonl")
    natural_rows = read_jsonl(ROOT / "analysis/wesep_speaker_selection/dev_all_trials.jsonl")
    frozen_swaps = read_jsonl(
        ROOT / "analysis/wesep_speaker_selection/dev_high_confidence_wrong.jsonl"
    )
    old_metrics = read_jsonl(ROOT / "dev_outputs/WeSep/per_trial_metrics.jsonl")
    results = json.loads((GATE / "oracle_candidate_results.json").read_text(encoding="utf-8"))

    trial_ids = [row["trial_id"] for row in candidate_rows]
    candidate_names = ["full", "first", "middle", "final", "tfmap_context_full"]
    expected_keys = {(trial_id, candidate) for trial_id in trial_ids for candidate in candidate_names}
    acoustic_by_id = {row["trial_id"]: row for row in acoustic_rows}
    natural_by_id = {row["trial_id"]: row for row in natural_rows}
    old_by_id = {row["trial_id"]: row for row in old_metrics}
    speaker_keys = {(row["trial_id"], row["candidate"]) for row in speaker_rows}
    asr_keys = {(row["trial_id"], row["candidate"]) for row in asr_rows}

    checks: dict[str, Any] = {}
    checks["trial_count_5991"] = len(candidate_rows) == 5991
    checks["trial_ids_unique"] = len(set(trial_ids)) == 5991
    checks["all_rows_dev"] = all(row["split"] == "dev" for row in candidate_rows)
    checks["candidate_names_exact"] = all(
        list(row["candidates"]) == candidate_names for row in candidate_rows
    )
    checks["candidate_key_count_29955"] = len(expected_keys) == 29955
    checks["speaker_keys_exact"] = speaker_keys == expected_keys
    checks["asr_keys_exact"] = asr_keys == expected_keys
    checks["all_candidate_waveforms_exist"] = all(
        Path(row["candidates"][candidate]["output_wav"]).is_file()
        for row in candidate_rows
        for candidate in candidate_names
    )
    checks["cohorts_exact"] = (
        sum(row["cohort"] == "natural_primary_swap" for row in candidate_rows) == 405
        and sum(row["cohort"] == "primary_correct_control" for row in candidate_rows)
        == 5586
    )
    selected_swap_ids = {
        row["trial_id"] for row in candidate_rows if row["cohort"] == "natural_primary_swap"
    }
    checks["swap_ids_match_frozen_file"] = selected_swap_ids == {
        row["trial_id"] for row in frozen_swaps
    }
    checks["control_ids_match_frozen_definition"] = {
        row["trial_id"]
        for row in candidate_rows
        if row["cohort"] == "primary_correct_control"
    } == {row["trial_id"] for row in natural_rows if not row["wrong_margin_0"]}

    margin_delta = target_delta = interferer_delta = 0.0
    wer_delta = 0.0
    content_switch_disagreements = 0
    for row in candidate_rows:
        trial_id = row["trial_id"]
        full = acoustic_by_id[trial_id]["candidates"]["full"]
        natural = natural_by_id[trial_id]
        target_delta = max(target_delta, abs(full["sisdr_target_db"] - natural["sisdr_target"]))
        interferer_delta = max(
            interferer_delta, abs(full["sisdr_interferer_db"] - natural["sisdr_interferer"])
        )
        margin_delta = max(
            margin_delta,
            abs(full["sisdr_margin_db"] - natural["speaker_selection_margin"]),
        )
        joined_full = row["candidates"]["full"]
        old = old_by_id[trial_id]
        wer_delta = max(wer_delta, abs(joined_full["target_WER"] - old["target_WER"]))
        content_switch_disagreements += joined_full["content_switch"] != old["content_switch"]
    checks["primary_sisdr_matches_frozen_audit"] = max(
        margin_delta, target_delta, interferer_delta
    ) < 1e-9
    checks["primary_wer_matches_frozen_protocol"] = wer_delta < 1e-12
    checks["primary_content_switch_matches_frozen_protocol"] = content_switch_disagreements == 0
    checks["scope_flags_fail_closed"] = (
        results["scope"]["split"] == "dev"
        and results["scope"]["test_used"] is False
        and results["scope"]["training_used"] is False
        and results["scope"]["selector_trained"] is False
    )
    runner_source = (ROOT / "scripts/tse_candidate_gate/run_frozen_wesep_candidates.py").read_text(
        encoding="utf-8"
    )
    checks["inference_runner_has_no_clean_reference_fields"] = (
        'row["target_wav"]' not in runner_source
        and 'row["interferer_wav"]' not in runner_source
    )
    checks["required_gate_artifacts_exist"] = all(
        path.is_file()
        for path in (
            ROOT / "docs/ALTERNATIVE_CANDIDATE_GATE_REPORT.md",
            GATE / "per_trial_candidates.jsonl",
            GATE / "candidate_summary.csv",
            GATE / "oracle_candidate_results.json",
        )
    )

    failures = [name for name, passed in checks.items() if not passed]
    validation = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "counts": {
            "trials": len(candidate_rows),
            "candidate_metrics": len(expected_keys),
            "speaker_metrics": len(speaker_rows),
            "asr_metrics": len(asr_rows),
            "natural_primary_swaps": len(selected_swap_ids),
        },
        "primary_reproduction_max_abs_delta": {
            "sisdr_target_db": target_delta,
            "sisdr_interferer_db": interferer_delta,
            "sisdr_margin_db": margin_delta,
            "target_WER": wer_delta,
            "content_switch_disagreements": content_switch_disagreements,
        },
        "checkpoint_sha256": {
            "primary_avg_model": sha256(ROOT / "pretrained/wesep/spk_emb_100/avg_model.pt"),
            "primary_config": sha256(ROOT / "pretrained/wesep/spk_emb_100/config.yaml"),
            "tfmap_context_avg_model": sha256(
                ROOT / "pretrained/wesep/tfmap_context_100/avg_model.pt"
            ),
            "tfmap_context_config": sha256(
                ROOT / "pretrained/wesep/tfmap_context_100/config.yaml"
            ),
        },
        "gate_decision": results["gate_evaluation"]["decision"],
    }
    output = GATE / "validation.json"
    output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
