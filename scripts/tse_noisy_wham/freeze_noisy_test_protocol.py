#!/usr/bin/env python3
"""Freeze DEV calibration, code, assets, and TEST inputs before TEST inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "analysis/noisy_wham/FROZEN_NOISY_TEST_PROTOCOL.json"
CACHE_ROOT = Path(
    os.environ.get("LLM_TSE_CACHE_ROOT", str(Path.home() / ".cache"))
).expanduser()
EXPECTED_BASE_PROTOCOL_SHA = "c67a7de6279bf942c89b3806558ecdc295c8c9f649f374154d8c3507fcbd1a30"
EXPECTED_AMENDMENT_SHA = "3efa66e26548dafe9501d34edd1a0c5ea059ba9cbf5c833e7ba6f43c2c7fcdcc"
EXPECTED_CLARIFICATION_SHA = "8f8cc747b184b8fd704def5e92a39d966db99fc4d8cdf09398f7a68af15c8d40"
EXPECTED_SCHEDULING_AMENDMENT_SHA = "a2ec1612c1e7593cf5a0d12f6d73f0341be0c12140e8b995b07c972a83d7d6cf"
EXPECTED_SCHEDULING_AUDIT_SHA = "23daa3b6214f9378e71b7628866ef39748dda6004d7d36c3416a14a1958966ea"
EXPECTED_GNR_RUNTIME_AMENDMENT_SHA = "f3b87b095e01af69f0455d94be98bc40c5843a3126f5016f2d38702779519d00"
EXPECTED_GNR_MECHANISM_AMENDMENT_SHA = "34a41bdbf2968f47512a79c23ea35e694695aa7a3cec4915a214c15d0012b5b4"
EXPECTED_REFERENCE_CACHE_AMENDMENT_SHA = "bd06e878236e3f1df0e2d4e21947c33ce1a9d36158b40faf0b27d827701a7d02"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_dev() -> dict:
    gnr_choice = read_json(ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json")
    selected_gnr = gnr_choice["selected_dev_slug"]
    systems = (
        "primary_wesep", "pool_d_selected", "pool_d_qfull_ud",
        "pool_d_fixed_csg", "pool_d_adaptive_csg",
        "pool_d_adaptive_csg_gnr_k50_r3",
        "pool_d_adaptive_csg_gnr_k20_r2",
    )
    for system in systems:
        path = ROOT / f"results/noisy_wham/dev/systems/{system}/summary.json"
        value = read_json(path)
        if value.get("status") != "COMPLETE" or value.get("expected") != 8400 or value.get("test_used"):
            raise ValueError(f"DEV core system incomplete: {system}")
    diagnostics = (
        "pool_d_grid_l0_w0", "pool_d_grid_l0p25_w0", "pool_d_grid_l0p5_w0",
        "pool_d_grid_l1_w0", "pool_d_grid_l1p5_w0", "pool_d_grid_l2_w0",
        "pool_d_source_csg_w0", "pool_d_source_csg_w1",
        "pool_d_tse_calibrated_csg", "pool_d_oracle_snr_csg",
    )
    for system in diagnostics:
        value = read_json(ROOT / f"results/noisy_wham/dev/systems/{system}/summary.json")
        if value.get("status") != "COMPLETE" or value.get("expected") != 2400 or value.get("test_used"):
            raise ValueError(f"DEV adaptive diagnostic incomplete: {system}")
    metrics = ("duration_estoi", "utmos", "speechbertscore", "lps")
    extended_systems = (
        "primary_wesep", "pool_d_selected", "pool_d_qfull_ud",
        "pool_d_fixed_csg", "pool_d_adaptive_csg", selected_gnr,
    )
    for metric in metrics:
        for system in extended_systems:
            path = ROOT / f"analysis/noisy_wham/dev/full/extended/{metric}/{system}/summary.json"
            value = read_json(path)
            expected = 8400
            if (
                value.get("status") != "COMPLETE"
                or value.get("expected") != expected
                or float(value.get("coverage", 0.0)) < 0.99
                or value.get("test_used")
            ):
                raise ValueError(f"DEV extended metric incomplete: {metric}/{system}: {value}")
    candidate = read_json(
        ROOT / "analysis/noisy_wham/dev/full/candidate_analysis/candidate_analysis.json"
    )
    if candidate.get("status") != "COMPLETE" or candidate.get("test_used"):
        raise ValueError("DEV candidate analysis incomplete")
    mechanism = read_json(
        ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism_v2/summary.json"
    )
    if (
        mechanism.get("status") != "COMPLETE"
        or mechanism.get("completed") != 8400
        or mechanism.get("failures") != 0
        or mechanism.get("test_used")
        or not mechanism.get("clean_reference_used_for_evaluation_analysis")
        or mechanism.get("target_token_source_trials") != 6000
        or mechanism.get("trials_with_one_boundary_position_excluded") != 820
        or mechanism.get("maximum_boundary_positions_excluded_per_trial") != 1
    ):
        raise ValueError("DEV GNR mechanism v2 analysis incomplete")
    calibration = read_json(
        ROOT / "analysis/noisy_wham/dev/full/adaptive/residual_snr_calibration.json"
    )
    if calibration.get("status") != "COMPLETE" or calibration.get("test_used"):
        raise ValueError("DEV residual calibration incomplete")
    for required in (
        "analysis/noisy_wham/dev/full/adaptive/tse_dev_calibrated_policy.json",
        "analysis/noisy_wham/dev/full/adaptive/temporal_tolerance_selection.json",
        "analysis/noisy_wham/dev/full/adaptive/frozen_adaptive_dev_choice.json",
        "analysis/noisy_wham/dev/full/gnr_selection.json",
        "analysis/noisy_wham/dev/full/best_generative_selection.json",
        "analysis/noisy_wham/dev/noisy_qfull_tail_bank.json",
    ):
        value = read_json(ROOT / required)
        if value.get("test_used") or value.get("status") not in {
            "FROZEN", "FROZEN_ON_DEV", "FROZEN_ON_NATURAL_NOISY_DEV"
        }:
            raise ValueError(f"DEV freeze artifact invalid: {required}")
    runtime = read_json(
        ROOT / "analysis/noisy_wham/dev/full/gnr_runtime_audit_v2.json"
    )
    if (
        runtime.get("status") != "PASS"
        or runtime.get("audit_version") != 2
        or runtime.get("test_used")
        or runtime.get("clean_reference_used")
        or runtime.get("structure_alignment", {}).get("status") != "PASS"
        or runtime.get("deployment_reproducibility", {}).get("status") != "PASS"
        or len(runtime.get("sample_selection", {}).get("examples", [])) < 8
    ):
        raise ValueError("selected full-DEV GNR runtime audit v2 failed")
    return calibration


def validate_metric_assets() -> dict:
    path = ROOT / "analysis/noisy_wham/metric_asset_audit.json"
    audit = read_json(path)
    required = ("utmos", "speechbertscore", "lps")
    if (
        audit.get("status") != "PASS"
        or audit.get("test_used")
        or not audit.get("heavy_models_serial")
        or not all(np.isfinite(float(audit.get("scores", {}).get(key, np.nan))) for key in required)
    ):
        raise ValueError(f"metric asset audit is not safe to freeze: {audit}")
    weight = Path(audit["utmos"]["weight"])
    if not weight.is_file() or sha256(weight) != audit["utmos"]["weight_sha256"]:
        raise ValueError("audited UTMOS weight is missing or changed")
    return audit


def validate_compute_optimizations() -> dict:
    """Require DEV evidence that compute-only reuse preserves metric values."""
    path = ROOT / "analysis/noisy_wham/optimization_audit/compute_reuse_audit.json"
    audit = read_json(path)
    if (
        audit.get("status") != "PASS"
        or audit.get("test_used")
        or any(value != "PASS" for value in audit.get("checks", {}).values())
    ):
        raise ValueError(f"compute-reuse audit is not safe to freeze: {audit}")
    # Runtime checks are emitted by the actual full-DEV processes. Whenever a
    # cache is exercised, require every configured fail-closed spot check.
    for system in (
        "pool_d_adaptive_csg",
        "pool_d_adaptive_csg_gnr_k20_r2",
        "pool_d_adaptive_csg_gnr_k50_r3",
    ):
        summary = read_json(
            ROOT / f"results/noisy_wham/dev/systems/{system}/audio_summary.json"
        )
        hits = int(summary.get("prompt_cache_hits", 0))
        checks = int(summary.get("prompt_cache_live_exact_checks", 0))
        if checks != min(hits, 2):
            raise ValueError(
                f"codec prompt-cache live check incomplete: {system}: {summary}"
            )
    hit_audit_path = (
        ROOT
        / "analysis/noisy_wham/optimization_audit/"
        "extended_reference_cache_hit_audit_v2.json"
    )
    hit_audit = read_json(hit_audit_path)
    if (
        hit_audit.get("status") != "PASS"
        or hit_audit.get("audit_version") != 2
        or hit_audit.get("test_used")
        or hit_audit.get("metric_values_changed") is not False
        or any(
            hit_audit.get(metric, {}).get("status") != "PASS"
            or len(hit_audit.get(metric, {}).get("examples", [])) != 2
            or not all(
                row.get("fresh_vs_cached_exact") is True
                for row in hit_audit.get(metric, {}).get("examples", [])
            )
            for metric in ("speechbertscore", "lps")
        )
    ):
        raise ValueError(f"reference-cache exact-hit audit failed: {hit_audit}")
    for metric in ("speechbertscore", "lps"):
        summaries = sorted(
            (ROOT / f"analysis/noisy_wham/dev/full/extended/{metric}").glob(
                "*/summary.json"
            )
        )
        if len(summaries) != 6:
            raise ValueError(f"missing full-DEV reference-cache evidence: {metric}")
        cache_roots = set()
        for summary_path in summaries:
            summary = read_json(summary_path)
            hits = int(summary.get("reference_cache_hits", 0))
            misses = int(summary.get("reference_cache_misses", 0))
            checks = int(summary.get("reference_cache_live_exact_checks", 0))
            expected = int(summary.get("expected", 0))
            cache_root = summary.get("shared_cache_root")
            if (
                summary.get("status") != "COMPLETE"
                or expected != 8400
                or int(summary.get("failures", -1)) != 0
                or hits + misses != expected
                or checks != min(hits, 2)
                or not cache_root
            ):
                raise ValueError(
                    f"reference-cache accounting incomplete: {summary_path}: {summary}"
                )
            cache_roots.add(str(Path(cache_root).resolve()))
        if len(cache_roots) != len(summaries):
            raise ValueError(f"parallel writers did not use private caches: {metric}")
    return {"compute_reuse": audit, "reference_cache_hit_audit": hit_audit}


def validate_scheduling_optimizations() -> dict:
    """Require the DEV-only exact-equivalence scheduling audit."""
    path = ROOT / "analysis/noisy_wham/optimization_audit/scheduling_v3_audit.json"
    if sha256(path) != EXPECTED_SCHEDULING_AUDIT_SHA:
        raise ValueError("v3 scheduling audit SHA changed")
    audit = read_json(path)
    decision = audit.get("batch_and_worker_trials", {}).get("decision", {})
    scheduler = audit.get("approved_scheduler", {})
    if (
        audit.get("status") != "PASS"
        or audit.get("test_used")
        or audit.get("resource_guard_observed") != "SAFE"
        or decision.get("asr_batch_size") != 1
        or decision.get("generative_batch_size") != 1
        or decision.get("num_workers") != 0
        or scheduler.get("extended_metrics", {}).get("systems_parallel_max") != 4
        or not scheduler.get("resource_guard_required_for_every_process")
    ):
        raise ValueError(f"v3 scheduling audit is not safe to freeze: {audit}")
    return audit


def ensure_no_test_outputs() -> None:
    forbidden = (
        ROOT / "analysis/noisy_wham/test/candidates_spk_emb",
        ROOT / "analysis/noisy_wham/test/candidates_tfmap_context",
        ROOT / "analysis/noisy_wham/test/full",
        ROOT / "results/noisy_wham/test",
        ROOT / "manifests/noisy_wham/test_full_inference.jsonl",
        ROOT / "manifests/noisy_wham/test_full_evaluation.jsonl",
    )
    present = []
    for path in forbidden:
        if path.is_file() or (path.is_dir() and any(path.rglob("*"))):
            present.append(str(path))
    if present:
        raise ValueError(f"noisy TEST model outputs already present: {present}")


def frozen_paths() -> list[Path]:
    paths = [
        ROOT / "analysis/noisy_wham/preregistered_noisy_protocol.json",
        ROOT / "analysis/noisy_wham/protocol_amendment_v2.json",
        ROOT / "analysis/noisy_wham/protocol_clarification_v2_1.json",
        ROOT / "analysis/noisy_wham/protocol_amendment_v3_compute_scheduling.json",
        ROOT / "analysis/noisy_wham/protocol_amendment_v4_gnr_runtime_audit.json",
        ROOT / "analysis/noisy_wham/protocol_amendment_v5_gnr_mechanism_target_tokens.json",
        ROOT / "analysis/noisy_wham/protocol_amendment_v6_reference_cache_gate.json",
        ROOT / "analysis/noisy_wham/dev/noisy_qfull_tail_bank.json",
        ROOT / "analysis/noisy_wham/metric_asset_audit.json",
        ROOT / "analysis/noisy_wham/optimization_audit/compute_reuse_audit.json",
        ROOT / "analysis/noisy_wham/optimization_audit/extended_reference_cache_hit_audit_v2.json",
        ROOT / "analysis/noisy_wham/optimization_audit/scheduling_v3_audit.json",
        ROOT / "docs/NOISY_COMPUTE_OPTIMIZATION_AUDIT.md",
        ROOT / "analysis/noisy_wham/test/candidate_inputs_full.jsonl",
        ROOT / "manifests/noisy_wham/natural_test.jsonl",
        ROOT / "manifests/noisy_wham/controlled_test.jsonl",
        ROOT / "analysis/noisy_wham/dev/full/adaptive/residual_snr_calibration.json",
        ROOT / "analysis/noisy_wham/dev/full/adaptive/tse_dev_calibrated_policy.json",
        ROOT / "analysis/noisy_wham/dev/full/adaptive/temporal_tolerance_selection.json",
        ROOT / "analysis/noisy_wham/dev/full/adaptive/frozen_adaptive_dev_choice.json",
        ROOT / "analysis/noisy_wham/dev/full/adaptive/frozen_adaptive_sidecar.jsonl",
        ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json",
        ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json",
        ROOT / "analysis/noisy_wham/dev/full/gnr_runtime_audit.json",
        ROOT / "analysis/noisy_wham/dev/full/gnr_runtime_audit_v2.json",
        ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism_v2/summary.json",
        ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism_v2/per_trial.jsonl",
        ROOT / "analysis/noisy_wham/dev/full/gnr_mechanism_v2/failures.jsonl",
        ROOT / "manifests/tse_dev_prepared.jsonl",
        ROOT / "docs/GNR_TSE_IMPLEMENTATION_AUDIT.md",
        ROOT / "experiments/qfull_sme_rawqwen_seed0_20260810_0858/checkpoints/best.pt",
        ROOT / "pretrained/wesep/spk_emb_100/avg_model.pt",
        ROOT / "pretrained/wesep/spk_emb_100/config.yaml",
        ROOT / "pretrained/wesep/tfmap_context_100/avg_model.pt",
        ROOT / "pretrained/wesep/tfmap_context_100/config.yaml",
        ROOT / "pretrained/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx",
        ROOT / "pretrained/Fun-CosyVoice3-0.5B/flow.pt",
        ROOT / "pretrained/Fun-CosyVoice3-0.5B/hift.pt",
        ROOT / "pretrained/Fun-CosyVoice3-0.5B/campplus.onnx",
        ROOT / "pretrained/whisper-small.en/model.safetensors",
        ROOT / "pretrained/Qwen2.5-0.5B-Instruct/model.safetensors",
        ROOT / "pretrained/wavlm-base-plus/pytorch_model.bin",
        ROOT / "grounding/fsq/fsq_codebook.npy",
        ROOT / "scripts/resource_guard.py",
        ROOT / "se_align/eval/metrics.py",
        ROOT / "se_align/codec/cosyvoice3_codec.py",
        CACHE_ROOT / "torch/hub/checkpoints/utmos22_strong_step7459_v1.pt",
    ]
    for pattern in (
        "scripts/tse_noisy_wham/*.py",
        "scripts/tse_noisy_wham/run_noisy_*.sh",
        "scripts/tse_candidate_gate/*.py",
        "scripts/tse_selected_evidence/decode_selected_evidence.py",
        "scripts/tse_selected_evidence/tokenize_selected_evidence_safe.py",
        "scripts/tse_selected_evidence/evaluate_selected_audio_safe.py",
        "scripts/tse_selected_evidence/evaluate_selected_asr_safe.py",
        "se_align/tse/*.py",
        "artifacts/tse/dev_features/target_tokens/*.npy",
    ):
        paths.extend(sorted(ROOT.glob(pattern)))
    utmos_source = CACHE_ROOT / "torch/hub/tarepan_SpeechMOS_v1.2.0"
    paths.extend(sorted(utmos_source.glob("*.py")))
    paths.extend(sorted((utmos_source / "speechmos").rglob("*.py")))
    phoneme_cache = (
        CACHE_ROOT
        / "huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft"
    )
    paths.extend(sorted(path for path in phoneme_cache.rglob("*") if path.is_file()))
    if not phoneme_cache.is_dir():
        raise FileNotFoundError("frozen LPS phoneme model cache is absent")
    unique = sorted(set(paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen file missing: {missing}")
    return unique


def main() -> int:
    base = ROOT / "analysis/noisy_wham/preregistered_noisy_protocol.json"
    if sha256(base) != EXPECTED_BASE_PROTOCOL_SHA:
        raise ValueError("base noisy protocol SHA changed")
    amendment = ROOT / "analysis/noisy_wham/protocol_amendment_v2.json"
    clarification = ROOT / "analysis/noisy_wham/protocol_clarification_v2_1.json"
    scheduling_amendment = (
        ROOT / "analysis/noisy_wham/protocol_amendment_v3_compute_scheduling.json"
    )
    gnr_runtime_amendment = (
        ROOT / "analysis/noisy_wham/protocol_amendment_v4_gnr_runtime_audit.json"
    )
    gnr_mechanism_amendment = (
        ROOT
        / "analysis/noisy_wham/protocol_amendment_v5_gnr_mechanism_target_tokens.json"
    )
    reference_cache_amendment = (
        ROOT / "analysis/noisy_wham/protocol_amendment_v6_reference_cache_gate.json"
    )
    if sha256(amendment) != EXPECTED_AMENDMENT_SHA:
        raise ValueError("noisy v2 amendment SHA changed")
    if sha256(clarification) != EXPECTED_CLARIFICATION_SHA:
        raise ValueError("noisy v2.1 clarification SHA changed")
    if sha256(scheduling_amendment) != EXPECTED_SCHEDULING_AMENDMENT_SHA:
        raise ValueError("noisy v3 scheduling amendment SHA changed")
    if sha256(gnr_runtime_amendment) != EXPECTED_GNR_RUNTIME_AMENDMENT_SHA:
        raise ValueError("noisy v4 GNR runtime amendment SHA changed")
    if sha256(gnr_mechanism_amendment) != EXPECTED_GNR_MECHANISM_AMENDMENT_SHA:
        raise ValueError("noisy v5 GNR mechanism amendment SHA changed")
    if sha256(reference_cache_amendment) != EXPECTED_REFERENCE_CACHE_AMENDMENT_SHA:
        raise ValueError("noisy v6 reference-cache amendment SHA changed")
    ensure_no_test_outputs()
    calibration = validate_dev()
    adaptive_choice = read_json(
        ROOT / "analysis/noisy_wham/dev/full/adaptive/frozen_adaptive_dev_choice.json"
    )
    gnr_choice = read_json(ROOT / "analysis/noisy_wham/dev/full/gnr_selection.json")
    best_generative = read_json(
        ROOT / "analysis/noisy_wham/dev/full/best_generative_selection.json"
    )
    metric_assets = validate_metric_assets()
    compute_optimizations = validate_compute_optimizations()
    scheduling_optimizations = validate_scheduling_optimizations()
    paths = frozen_paths()
    frozen_files = {}
    for path in paths:
        try:
            display = str(path.relative_to(ROOT))
        except ValueError:
            display = str(path)
        frozen_files[display] = {
            "path": display, "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    result = {
        "status": "LOCKED_BEFORE_NOISY_TEST_MODEL_INFERENCE",
        "frozen_at": datetime.now(ZoneInfo("Australia/Sydney")).isoformat(),
        "base_protocol_sha256": EXPECTED_BASE_PROTOCOL_SHA,
        "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA,
        "protocol_clarification_sha256": EXPECTED_CLARIFICATION_SHA,
        "protocol_scheduling_amendment_sha256": EXPECTED_SCHEDULING_AMENDMENT_SHA,
        "protocol_gnr_runtime_amendment_sha256": EXPECTED_GNR_RUNTIME_AMENDMENT_SHA,
        "protocol_gnr_mechanism_amendment_sha256": EXPECTED_GNR_MECHANISM_AMENDMENT_SHA,
        "protocol_reference_cache_amendment_sha256": EXPECTED_REFERENCE_CACHE_AMENDMENT_SHA,
        "test_execution_authorized": True,
        "noisy_test_model_outputs_present_at_lock": False,
        "test_execution_count_before_lock": 0,
        "post_test_tuning_allowed": False,
        "frozen_configuration": {
            "candidate_order": ["full", "first", "middle", "final", "tfmap_context_full"],
            "candidate_selection": "frozen enrollment ECAPA cosine and fixed tie order",
            "fixed_csg": {"lambda": 1.0, "temporal_tolerance": 0},
            "adaptive_csg": {
                "policy_name": adaptive_choice["selected_policy"],
                "residual_snr_valid": calibration["residual_snr_valid"],
                "theil_sen_slope": calibration["slope"],
                "theil_sen_intercept": calibration["intercept"],
                "lambda_values": [2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0],
                "thresholds": [-3.013, -1.640, -0.267, 1.179, 2.697, 4.216, 7.515, 13.158],
                "temporal_tolerance": adaptive_choice["selected_temporal_tolerance"],
            },
            "gnr": {"K": gnr_choice["K"], "R": gnr_choice["R"],
                    "anchor": "full frozen adaptive CSG", "refined_feedback": False},
            "best_generative_comparison_arm": best_generative["selected_code"],
            "batch_size": 1,
            "num_workers_max": 0,
            "seed": 1986,
            "execution_schedule": {
                "generative_batch_size": 1,
                "num_workers": 0,
                "independent_generation_branches_max": 2,
                "extended_metric_systems_parallel_max": 4,
                "cosyvoice_processes_max": 1,
                "private_cache_namespace_for_parallel_writers": True,
            },
            "test_trials": {"natural": 6000, "controlled": 2400, "total": 8400},
        },
        "dev_calibration": calibration,
        "metric_asset_audit": metric_assets,
        "compute_optimization_audit": compute_optimizations,
        "scheduling_optimization_audit": scheduling_optimizations,
        "frozen_files": frozen_files,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "tail_protocol": {
            "dev_ids_frozen": True,
            "test_algorithm_frozen_before_test": True,
            "ranking": "natural Pool-D Q-Full UD raw WER descending, trial_id tie-break",
            "sets": [50, 100, 200, "10_percent"],
        },
    }
    atomic(OUTPUT, json.dumps(result, indent=2) + "\n")
    digest = sha256(OUTPUT)
    atomic(OUTPUT.with_suffix(".sha256"), f"{digest}  {OUTPUT.name}\n")
    print(json.dumps({
        "status": result["status"],
        "frozen_protocol_sha256": digest,
        "frozen_files": len(frozen_files),
        "noisy_test_model_outputs_present_at_lock": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
