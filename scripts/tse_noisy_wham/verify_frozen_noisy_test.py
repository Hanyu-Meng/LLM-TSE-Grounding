#!/usr/bin/env python3
"""Fail closed unless the frozen noisy TEST protocol and every file hash match."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "analysis/noisy_wham/FROZEN_NOISY_TEST_PROTOCOL.json"
COMPANION = PROTOCOL.with_suffix(".sha256")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not PROTOCOL.is_file() or not COMPANION.is_file():
        raise SystemExit("FATAL: frozen noisy TEST protocol/companion is absent")
    expected_self = COMPANION.read_text().split()[0]
    actual_self = sha256(PROTOCOL)
    if expected_self != actual_self:
        raise SystemExit("FATAL: frozen noisy TEST protocol SHA mismatch")
    frozen = json.loads(PROTOCOL.read_text())
    if (
        frozen.get("status") != "LOCKED_BEFORE_NOISY_TEST_MODEL_INFERENCE"
        or not frozen.get("test_execution_authorized")
        or frozen.get("noisy_test_model_outputs_present_at_lock")
    ):
        raise SystemExit("FATAL: noisy TEST execution is not validly authorized")
    mismatches = []
    for name, item in frozen["frozen_files"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = ROOT / path
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"]:
            mismatches.append({"name": name, "path": str(path), "expected": item["sha256"], "actual": actual})
    if mismatches:
        raise SystemExit("FATAL: post-freeze semantic/file change: " + json.dumps(mismatches))
    print(json.dumps({
        "status": "PASS",
        "frozen_protocol_sha256": actual_self,
        "verified_files": len(frozen["frozen_files"]),
        "post_freeze_semantic_change": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
