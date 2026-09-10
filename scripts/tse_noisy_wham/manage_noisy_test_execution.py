#!/usr/bin/env python3
"""Create a single resume-safe noisy TEST execution ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "analysis/noisy_wham/NOISY_TEST_EXECUTION.json"
FROZEN = ROOT / "analysis/noisy_wham/FROZEN_NOISY_TEST_PROTOCOL.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(value: dict) -> None:
    temporary = LEDGER.with_suffix(LEDGER.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(LEDGER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "complete"))
    args = parser.parse_args()
    digest = sha256(FROZEN)
    now = datetime.now(ZoneInfo("Australia/Sydney")).isoformat()
    if args.command == "start":
        if LEDGER.is_file():
            value = json.loads(LEDGER.read_text())
            if value.get("frozen_protocol_sha256") != digest:
                raise SystemExit("FATAL: TEST resume protocol differs from the started run")
            if value.get("status") == "COMPLETE":
                raise SystemExit("FATAL: noisy TEST already completed once; rerun prohibited")
            if value.get("status") != "STARTED" or value.get("execution_count") != 1:
                raise SystemExit("FATAL: invalid noisy TEST execution ledger")
            value["resume_count"] = int(value.get("resume_count", 0)) + 1
            value["last_resumed_at"] = now
        else:
            value = {
                "status": "STARTED", "execution_count": 1, "resume_count": 0,
                "started_at": now, "frozen_protocol_sha256": digest,
                "post_test_tuning": False,
            }
        atomic(value)
    else:
        if not LEDGER.is_file():
            raise SystemExit("FATAL: cannot complete absent TEST ledger")
        value = json.loads(LEDGER.read_text())
        if (value.get("status") != "STARTED" or value.get("execution_count") != 1
                or value.get("frozen_protocol_sha256") != digest):
            raise SystemExit("FATAL: invalid TEST ledger completion")
        value["status"] = "COMPLETE"
        value["completed_at"] = now
        atomic(value)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
