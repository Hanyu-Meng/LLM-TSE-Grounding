#!/usr/bin/env python3
"""Resource safety guard for serial selected-evidence inference stages.

The module is intentionally dependency-light so heavy jobs can call
``check_or_raise`` between batches. The CLI can also perform a preflight check
or monitor one already-running PID. It never restarts a stopped job.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GIB = 1024 ** 3


@dataclass(frozen=True)
class Thresholds:
    host_warn_percent: float = 80.0
    host_block_new_percent: float = 88.0
    host_stop_percent: float = 92.0
    swap_warn_percent: float = 50.0
    swap_stop_percent: float = 75.0
    gpu_min_free_gib_for_new_model: float = 6.0
    gpu_warn_used_gib: float = 28.0
    gpu_stop_used_gib: float = 30.0
    disk_warn_free_gib: float = 100.0
    disk_stop_free_gib: float = 50.0


THRESHOLDS = Thresholds()


class ResourceGuardStop(RuntimeError):
    """Raised at a safe batch boundary when a graceful stop is required."""


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    return values


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()[0]
        index, name, total, used, free, utilization = [value.strip() for value in line.split(",")]
        return {
            "available": True,
            "index": int(index),
            "name": name,
            "total_bytes": int(total) * 1024 ** 2,
            "used_bytes": int(used) * 1024 ** 2,
            "free_bytes": int(free) * 1024 ** 2,
            "utilization_percent": float(utilization),
        }
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def snapshot(disk_path: str | Path) -> dict[str, Any]:
    memory = _meminfo()
    memory_total = memory["MemTotal"]
    memory_available = memory["MemAvailable"]
    memory_used = memory_total - memory_available
    swap_total = memory.get("SwapTotal", 0)
    swap_free = memory.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    disk = shutil.disk_usage(Path(disk_path))
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "host": {
            "total_bytes": memory_total,
            "used_bytes": memory_used,
            "available_bytes": memory_available,
            "used_percent": 100.0 * memory_used / memory_total,
        },
        "swap": {
            "total_bytes": swap_total,
            "used_bytes": swap_used,
            "free_bytes": swap_free,
            "used_percent": 100.0 * swap_used / swap_total if swap_total else 0.0,
        },
        "gpu": _gpu_snapshot(),
        "disk": {
            "path": str(Path(disk_path).resolve()),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "used_percent": 100.0 * disk.used / disk.total,
        },
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def evaluate(values: dict[str, Any], starting_new_stage: bool) -> dict[str, Any]:
    warnings: list[str] = []
    blockers: list[str] = []
    stops: list[str] = []
    host = values["host"]["used_percent"]
    swap = values["swap"]["used_percent"]
    disk_free = values["disk"]["free_bytes"] / GIB
    if host >= THRESHOLDS.host_stop_percent:
        stops.append(f"host RAM used {host:.1f}% >= {THRESHOLDS.host_stop_percent:.0f}%")
    elif host >= THRESHOLDS.host_block_new_percent:
        blockers.append(
            f"host RAM used {host:.1f}% >= {THRESHOLDS.host_block_new_percent:.0f}%"
        )
    elif host >= THRESHOLDS.host_warn_percent:
        warnings.append(f"host RAM used {host:.1f}% >= {THRESHOLDS.host_warn_percent:.0f}%")
    if swap >= THRESHOLDS.swap_stop_percent:
        stops.append(f"swap used {swap:.1f}% >= {THRESHOLDS.swap_stop_percent:.0f}%")
    elif swap >= THRESHOLDS.swap_warn_percent:
        warnings.append(f"swap used {swap:.1f}% >= {THRESHOLDS.swap_warn_percent:.0f}%")
    gpu = values["gpu"]
    if gpu.get("available"):
        gpu_used = gpu["used_bytes"] / GIB
        gpu_free = gpu["free_bytes"] / GIB
        if gpu_used >= THRESHOLDS.gpu_stop_used_gib:
            stops.append(
                f"GPU used {gpu_used:.2f} GiB >= {THRESHOLDS.gpu_stop_used_gib:.0f} GiB"
            )
        elif gpu_used >= THRESHOLDS.gpu_warn_used_gib:
            warnings.append(
                f"GPU used {gpu_used:.2f} GiB >= {THRESHOLDS.gpu_warn_used_gib:.0f} GiB"
            )
        if starting_new_stage and gpu_free < THRESHOLDS.gpu_min_free_gib_for_new_model:
            blockers.append(
                f"GPU free {gpu_free:.2f} GiB < {THRESHOLDS.gpu_min_free_gib_for_new_model:.0f} GiB"
            )
    elif starting_new_stage:
        blockers.append("GPU status unavailable before heavy stage")
    if disk_free < THRESHOLDS.disk_stop_free_gib:
        stops.append(
            f"disk free {disk_free:.1f} GiB < {THRESHOLDS.disk_stop_free_gib:.0f} GiB"
        )
    elif disk_free < THRESHOLDS.disk_warn_free_gib:
        warnings.append(
            f"disk free {disk_free:.1f} GiB < {THRESHOLDS.disk_warn_free_gib:.0f} GiB"
        )
    if stops:
        decision = "GRACEFUL_STOP"
    elif starting_new_stage and blockers:
        decision = "BLOCK_NEW_STAGE"
    elif warnings or blockers:
        decision = "WARNING"
    else:
        decision = "SAFE"
    return {
        "decision": decision,
        "warnings": warnings,
        "blockers": blockers,
        "stop_reasons": stops,
        "starting_new_stage": starting_new_stage,
        "thresholds": asdict(THRESHOLDS),
    }


def append_log(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check(
    *,
    phase: str,
    disk_path: str | Path,
    log_path: str | Path | None = None,
    starting_new_stage: bool = False,
) -> dict[str, Any]:
    values = snapshot(disk_path)
    result = {
        "phase": phase,
        "snapshot": values,
        "evaluation": evaluate(values, starting_new_stage),
    }
    if log_path is not None:
        append_log(log_path, result)
    return result


def check_or_raise(**kwargs: Any) -> dict[str, Any]:
    result = check(**kwargs)
    if result["evaluation"]["decision"] in {"GRACEFUL_STOP", "BLOCK_NEW_STAGE"}:
        reasons = (
            result["evaluation"]["stop_reasons"]
            or result["evaluation"]["blockers"]
        )
        raise ResourceGuardStop("; ".join(reasons))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--starting-new-stage", action="store_true")
    parser.add_argument("--monitor-pid", type=int)
    parser.add_argument("--interval", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        result = check(
            phase=args.phase,
            disk_path=args.disk_path,
            log_path=args.log,
            starting_new_stage=args.starting_new_stage,
        )
        print(json.dumps(result, indent=2), flush=True)
        decision = result["evaluation"]["decision"]
        if decision in {"GRACEFUL_STOP", "BLOCK_NEW_STAGE"}:
            return 3
        if args.monitor_pid is None:
            return 0
        try:
            os.kill(args.monitor_pid, 0)
        except ProcessLookupError:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
