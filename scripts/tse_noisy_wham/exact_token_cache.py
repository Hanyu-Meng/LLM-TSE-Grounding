#!/usr/bin/env python3
"""Persistent exact signatures for immutable raw S3 token arrays."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class ExactTokenCache:
    """Cache SHA-256 token signatures and confirm every matching pair exactly."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.validated: set[str] = set()
        self.dirty = False
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = row.get("path")
                    if key is not None:
                        self.rows[str(key)] = row

    @staticmethod
    def _identity(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _array(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in self.arrays:
            value = np.load(path, allow_pickle=False).reshape(-1)
            if value.size == 0 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"invalid exact-token cache input: {path}")
            self.arrays[key] = value
        return self.arrays[key]

    def signature(
        self, path_value: str, *, allow_stale_prefilter: bool = False,
    ) -> dict[str, Any]:
        path = Path(path_value).resolve()
        key = str(path)
        prior = self.rows.get(key)
        if prior is not None and key in self.validated:
            return prior
        # A stale source signature may only cause a missed optimization.  It is
        # never accepted as equality: matching signatures are revalidated and
        # followed by full np.array_equal below.  Avoiding thousands of source
        # stat calls is important on the large waveform volume.
        if prior is not None and allow_stale_prefilter:
            return prior
        identity = self._identity(path)
        if prior is not None and prior.get("identity") == identity:
            self.validated.add(key)
            return prior
        value = self._array(path)
        digest = hashlib.sha256()
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes(order="C"))
        row = {
            "path": key,
            "identity": identity,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": digest.hexdigest(),
        }
        self.rows[key] = row
        self.validated.add(key)
        self.dirty = True
        return row

    def equal(self, left_value: str, right_value: str) -> bool:
        left_path = Path(left_value).resolve()
        right_path = Path(right_value).resolve()
        if left_path == right_path:
            return True
        left = self.signature(str(left_path))
        right = self.signature(str(right_path), allow_stale_prefilter=True)
        if (
            left["dtype"] != right["dtype"]
            or left["shape"] != right["shape"]
            or left["sha256"] != right["sha256"]
        ):
            return False
        right = self.signature(str(right_path))
        if (
            left["dtype"] != right["dtype"]
            or left["shape"] != right["shape"]
            or left["sha256"] != right["sha256"]
        ):
            return False
        # SHA-256 is used only as a prefilter; equality is still confirmed over
        # the complete integer arrays, so the scientific reuse rule is exact.
        a = self._array(left_path)
        b = self._array(right_path)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for key in sorted(self.rows):
                handle.write(json.dumps(self.rows[key]) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)
        self.dirty = False
