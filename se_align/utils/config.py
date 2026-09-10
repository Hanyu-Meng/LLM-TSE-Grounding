"""Tiny YAML config loader with dotted access and CLI overrides."""
from __future__ import annotations

import os
from typing import Any, Mapping

import yaml


class Config(dict):
    """dict that also supports attribute and dotted-key access.

    cfg.cosyvoice3.model_dir  and  cfg["cosyvoice3.model_dir"]  both work.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str) and "." in key and key not in self.keys():
            node: Any = self
            for part in key.split("."):
                node = dict.__getitem__(node, part) if isinstance(node, dict) else node[part]
            return node
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


def load_config(path: str, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load YAML and apply ``a.b.c=value`` style overrides."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(raw)
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=val`` CLI tokens; values are YAML-parsed."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(val)
    return out


def resolve_split_dirs(cfg: Config, split: str, subset: str) -> str:
    """Absolute path to a VB-DEMAND subset dir for (split, subset)."""
    sub = cfg["data.subdirs"][split][subset]
    return os.path.join(cfg["data.root"], sub)
