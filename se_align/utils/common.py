"""Seed control and logging helpers."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all RNGs for reproducible decoding/eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "se_align", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
