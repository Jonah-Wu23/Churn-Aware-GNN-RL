"""Reproducibility utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and framework-specific RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        return None


def get_git_commit() -> Optional[str]:
    """Return current git commit hash if available."""
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
