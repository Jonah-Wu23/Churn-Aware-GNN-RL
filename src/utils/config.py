"""Config loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    This helper keeps YAML parsing localized and optional.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required to load config files") from exc

    with open(path, "r", encoding="ascii") as handle:
        return yaml.safe_load(handle) or {}
