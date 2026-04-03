"""
utils/config.py
---------------
YAML configuration loader with dot-notation access.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a YAML dict that supports attribute-style access."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and merge model_config.yaml and pipeline_config.yaml.

    Parameters
    ----------
    config_path:
        Explicit path to a single YAML file. When *None* (default) both
        ``configs/model_config.yaml`` and ``configs/pipeline_config.yaml``
        are loaded relative to the project root and merged.

    Returns
    -------
    Config
        Nested Config object with dot-notation access.
    """
    project_root = _find_project_root()

    if config_path is not None:
        raw = _read_yaml(Path(config_path))
    else:
        model_cfg = _read_yaml(project_root / "configs" / "model_config.yaml")
        pipeline_cfg = _read_yaml(project_root / "configs" / "pipeline_config.yaml")
        raw = {**model_cfg, **pipeline_cfg}

    return Config(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Walk up from this file until we find a directory that contains 'configs/'."""
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        "Could not locate project root (no 'configs/' directory found)."
    )


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
