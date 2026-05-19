from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_device(config: dict[str, Any]) -> str:
    import torch

    device = config.get("training", {}).get("device", "auto")
    if device != "auto":
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"
