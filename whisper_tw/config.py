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


def resolve_common_voice_split_source(
    data_cfg: dict[str, Any],
    split: str,
) -> str | Path:
    split_paths = data_cfg.get("split_paths", {})
    if isinstance(split_paths, dict):
        split_source = split_paths.get(split)
        if split_source:
            return split_source

    split_tsv_key = f"{split}_tsv"
    if data_cfg.get(split_tsv_key):
        return data_cfg[split_tsv_key]

    split_name_key = f"{split}_split"
    if data_cfg.get(split_name_key):
        return data_cfg[split_name_key]

    return split
