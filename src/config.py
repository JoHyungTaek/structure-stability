from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = deepcopy(cfg)
    if overrides:
        override_dict: Dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid override: {item}")
            key, raw_value = item.split("=", 1)
            key_parts = key.split(".")
            cursor = override_dict
            for part in key_parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[key_parts[-1]] = yaml.safe_load(raw_value)
        cfg = _deep_update(cfg, override_dict)
    return cfg
