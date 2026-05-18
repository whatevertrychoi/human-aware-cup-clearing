from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(RuntimeError):
    """Raised when config loading or validation fails."""


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigError(
            f"Config file not found: {config_file}. "
            "Create configs/config.yaml or pass --config with a valid path."
        )

    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {config_file}")

    return data


def get_required(config: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = config
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(traversed)
            raise ConfigError(f"Missing required config key: {joined}")
        current = current[key]
    return current

