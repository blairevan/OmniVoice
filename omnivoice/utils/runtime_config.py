"""Load deployment-owned OmniVoice runtime configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


class RuntimeConfigError(RuntimeError):
    """Report an invalid or unreadable OmniVoice runtime configuration."""


@dataclass(frozen=True)
class WhisperXRuntimeConfig:
    """Store optional WhisperX runtime paths loaded from OmniVoice YAML."""

    runtime_dir: str | None = None
    model_dir: str | None = None


def default_runtime_config_path() -> Path:
    """Return the repository-root OmniVoice configuration path."""
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _optional_path(section: Mapping[object, object], field: str) -> str | None:
    """Return one optional non-empty string path from a YAML mapping."""
    value = section.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(
            f"WhisperX configuration field {field} must be a non-empty string"
        )
    return value.strip()


def load_whisperx_runtime_config(
    path: Path | None = None,
) -> WhisperXRuntimeConfig:
    """Load and validate optional WhisperX paths from one YAML file."""
    config_path = path if path is not None else default_runtime_config_path()
    if not config_path.exists():
        return WhisperXRuntimeConfig()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RuntimeConfigError(
            f"OmniVoice runtime configuration has invalid YAML: {config_path}"
        ) from error
    except OSError as error:
        raise RuntimeConfigError(
            f"Unable to read OmniVoice runtime configuration: {config_path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeConfigError("OmniVoice runtime configuration root must be a mapping")
    if "whisperx" not in payload:
        return WhisperXRuntimeConfig()
    whisperx = payload["whisperx"]
    if not isinstance(whisperx, Mapping):
        raise RuntimeConfigError("OmniVoice runtime configuration whisperx must be a mapping")
    return WhisperXRuntimeConfig(
        runtime_dir=_optional_path(whisperx, "runtime_dir"),
        model_dir=_optional_path(whisperx, "model_dir"),
    )


__all__ = [
    "RuntimeConfigError",
    "WhisperXRuntimeConfig",
    "default_runtime_config_path",
    "load_whisperx_runtime_config",
]
