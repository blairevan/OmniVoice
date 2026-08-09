"""Invoke the shared isolated WhisperX runtime for known text alignment."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class WhisperXAlignmentError(RuntimeError):
    """Represent an alignment failure that can safely trigger timing fallback."""


@dataclass(frozen=True)
class AlignedCharacter:
    """Represent one valid character timestamp from WhisperX."""

    character: str
    start: float
    end: float


@dataclass(frozen=True)
class WhisperXAlignmentDeadline:
    """Track one monotonic deadline shared by all group alignments."""

    started_at: float
    timeout_seconds: float

    @classmethod
    def from_timeout_seconds(cls, timeout_seconds: float) -> "WhisperXAlignmentDeadline":
        """Create a deadline from a positive task timeout."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("WhisperX timeout must be positive and finite")
        return cls(time.monotonic(), timeout_seconds)

    def remaining_seconds(self) -> float:
        """Return remaining time or zero after the task deadline has passed."""
        return max(0.0, self.started_at + self.timeout_seconds - time.monotonic())


class WhisperXAligner:
    """Run the shared fixed WhisperX environment without importing it into OmniVoice."""

    def __init__(
        self,
        runtime_dir: str,
        language: str,
        device: str,
        model_name: str,
    ) -> None:
        """Store explicit runtime, language, device, and local model settings."""
        if device != "cuda":
            raise ValueError("WhisperX device must be cuda")
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.language = language
        self.device = device
        self.model_name = Path(model_name).expanduser().resolve()

    def _validate_paths(self) -> None:
        """Verify all offline runtime inputs before spawning a child process."""
        required = (
            self.runtime_dir / "pyproject.toml",
            self.runtime_dir / "align.py",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if not self.model_name.is_dir():
            missing.append(str(self.model_name))
        if missing:
            raise WhisperXAlignmentError(
                "WhisperX runtime or local model is missing: " + ", ".join(missing)
            )

    def align(
        self,
        audio_path: str,
        text: str,
        start: float,
        end: float,
        deadline: WhisperXAlignmentDeadline,
    ) -> tuple[AlignedCharacter, ...]:
        """Return character timestamps for one known transcript interval."""
        if not text.strip():
            raise ValueError("WhisperX alignment text must not be blank")
        if start < 0 or end <= start:
            raise ValueError("WhisperX alignment interval must be non-empty")
        remaining = deadline.remaining_seconds()
        if remaining <= 0:
            raise WhisperXAlignmentError("WhisperX task deadline has expired")
        self._validate_paths()

        with tempfile.TemporaryDirectory(prefix="omnivoice-whisperx-") as temp_dir:
            output_path = Path(temp_dir) / "alignment.json"
            command = [
                "uv",
                "run",
                "--project",
                str(self.runtime_dir),
                "python",
                str(self.runtime_dir / "align.py"),
                "--audio",
                audio_path,
                "--text",
                text,
                "--language",
                self.language,
                "--device",
                self.device,
                "--model-name",
                str(self.model_name),
                "--start",
                str(start),
                "--end",
                str(end),
                "--output",
                str(output_path),
            ]
            environment = os.environ.copy()
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise WhisperXAlignmentError("WhisperX alignment timed out") from error
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                raise WhisperXAlignmentError(
                    f"WhisperX alignment process failed: {detail[:1000]}"
                ) from error
            del completed
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise WhisperXAlignmentError("WhisperX alignment JSON is invalid") from error
        return _parse_characters(payload)


def _parse_characters(payload: object) -> tuple[AlignedCharacter, ...]:
    """Extract finite character timestamps from one WhisperX response."""
    if not isinstance(payload, dict):
        raise WhisperXAlignmentError("WhisperX alignment output must be an object")
    segments = payload.get("segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise WhisperXAlignmentError("WhisperX alignment output must contain one segment")
    chars = segments[0].get("chars") if isinstance(segments[0], dict) else None
    if not isinstance(chars, list):
        raise WhisperXAlignmentError("WhisperX alignment output has no chars")
    parsed: list[AlignedCharacter] = []
    previous_end = 0.0
    for item in chars:
        if not isinstance(item, dict):
            continue
        character = item.get("char")
        start = item.get("start")
        end = item.get("end")
        if (
            isinstance(character, str)
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and math.isfinite(float(start))
            and math.isfinite(float(end))
            and 0 <= float(start) < float(end)
            and float(start) >= previous_end
        ):
            parsed.append(AlignedCharacter(character, float(start), float(end)))
            previous_end = float(end)
    if not parsed:
        raise WhisperXAlignmentError("WhisperX alignment output has no valid chars")
    return tuple(parsed)


__all__ = [
    "AlignedCharacter",
    "WhisperXAligner",
    "WhisperXAlignmentDeadline",
    "WhisperXAlignmentError",
]
