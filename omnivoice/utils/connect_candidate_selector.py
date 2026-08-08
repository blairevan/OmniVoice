"""Validate character timestamps and rank connect candidate versions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from omnivoice.utils.connect_markup import ConnectMarkup, connect_boundaries


@dataclass(frozen=True)
class CharacterTimestamp:
    """Represent one aligned Chinese character in milliseconds."""

    character: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class CandidateScore:
    """Hold stable ranking values for one candidate version."""

    candidate_index: int
    version: str
    max_pause_ms: float
    total_pause_ms: float
    max_gap_ms: float
    total_gap_ms: float
    removed_samples: int

    def key(self) -> tuple[float, float, float, float, int, int, int]:
        """Return deterministic ascending rank, preferring original exact ties."""
        return (self.max_pause_ms, self.total_pause_ms, self.max_gap_ms, self.total_gap_ms, self.removed_samples, self.candidate_index, 0 if self.version == "original" else 1)


def validate_alignment(markup: ConnectMarkup, timestamps: Sequence[CharacterTimestamp], duration_ms: float) -> tuple[CharacterTimestamp, ...]:
    """Require aligned Chinese synthesis characters in order and within audio."""
    expected = list(markup.alignment_text or "".join(char for char in markup.synthesis_text if "\u3400" <= char <= "\u9fff"))
    if len(expected) != len(timestamps):
        raise ValueError("alignment character count does not match synthesis text")
    previous_end = 0.0
    for expected_char, item in zip(expected, timestamps):
        if item.character != expected_char:
            raise ValueError("alignment character order does not match synthesis text")
        if not all(math.isfinite(value) for value in (item.start_ms, item.end_ms)) or item.start_ms < previous_end or item.start_ms >= item.end_ms or item.end_ms > duration_ms:
            raise ValueError("alignment contains invalid timestamps")
        previous_end = item.end_ms
    return tuple(timestamps)


def internal_boundaries(markup: ConnectMarkup, timestamps: Sequence[CharacterTimestamp]) -> tuple[tuple[CharacterTimestamp, CharacterTimestamp], ...]:
    """Map connect-only synthesis offsets to aligned Chinese timestamp pairs."""
    chinese_offsets = list(markup.alignment_source_offsets) or [index for index, char in enumerate(markup.synthesis_text) if "\u3400" <= char <= "\u9fff"]
    offset_map = dict(zip(chinese_offsets, timestamps))
    result = []
    for left, right in connect_boundaries(markup.ranges):
        if left not in offset_map or right not in offset_map:
            raise ValueError("connect character lacks an alignment timestamp")
        result.append((offset_map[left], offset_map[right]))
    return tuple(result)
