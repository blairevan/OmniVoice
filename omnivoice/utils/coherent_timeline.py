"""Normalized text layers and token-bounded coherent synthesis groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


TokenCounter = Callable[[str], Sequence[object]]
TextRange = tuple[int, int]


@dataclass(frozen=True)
class SubtitleUnit:
    """Represent one stable subtitle sentence across all text layers."""

    sentence_id: int
    display_text: str
    synthesis_text: str
    alignment_text: str
    alignment_units: tuple[str, ...]
    synthesis_range: TextRange
    alignment_unit_range: TextRange
    explicit_pause_ms: float = 0.0
    alignment_source_offsets: tuple[int | None, ...] = ()
    connect_ranges: tuple[TextRange, ...] = ()


@dataclass(frozen=True)
class CoherentSynthesisGroup:
    """Represent adjacent subtitle units sent to TTS as one generation."""

    units: tuple[SubtitleUnit, ...]
    display_text: str
    synthesis_text: str
    alignment_text: str
    alignment_units: tuple[str, ...]
    synthesis_ranges: tuple[TextRange, ...]
    alignment_unit_ranges: tuple[TextRange, ...]
    explicit_pause_ms: float
    alignment_source_offsets: tuple[int | None, ...]
    connect_ranges: tuple[TextRange, ...]


def merge_coherent_units(
    units: Sequence[SubtitleUnit],
) -> CoherentSynthesisGroup:
    """Merge ordered subtitle units while rebuilding all local ranges."""
    if not units:
        raise ValueError("A coherent synthesis group requires at least one unit")

    ordered = tuple(units)
    sentence_ids = [unit.sentence_id for unit in ordered]
    if sentence_ids != sorted(set(sentence_ids)):
        raise ValueError("Coherent subtitle units must have ordered unique sentence IDs")

    display_parts: list[str] = []
    synthesis_parts: list[str] = []
    alignment_parts: list[str] = []
    alignment_units: list[str] = []
    synthesis_ranges: list[TextRange] = []
    alignment_ranges: list[TextRange] = []
    alignment_offsets: list[int | None] = []
    connect_ranges: list[TextRange] = []
    synthesis_offset = 0
    alignment_offset = 0

    for unit in ordered:
        display_parts.append(unit.display_text)
        synthesis_parts.append(unit.synthesis_text)
        alignment_parts.append(unit.alignment_text)
        alignment_units.extend(unit.alignment_units)
        synthesis_ranges.append(
            (
                synthesis_offset + unit.synthesis_range[0],
                synthesis_offset + unit.synthesis_range[1],
            )
        )
        alignment_ranges.append(
            (
                alignment_offset + unit.alignment_unit_range[0],
                alignment_offset + unit.alignment_unit_range[1],
            )
        )
        alignment_offsets.extend(
            None if offset is None else synthesis_offset + offset
            for offset in unit.alignment_source_offsets
        )
        connect_ranges.extend(
            (
                synthesis_offset + start,
                synthesis_offset + end,
            )
            for start, end in unit.connect_ranges
        )
        synthesis_offset += len(unit.synthesis_text)
        alignment_offset += len(unit.alignment_units)

    return CoherentSynthesisGroup(
        units=ordered,
        display_text="".join(display_parts),
        synthesis_text="".join(synthesis_parts),
        alignment_text="".join(alignment_parts),
        alignment_units=tuple(alignment_units),
        synthesis_ranges=tuple(synthesis_ranges),
        alignment_unit_ranges=tuple(alignment_ranges),
        explicit_pause_ms=sum(unit.explicit_pause_ms for unit in ordered),
        alignment_source_offsets=tuple(alignment_offsets),
        connect_ranges=tuple(connect_ranges),
    )


def build_coherent_synthesis_groups(
    units: Sequence[SubtitleUnit],
    tokenizer: TokenCounter | object | None,
    max_tokens: int,
    max_synthesis_chars: int,
) -> tuple[CoherentSynthesisGroup, ...]:
    """Pack adjacent units without exceeding token or text limits."""
    if not units:
        raise ValueError("Coherent synthesis requires at least one subtitle unit")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_synthesis_chars <= 0:
        raise ValueError("max_synthesis_chars must be positive")

    def token_count(text: str) -> int | None:
        """Return the model token count without mistaking tokenizer metadata for tokens."""
        if tokenizer is None:
            return None
        tokenize = getattr(tokenizer, "tokenize", None)
        if callable(tokenize):
            return len(tokenize(text))
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable or expose tokenize()")
        encoded = tokenizer(text)
        input_ids = (
            encoded.get("input_ids")
            if isinstance(encoded, dict)
            else getattr(encoded, "input_ids", None)
        )
        if input_ids is None:
            return len(encoded)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)

    def exceeds(group: CoherentSynthesisGroup) -> bool:
        """Check both configured safety limits for one candidate group."""
        measured_tokens = token_count(group.synthesis_text)
        if measured_tokens is not None:
            return measured_tokens > max_tokens
        return len(group.synthesis_text) > max_synthesis_chars

    result: list[CoherentSynthesisGroup] = []
    current: list[SubtitleUnit] = []
    for unit in units:
        candidate = merge_coherent_units((*current, unit))
        if current and exceeds(candidate):
            result.append(merge_coherent_units(current))
            current = []
            candidate = merge_coherent_units((unit,))
        if exceeds(candidate):
            raise ValueError(
                f"Subtitle unit {unit.sentence_id} exceeds the coherent synthesis limit"
            )
        current.append(unit)

    if current:
        result.append(merge_coherent_units(current))
    return tuple(result)


__all__ = [
    "CoherentSynthesisGroup",
    "SubtitleUnit",
    "build_coherent_synthesis_groups",
    "merge_coherent_units",
]
