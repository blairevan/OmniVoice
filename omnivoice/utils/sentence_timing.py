"""Sentence-level timing from WhisperX characters or weighted fallback."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from omnivoice.utils.whisperx_alignment import AlignedCharacter


@dataclass(frozen=True)
class SentenceTiming:
    """Represent one sentence boundary on a generated sample axis."""

    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class TimingResolution:
    """Return sentence timings and the evidence used to produce them."""

    timings: tuple[SentenceTiming, ...]
    timing_method: str
    alignment_coverage: float
    fallback_reason: str | None


def _alignment_units(text: str) -> list[str]:
    """Convert one normalized alignment string into non-whitespace units."""
    return [character for character in text if not character.isspace()]


def _lcs_matches(
    expected: Sequence[str],
    actual: Sequence[AlignedCharacter],
) -> dict[int, AlignedCharacter]:
    """Return deterministic longest-common-subsequence matches by expected index."""
    rows = len(expected) + 1
    columns = len(actual) + 1
    lengths = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        for column in range(1, columns):
            if expected[row - 1] == actual[column - 1].character:
                lengths[row][column] = lengths[row - 1][column - 1] + 1
            else:
                lengths[row][column] = max(lengths[row - 1][column], lengths[row][column - 1])

    matches: dict[int, AlignedCharacter] = {}
    row = len(expected)
    column = len(actual)
    while row and column:
        if expected[row - 1] == actual[column - 1].character:
            matches[row - 1] = actual[column - 1]
            row -= 1
            column -= 1
        elif lengths[row - 1][column] >= lengths[row][column - 1]:
            row -= 1
        else:
            column -= 1
    return matches


def _sentence_weights(texts: Sequence[str]) -> list[float]:
    """Calculate deterministic pronunciation and punctuation weights."""
    weights: list[float] = []
    for text in texts:
        cjk = sum(1 for character in text if "\u3400" <= character <= "\u9fff")
        latin_words = len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text))
        digits = len(re.findall(r"\d+(?:\.\d+)?", text))
        units = cjk + latin_words + digits
        if units == 0:
            units = sum(1 for character in text if character.isalpha())
        punctuation = sum(
            0.35 if character in ",，、；;" else 0.75
            for character in text
            if character in ",，、；;。.!！?？"
        )
        weights.append(max(0.1, units + punctuation))
    return weights


def build_weighted_sentence_timings(
    sentence_texts: Sequence[str],
    generated_samples: int,
    left_overlap_samples: int,
    right_overlap_samples: int,
) -> tuple[SentenceTiming, ...]:
    """Allocate a closed monotonic timeline over the non-overlap window."""
    if not sentence_texts:
        raise ValueError("At least one sentence is required")
    window_start = left_overlap_samples
    window_end = generated_samples - right_overlap_samples
    if generated_samples <= 0 or window_start >= window_end:
        raise ValueError("Overlap consumes the generated waveform")
    weights = _sentence_weights(sentence_texts)
    total = sum(weights)
    timings: list[SentenceTiming] = []
    previous = window_start
    for index, weight in enumerate(weights):
        end = (
            window_end
            if index == len(weights) - 1
            else window_start + round((window_end - window_start) * sum(weights[: index + 1]) / total)
        )
        if end <= previous:
            raise ValueError("Weighted timing produced a zero-length sentence")
        timings.append(SentenceTiming(previous, end))
        previous = end
    return tuple(timings)


def resolve_sentence_timings(
    sentence_texts: Sequence[str],
    characters: Sequence[AlignedCharacter],
    sample_rate: int,
    generated_samples: int,
    left_overlap_samples: int,
    right_overlap_samples: int,
    coverage_threshold: float = 0.95,
) -> TimingResolution:
    """Resolve sentence boundaries with 95% alignment and whole-group fallback."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    expected: list[str] = []
    sentence_ranges: list[tuple[int, int]] = []
    for text in sentence_texts:
        start = len(expected)
        expected.extend(_alignment_units(text))
        sentence_ranges.append((start, len(expected)))
    matches = _lcs_matches(expected, characters)
    coverage = len(matches) / len(expected) if expected else 0.0

    if coverage < coverage_threshold or any(
        not any(index in matches for index in range(start, end))
        for start, end in sentence_ranges
    ):
        return TimingResolution(
            build_weighted_sentence_timings(
                sentence_texts,
                generated_samples,
                left_overlap_samples,
                right_overlap_samples,
            ),
            "pronunciation_weight_fallback",
            coverage,
            "alignment_coverage_below_threshold_or_sentence_missing",
        )

    samples_by_index: dict[int, tuple[int, int]] = {
        index: (
            round(character.start * sample_rate),
            round(character.end * sample_rate),
        )
        for index, character in matches.items()
    }
    if min(samples_by_index) != 0 or max(samples_by_index) != len(expected) - 1:
        return TimingResolution(
            build_weighted_sentence_timings(
                sentence_texts,
                generated_samples,
                left_overlap_samples,
                right_overlap_samples,
            ),
            "pronunciation_weight_fallback",
            coverage,
            "alignment_missing_unbounded_unit",
        )
    for index in range(len(expected)):
        if index in samples_by_index:
            continue
        previous = max((key for key in samples_by_index if key < index), default=None)
        following = min((key for key in samples_by_index if key > index), default=None)
        if previous is None:
            start = 0
            end = samples_by_index[following][0] if following is not None else generated_samples
        elif following is None:
            start = samples_by_index[previous][1]
            end = generated_samples
        else:
            start = samples_by_index[previous][1]
            end = samples_by_index[following][0]
        samples_by_index[index] = (start, max(start + 1, end))

    timings: list[SentenceTiming] = []
    previous_end = left_overlap_samples
    for start, end in sentence_ranges:
        start_sample = max(previous_end, samples_by_index[start][0])
        end_sample = min(generated_samples - right_overlap_samples, samples_by_index[end - 1][1])
        if end_sample <= start_sample:
            return TimingResolution(
                build_weighted_sentence_timings(
                    sentence_texts,
                    generated_samples,
                    left_overlap_samples,
                    right_overlap_samples,
                ),
                "pronunciation_weight_fallback",
                coverage,
                "aligned_boundaries_invalid_after_overlap_clipping",
            )
        timings.append(SentenceTiming(start_sample, end_sample))
        previous_end = end_sample
    return TimingResolution(tuple(timings), "whisperx_alignment", coverage, None)


__all__ = [
    "SentenceTiming",
    "TimingResolution",
    "build_weighted_sentence_timings",
    "resolve_sentence_timings",
]
