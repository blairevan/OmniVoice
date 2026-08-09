"""Apply guarded low-energy compression inside aligned connect boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from omnivoice.utils.connect_candidate_selector import CharacterTimestamp


@dataclass(frozen=True)
class ConnectProcessingOptions:
    """Configure conservative connect waveform editing."""

    maximum_shorten_ms: float = 120.0
    boundary_guard_ms: float = 20.0
    minimum_low_energy_ms: float = 30.0
    residual_gap_ms: float = 20.0
    crossfade_ms: float = 8.0
    relative_drop_db: float = 18.0
    absolute_ceiling_dbfs: float = -40.0
    frame_ms: float = 5.0
    hop_ms: float = 2.5
    zero_crossing_search_ms: float = 3.0


@dataclass(frozen=True)
class BoundaryEdit:
    """Record a single boundary decision in original sample coordinates."""

    left_text: str
    right_text: str
    removed_samples: int
    processed: bool
    skip_reason: str | None


@dataclass(frozen=True)
class ProcessedWaveform:
    """Return edited audio and exact sample accounting."""

    waveform: np.ndarray
    edits: tuple[BoundaryEdit, ...]
    total_removed_samples: int


def measure_low_energy_pauses(
    waveform: np.ndarray,
    sample_rate: int,
    boundaries: Sequence[tuple[CharacterTimestamp, CharacterTimestamp]],
    options: ConnectProcessingOptions,
) -> tuple[float, float]:
    """Measure guarded RMS low-energy pauses for stable candidate ranking."""
    mono = waveform.mean(axis=0) if waveform.ndim == 2 else waveform
    guard = round(options.boundary_guard_ms * sample_rate / 1000)
    frame = max(1, round(options.frame_ms * sample_rate / 1000))
    hop = max(1, round(options.hop_ms * sample_rate / 1000))
    pauses = []
    for left, right in boundaries:
        start = round(left.end_ms * sample_rate / 1000) + guard
        end = round(right.start_ms * sample_rate / 1000) - guard
        if end <= start:
            continue
        reference = np.concatenate((mono[max(0, start - round(.08 * sample_rate)):start], mono[end:min(mono.size, end + round(.08 * sample_rate))]))
        if not reference.size:
            continue
        threshold = min(_rms_dbfs(reference) - options.relative_drop_db, options.absolute_ceiling_dbfs)
        longest = current = 0
        for position in range(start, end, hop):
            if _rms_dbfs(mono[position:min(end, position + frame)]) <= threshold:
                current += hop; longest = max(longest, current)
            else:
                current = 0
        if longest >= round(options.minimum_low_energy_ms * sample_rate / 1000):
            pauses.append(longest * 1000 / sample_rate)
    return max(pauses, default=0.0), sum(pauses)


def _rms_dbfs(samples: np.ndarray) -> float:
    """Return a bounded dBFS RMS value for a non-empty analysis window."""
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(np.square(samples)))), 1e-12))


def _crossing(samples: np.ndarray, target: int, radius: int) -> int:
    """Find the nearest sign change, falling back to minimum amplitude."""
    start, end = max(1, target - radius), min(samples.size - 1, target + radius)
    choices = [index for index in range(start, end + 1) if samples[index] == 0 or samples[index - 1] == 0 or (samples[index - 1] < 0) != (samples[index] < 0)]
    return min(choices or list(range(start, end + 1)), key=lambda index: (abs(index - target), abs(samples[index])))


def process_connect_waveform(waveform: np.ndarray, sample_rate: int, boundaries: Sequence[tuple[CharacterTimestamp, CharacterTimestamp]], options: ConnectProcessingOptions) -> ProcessedWaveform:
    """Remove only safe low-energy centers, processing boundaries right to left."""
    if waveform.ndim not in {1, 2} or waveform.size == 0 or sample_rate <= 0 or not np.isfinite(waveform).all():
        raise ValueError("waveform must be finite audio with a positive sample rate")
    channels_first = waveform if waveform.ndim == 2 else waveform[np.newaxis, :]
    result = channels_first.astype(np.float32, copy=True)
    edits: list[BoundaryEdit] = []
    total_removed = 0
    guard = round(options.boundary_guard_ms * sample_rate / 1000)
    minimum = round(options.minimum_low_energy_ms * sample_rate / 1000)
    residual = round(options.residual_gap_ms * sample_rate / 1000)
    maximum = round(options.maximum_shorten_ms * sample_rate / 1000)
    fade = round(options.crossfade_ms * sample_rate / 1000)
    for left, right in reversed(boundaries):
        start = round(left.end_ms * sample_rate / 1000) + guard
        end = round(right.start_ms * sample_rate / 1000) - guard
        if end - start < minimum + residual:
            edits.append(BoundaryEdit(left.character, right.character, 0, False, "protected_region_overlap"))
            continue
        analysis = result.mean(axis=0)
        reference = np.concatenate((analysis[max(0, start - round(.08 * sample_rate)):start], analysis[end:min(analysis.size, end + round(.08 * sample_rate))]))
        if not reference.size:
            edits.append(BoundaryEdit(left.character, right.character, 0, False, "missing_local_reference")); continue
        threshold_db = min(_rms_dbfs(reference) - options.relative_drop_db, options.absolute_ceiling_dbfs)
        frame, hop = max(1, round(options.frame_ms * sample_rate / 1000)), max(1, round(options.hop_ms * sample_rate / 1000))
        runs, run_start, run_end = [], None, None
        for position in range(start, end, hop):
            frame_end = min(end, position + frame)
            if _rms_dbfs(analysis[position:frame_end]) <= threshold_db:
                run_start = position if run_start is None else run_start; run_end = frame_end
            elif run_start is not None:
                runs.append((run_start, run_end)); run_start = run_end = None
        if run_start is not None: runs.append((run_start, run_end))
        if not runs:
            edits.append(BoundaryEdit(left.character, right.character, 0, False, "no_continuous_low_energy_region"))
            continue
        best = max(runs, key=lambda item: item[1] - item[0])
        cut_budget = max(0, maximum - fade)
        removable = min(cut_budget, best[1] - best[0] - residual)
        if removable < minimum:
            edits.append(BoundaryEdit(left.character, right.character, 0, False, "no_continuous_low_energy_region"))
            continue
        cut_start = _crossing(analysis, best[0] + (best[1] - best[0] - removable) // 2, round(options.zero_crossing_search_ms * sample_rate / 1000))
        cut_end = _crossing(analysis, cut_start + removable, round(options.zero_crossing_search_ms * sample_rate / 1000))
        if cut_end <= cut_start: edits.append(BoundaryEdit(left.character, right.character, 0, False, "invalid_crossfade")); continue
        if cut_end - cut_start > maximum:
            cut_end = cut_start + maximum
        fade_size = min(
            fade,
            cut_start,
            result.shape[-1] - cut_end,
            max(0, maximum - (cut_end - cut_start)),
        )
        if fade_size:
            left_gain = np.cos(np.linspace(0, math.pi / 2, fade_size, dtype=np.float32))
            right_gain = np.sin(np.linspace(0, math.pi / 2, fade_size, dtype=np.float32))
            mixed = result[:, cut_start - fade_size:cut_start] * left_gain + result[:, cut_end:cut_end + fade_size] * right_gain
            result = np.concatenate((result[:, :cut_start - fade_size], mixed, result[:, cut_end + fade_size:]), axis=1)
            removed = cut_end - cut_start + fade_size
        else:
            result = np.concatenate((result[:, :cut_start], result[:, cut_end:]), axis=1); removed = cut_end - cut_start
        if removed > maximum:
            raise RuntimeError("connect edit exceeded maximum_shorten_ms")
        total_removed += removed
        edits.append(BoundaryEdit(left.character, right.character, removed, True, None))
    if channels_first.shape[-1] - result.shape[-1] != total_removed or not np.isfinite(result).all():
        raise RuntimeError("connect waveform sample accounting failed")
    return ProcessedWaveform(result if waveform.ndim == 2 else result[0], tuple(reversed(edits)), total_removed)
