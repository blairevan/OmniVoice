"""Audio splicing and safe publication for subtitle-aligned regeneration."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from omnivoice.utils.audio import load_waveform
from omnivoice.utils.segment_timeline import (
    ReplacementRequest,
    SubtitleItem,
    SubtitleSplitter,
    load_subtitles,
    parse_cli_timestamp,
    parse_replacements,
    normalize_replacement_requests,
    rebuild_subtitle_timeline,
    resolve_cli_mode,
    validate_subtitles,
    validate_regeneration_inputs,
)
from omnivoice.utils.sentence_timing import (
    SentenceTiming,
    TimingResolution,
    build_weighted_sentence_timings,
)


SUPPORTED_OUTPUT_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
OVERLAP_DURATION_SECONDS = 0.015
SegmentGenerator = Callable[[ReplacementRequest, str], None]
SubtitleCleaner = Callable[[str], str]
AlignmentTextResolver = Callable[[str], str]
AlignmentResolver = Callable[
    [str, Sequence[str], int, int], TimingResolution
]
BeforeAlignment = Callable[[], None]


def _build_splice_plan(
    source: np.ndarray,
    sample_rate: int,
    requests: Sequence[ReplacementRequest],
    generated: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[int], tuple[int, ...]]:
    """Build original-axis pieces, replacement indexes, and actual overlaps."""
    pieces: list[np.ndarray] = []
    replacement_indexes: list[int] = []
    cursor = 0
    for request, replacement in zip(requests, generated):
        start_sample = round(request.start * sample_rate)
        end_sample = round(request.end * sample_rate)
        retained = source[:, cursor:start_sample]
        if retained.shape[-1] > 0:
            pieces.append(retained)
        pieces.append(replacement)
        replacement_indexes.append(len(pieces) - 1)
        cursor = end_sample
    trailing = source[:, cursor:]
    if trailing.shape[-1] > 0:
        pieces.append(trailing)
    overlaps = _choose_overlap_lengths(
        [piece.shape[-1] for piece in pieces],
        round(OVERLAP_DURATION_SECONDS * sample_rate),
    )
    return pieces, replacement_indexes, overlaps


def _covered_subtitle_indexes(
    subtitles: Sequence[SubtitleItem],
    request: ReplacementRequest,
) -> list[int]:
    """Return subtitle indexes covered by one complete replacement request."""
    return [
        index
        for index, item in enumerate(subtitles)
        if item.end > request.start + 0.01
        and item.start < request.end - 0.01
    ]


def _validate_sentence_timings(
    timings: Sequence[SentenceTiming],
    generated_samples: int,
) -> None:
    """Validate an alignment resolver result before output mapping."""
    previous_end = 0
    for timing in timings:
        if (
            timing.start_sample < 0
            or timing.end_sample > generated_samples
            or timing.start_sample < previous_end
            or timing.end_sample <= timing.start_sample
        ):
            raise ValueError("Alignment returned invalid sentence sample bounds")
        previous_end = timing.end_sample


def _clip_sentence_timings(
    timings: Sequence[SentenceTiming],
    window_start: int,
    window_end: int,
) -> tuple[SentenceTiming, ...]:
    """Clip aligned sentence boundaries to the non-overlap subtitle window."""
    clipped: list[SentenceTiming] = []
    previous = window_start
    for index, timing in enumerate(timings):
        start = previous
        end = window_end if index == len(timings) - 1 else min(window_end, timing.end_sample)
        if end <= start:
            raise ValueError("Alignment is empty after overlap clipping")
        clipped.append(SentenceTiming(start, end))
        previous = end
    return tuple(clipped)


@dataclass(frozen=True)
class RegenerationOptions:
    """Describe source and destination paths for one regeneration task."""

    source_audio: str
    source_subtitle: str
    output_audio: str
    output_srt: str | None
    output_json_subtitle: str | None
    max_char_len: int


@dataclass(frozen=True)
class RegenerationResult:
    """Summarize the generated audio and rebuilt subtitle timeline."""

    sample_rate: int
    total_seconds: float
    subtitles: tuple[SubtitleItem, ...]
    group_metadata: tuple[dict[str, object], ...] = ()


def validate_regeneration_paths(options: RegenerationOptions) -> None:
    """Reject missing inputs, path collisions, and existing output files."""
    source_paths = [Path(options.source_audio), Path(options.source_subtitle)]
    for source_path in source_paths:
        if not source_path.is_file():
            raise ValueError(f"Source file does not exist: {source_path}")
        if not os.access(source_path, os.R_OK):
            raise ValueError(f"Source file is not readable: {source_path}")

    output_values = [options.output_audio]
    if options.output_srt:
        output_values.append(options.output_srt)
    if options.output_json_subtitle:
        output_values.append(options.output_json_subtitle)
    if len(output_values) < 2:
        raise ValueError("At least one subtitle output path is required")

    source_resolved = {path.resolve() for path in source_paths}
    output_paths = [Path(value) for value in output_values]
    output_resolved = [path.resolve() for path in output_paths]
    if len(set(output_resolved)) != len(output_resolved):
        raise ValueError("Regeneration output paths must be distinct")
    for output_path, resolved_path in zip(output_paths, output_resolved):
        if resolved_path in source_resolved:
            raise ValueError(f"Output path must not match an input path: {output_path}")
        if output_path.exists():
            raise ValueError(f"Output path already exists: {output_path}")

    audio_extension = Path(options.output_audio).suffix.lower()
    if audio_extension not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise ValueError(f"Unsupported output audio format; expected one of: {supported}")
    if options.output_srt and Path(options.output_srt).suffix.lower() != ".srt":
        raise ValueError("SRT output path must use the .srt extension")
    if (
        options.output_json_subtitle
        and Path(options.output_json_subtitle).suffix.lower() != ".json"
    ):
        raise ValueError("JSON subtitle output path must use the .json extension")
    if options.max_char_len <= 0:
        raise ValueError("Maximum subtitle character length must be positive")


def _normalize_generated_audio(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    target_channels: int,
) -> np.ndarray:
    """Match one generated clip to the source sampling rate and channels."""
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError("Generated replacement audio must contain samples")
    normalized = waveform.astype(np.float32, copy=False)
    if sample_rate != target_sample_rate:
        normalized = librosa.resample(
            normalized,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
            axis=-1,
        ).astype(np.float32, copy=False)
    if normalized.shape[0] == target_channels:
        return normalized
    mono = np.mean(normalized, axis=0, keepdims=True)
    if target_channels == 1:
        return mono
    return np.repeat(mono, target_channels, axis=0)


def _choose_overlap_lengths(
    piece_lengths: Sequence[int], target_samples: int
) -> tuple[int, ...]:
    """Choose safe overlap lengths for each adjacent pair of pieces."""
    if target_samples < 0:
        raise ValueError("Overlap target must not be negative")
    if len(piece_lengths) < 2:
        return ()
    return tuple(
        min(target_samples, left, right)
        for left, right in zip(piece_lengths, piece_lengths[1:])
    )


def _overlap_add(
    left: np.ndarray,
    right: np.ndarray,
    overlap_samples: int,
) -> np.ndarray:
    """Join two channels-first waveforms with equal-power overlap-add."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Overlap-add waveforms must be channels-first with matching channels")
    overlap = min(overlap_samples, left.shape[-1], right.shape[-1])
    if overlap <= 0:
        return np.concatenate((left, right), axis=-1)
    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    left_gain = np.cos(phase)
    right_gain = np.sin(phase)
    mixed = left[:, -overlap:] * left_gain + right[:, :overlap] * right_gain
    return np.concatenate((left[:, :-overlap], mixed, right[:, overlap:]), axis=-1)


def _splice_waveforms(
    source: np.ndarray,
    sample_rate: int,
    requests: Sequence[ReplacementRequest],
    generated: Sequence[np.ndarray],
) -> np.ndarray:
    """Cut on the original sample timeline and overlap-add replacement clips."""
    pieces, _, overlaps = _build_splice_plan(source, sample_rate, requests, generated)
    if not pieces:
        raise ValueError("Regenerated audio contains no samples")
    result = pieces[0]
    for index, piece in enumerate(pieces[1:]):
        result = _overlap_add(result, piece, overlaps[index])
    return result


def _write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Encode a channels-first waveform according to the output extension."""
    if path.suffix.lower() != ".mp3":
        sf.write(path, waveform.T, sample_rate)
        return

    pcm = (waveform * 32768.0).clip(-32768, 32767).astype(np.int16)
    interleaved = pcm.T.reshape(-1)
    segment = AudioSegment(
        data=interleaved.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=waveform.shape[0],
    )
    segment.export(path, format="mp3")


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp with millisecond precision."""
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _serialize_srt(subtitles: Sequence[SubtitleItem]) -> str:
    """Serialize one internal timeline into consecutively numbered SRT."""
    blocks = []
    for index, item in enumerate(subtitles, 1):
        blocks.append(
            f"{index}\n{_format_srt_timestamp(item.start)} --> "
            f"{_format_srt_timestamp(item.end)}\n{item.text}"
        )
    return "\n\n".join(blocks) + "\n"


def _write_subtitle_outputs(
    options: RegenerationOptions,
    temp_dir: Path,
    subtitles: Sequence[SubtitleItem],
    sample_rate: int,
    total_seconds: float,
    group_metadata: Sequence[dict[str, object]] | None = None,
) -> list[tuple[Path, Path]]:
    """Write requested temporary subtitle artifacts from one shared timeline."""
    artifacts: list[tuple[Path, Path]] = []
    if options.output_srt:
        target = Path(options.output_srt)
        temporary = temp_dir / target.name
        temporary.write_text(_serialize_srt(subtitles), encoding="utf-8")
        artifacts.append((temporary, target))
    if options.output_json_subtitle:
        target = Path(options.output_json_subtitle)
        temporary = temp_dir / target.name
        payload = {
            "audio_file": Path(options.output_audio).name,
            "sample_rate": sample_rate,
            "total_seconds": round(total_seconds, 3),
            "sentences": [
                {"start": round(item.start, 3), "end": round(item.end, 3), "text": item.text}
                for item in subtitles
            ],
            "groups": list(group_metadata or ()),
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts.append((temporary, target))
    return artifacts


def _publish_artifacts(artifacts: Sequence[tuple[Path, Path]]) -> None:
    """Publish complete artifacts and roll back only files moved by this task."""
    published: list[Path] = []
    try:
        for temporary, target in artifacts:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as target_file:
                published.append(target)
                with temporary.open("rb") as temporary_file:
                    shutil.copyfileobj(temporary_file, target_file)
            temporary.unlink()
    except OSError:
        for target in published:
            target.unlink(missing_ok=True)
        raise


def regenerate_audio_segments(
    options: RegenerationOptions,
    requests: Sequence[ReplacementRequest],
    segment_generator: SegmentGenerator,
    subtitle_cleaner: SubtitleCleaner,
    subtitle_splitter: SubtitleSplitter,
    alignment_resolver: AlignmentResolver | None = None,
    alignment_text_resolver: AlignmentTextResolver | None = None,
    before_alignment: BeforeAlignment | None = None,
) -> RegenerationResult:
    """Generate replacement clips, splice audio, rebuild subtitles, and publish."""
    validate_regeneration_paths(options)
    source_waveform, source_sample_rate = load_waveform(options.source_audio)
    source_duration = source_waveform.shape[-1] / source_sample_rate
    source_subtitles = load_subtitles(options.source_subtitle)
    validate_regeneration_inputs(source_subtitles, requests, source_duration)

    with tempfile.TemporaryDirectory(prefix="omnivoice-regenerate-") as temp_value:
        temp_dir = Path(temp_value)
        generated_waveforms: list[np.ndarray] = []
        generated_durations: list[float] = []
        display_texts: list[list[str]] = []
        alignment_texts: list[list[str]] = []
        segment_paths: list[Path] = []
        for index, request in enumerate(requests, 1):
            segment_path = temp_dir / f"replacement-{index}.wav"
            segment_generator(request, str(segment_path))
            if not segment_path.is_file():
                raise RuntimeError(f"Replacement #{index} did not produce an audio file")
            waveform, sample_rate = load_waveform(str(segment_path))
            normalized = _normalize_generated_audio(
                waveform,
                sample_rate,
                source_sample_rate,
                source_waveform.shape[0],
            )
            generated_waveforms.append(normalized)
            generated_durations.append(normalized.shape[-1] / source_sample_rate)
            if isinstance(request.text, list):
                display_text = [subtitle_cleaner(item).strip() for item in request.text]
            else:
                display_text = [subtitle_cleaner(request.text).strip()]
            if not display_text or any(not item for item in display_text):
                raise ValueError(f"Replacement #{index} has no display subtitle text")
            display_texts.append(display_text)
            raw_texts = request.text if isinstance(request.text, list) else [request.text]
            resolved_alignment_texts = [
                (alignment_text_resolver(item) if alignment_text_resolver else item).strip()
                for item in raw_texts
            ]
            if len(resolved_alignment_texts) != len(display_text):
                raise ValueError(
                    f"Replacement #{index} has no one-to-one alignment subtitle text"
                )
            alignment_texts.append(resolved_alignment_texts)
            segment_paths.append(segment_path)

        if before_alignment is not None:
            before_alignment()

        pieces, replacement_indexes, overlaps = _build_splice_plan(
            source_waveform,
            source_sample_rate,
            requests,
            generated_waveforms,
        )
        final_waveform = pieces[0]
        for index, piece in enumerate(pieces[1:]):
            final_waveform = _overlap_add(final_waveform, piece, overlaps[index])
        overlap_durations = []
        piece_starts: list[int] = []
        output_samples = 0
        for index, piece in enumerate(pieces):
            left_overlap = overlaps[index - 1] if index > 0 else 0
            piece_starts.append(output_samples - left_overlap)
            output_samples += piece.shape[-1] - left_overlap
        for piece_index in replacement_indexes:
            left_overlap = overlaps[piece_index - 1] if piece_index > 0 else 0
            right_overlap = overlaps[piece_index] if piece_index < len(overlaps) else 0
            overlap_durations.append(
                (
                    left_overlap / source_sample_rate,
                    right_overlap / source_sample_rate,
                )
            )
        subtitles: list[SubtitleItem] = []
        group_metadata: list[dict[str, object]] = []
        subtitle_index = 0
        cumulative_delta_samples = 0
        for request_index, (request, generated, piece_index) in enumerate(
            zip(requests, generated_waveforms, replacement_indexes)
        ):
            covered_indexes = _covered_subtitle_indexes(source_subtitles, request)
            if not covered_indexes:
                raise ValueError("Replacement group does not cover source subtitles")
            while subtitle_index < covered_indexes[0]:
                item = source_subtitles[subtitle_index]
                offset = cumulative_delta_samples / source_sample_rate
                subtitles.append(SubtitleItem(round(item.start + offset, 6), round(item.end + offset, 6), item.text))
                subtitle_index += 1

            left_overlap = overlaps[piece_index - 1] if piece_index > 0 else 0
            right_overlap = overlaps[piece_index] if piece_index < len(overlaps) else 0
            generated_samples = generated.shape[-1]
            sentence_texts = display_texts[request_index]
            sentence_alignment_texts = alignment_texts[request_index]
            timings: Sequence[SentenceTiming]
            timing_method = "pronunciation_weight_fallback"
            alignment_coverage = 0.0
            fallback_reason = "alignment_not_requested"
            if len(sentence_texts) == 1:
                timings = (
                    SentenceTiming(left_overlap, generated_samples - right_overlap),
                )
                timing_method = "single_sentence_duration"
            else:
                if alignment_resolver is not None:
                    try:
                        candidate = alignment_resolver(
                            str(segment_paths[request_index]),
                            sentence_alignment_texts,
                            generated_samples,
                            source_sample_rate,
                        )
                        _validate_sentence_timings(candidate.timings, generated_samples)
                        if len(candidate.timings) != len(sentence_texts):
                            raise ValueError("Alignment sentence count does not match subtitles")
                        timings = _clip_sentence_timings(
                            candidate.timings,
                            left_overlap,
                            generated_samples - right_overlap,
                        )
                        timing_method = candidate.timing_method
                        fallback_reason = candidate.fallback_reason
                        alignment_coverage = candidate.alignment_coverage
                    except Exception as error:
                        fallback_reason = str(error)[:500]
                        timings = build_weighted_sentence_timings(
                            sentence_texts,
                            generated_samples,
                            left_overlap,
                            right_overlap,
                        )
                else:
                    timings = build_weighted_sentence_timings(
                        sentence_texts,
                        generated_samples,
                        left_overlap,
                        right_overlap,
                    )
            output_origin = piece_starts[piece_index]
            for item_index, timing, sentence_text in zip(
                covered_indexes, timings, sentence_texts
            ):
                del item_index
                subtitles.append(
                    SubtitleItem(
                        round((output_origin + timing.start_sample) / source_sample_rate, 6),
                        round((output_origin + timing.end_sample) / source_sample_rate, 6),
                        sentence_text,
                    )
                )
            subtitle_index = covered_indexes[-1] + 1
            original_samples = round((request.end - request.start) * source_sample_rate)
            group_delta = generated_samples - original_samples - left_overlap - right_overlap
            cumulative_delta_samples += group_delta
            group_metadata.append(
                {
                    "sentence_ids": [index + 1 for index in covered_indexes],
                    "timing_method": timing_method,
                    "alignment_coverage": alignment_coverage,
                    "fallback_reason": fallback_reason,
                    "generated_samples": generated_samples,
                    "left_overlap_samples": left_overlap,
                    "right_overlap_samples": right_overlap,
                    "delta_samples": group_delta,
                }
            )
        while subtitle_index < len(source_subtitles):
            item = source_subtitles[subtitle_index]
            offset = cumulative_delta_samples / source_sample_rate
            subtitles.append(SubtitleItem(round(item.start + offset, 6), round(item.end + offset, 6), item.text))
            subtitle_index += 1
        validate_subtitles(subtitles)

        output_audio = Path(options.output_audio)
        temporary_audio = temp_dir / output_audio.name
        _write_audio(temporary_audio, final_waveform, source_sample_rate)
        verified_waveform, verified_sample_rate = load_waveform(str(temporary_audio))
        total_seconds = verified_waveform.shape[-1] / verified_sample_rate
        if verified_sample_rate != source_sample_rate:
            raise RuntimeError("Encoded output sample rate does not match source audio")
        if subtitles[-1].end > total_seconds + 0.01:
            raise RuntimeError("Rebuilt subtitle timeline exceeds regenerated audio duration")

        artifacts = [(temporary_audio, output_audio)]
        artifacts.extend(
            _write_subtitle_outputs(
                options,
                temp_dir,
                subtitles,
                source_sample_rate,
                total_seconds,
                group_metadata,
            )
        )
        _publish_artifacts(artifacts)

    return RegenerationResult(
        sample_rate=source_sample_rate,
        total_seconds=total_seconds,
        subtitles=tuple(subtitles),
        group_metadata=tuple(group_metadata),
    )


__all__ = [
    "RegenerationOptions",
    "RegenerationResult",
    "ReplacementRequest",
    "SubtitleItem",
    "load_subtitles",
    "parse_cli_timestamp",
    "parse_replacements",
    "normalize_replacement_requests",
    "rebuild_subtitle_timeline",
    "regenerate_audio_segments",
    "resolve_cli_mode",
    "validate_regeneration_inputs",
    "validate_regeneration_paths",
]
