"""Audio splicing and safe publication for subtitle-aligned regeneration."""

from __future__ import annotations

import json
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
    rebuild_subtitle_timeline,
    resolve_cli_mode,
    validate_regeneration_inputs,
)


SUPPORTED_OUTPUT_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
SegmentGenerator = Callable[[str, str], None]
SubtitleCleaner = Callable[[str], str]


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


def _fade_piece(
    waveform: np.ndarray,
    fade_samples: int,
    fade_in: bool,
    fade_out: bool,
) -> np.ndarray:
    """Apply edge fades without changing a piece's sample count."""
    if waveform.shape[-1] == 0 or fade_samples <= 0:
        return waveform
    result = waveform.copy()
    length = min(fade_samples, result.shape[-1] // 2)
    if length == 0:
        return result
    if fade_in:
        result[:, :length] *= np.linspace(0.0, 1.0, length, dtype=np.float32)
    if fade_out:
        result[:, -length:] *= np.linspace(1.0, 0.0, length, dtype=np.float32)
    return result


def _splice_waveforms(
    source: np.ndarray,
    sample_rate: int,
    requests: Sequence[ReplacementRequest],
    generated: Sequence[np.ndarray],
    fade_duration: float = 0.01,
) -> np.ndarray:
    """Cut on the original sample timeline and concatenate replacement clips."""
    pieces: list[np.ndarray] = []
    cursor = 0
    for request, replacement in zip(requests, generated):
        start_sample = round(request.start * sample_rate)
        end_sample = round(request.end * sample_rate)
        pieces.append(source[:, cursor:start_sample])
        pieces.append(replacement)
        cursor = end_sample
    pieces.append(source[:, cursor:])

    fade_samples = round(fade_duration * sample_rate)
    faded = [
        _fade_piece(
            piece,
            fade_samples,
            fade_in=index > 0,
            fade_out=index < len(pieces) - 1,
        )
        for index, piece in enumerate(pieces)
        if piece.shape[-1] > 0
    ]
    if not faded:
        raise ValueError("Regenerated audio contains no samples")
    return np.concatenate(faded, axis=-1)


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
        display_texts: list[str] = []
        for index, request in enumerate(requests, 1):
            segment_path = temp_dir / f"replacement-{index}.wav"
            segment_generator(request.text, str(segment_path))
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
            display_text = subtitle_cleaner(request.text).strip()
            if not display_text:
                raise ValueError(f"Replacement #{index} has no display subtitle text")
            display_texts.append(display_text)

        final_waveform = _splice_waveforms(
            source_waveform,
            source_sample_rate,
            requests,
            generated_waveforms,
        )
        subtitles = rebuild_subtitle_timeline(
            source_subtitles,
            requests,
            generated_durations,
            display_texts,
            subtitle_splitter,
            options.max_char_len,
        )

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
            )
        )
        _publish_artifacts(artifacts)

    return RegenerationResult(
        sample_rate=source_sample_rate,
        total_seconds=total_seconds,
        subtitles=tuple(subtitles),
    )


__all__ = [
    "RegenerationOptions",
    "RegenerationResult",
    "ReplacementRequest",
    "SubtitleItem",
    "load_subtitles",
    "parse_cli_timestamp",
    "parse_replacements",
    "rebuild_subtitle_timeline",
    "regenerate_audio_segments",
    "resolve_cli_mode",
    "validate_regeneration_inputs",
    "validate_regeneration_paths",
]
