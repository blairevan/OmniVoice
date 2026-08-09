"""Two-phase coherent OmniVoice synthesis orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
import tempfile
import time
from typing import Sequence

import numpy as np
import soundfile as sf

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.coherent_timeline import (
    CoherentSynthesisGroup,
    SubtitleUnit,
    build_coherent_synthesis_groups,
)
from omnivoice.utils.connect_candidate_pipeline import (
    ConnectRuntimeOptions,
    FunASRAligner,
    select_connect_waveform,
)
from omnivoice.utils.connect_markup import ConnectMarkup, ConnectRange, parse_connect_markup, split_connect_markup
from omnivoice.utils.connect_waveform_processor import ConnectProcessingOptions
from omnivoice.utils.sentence_timing import resolve_sentence_timings
from omnivoice.utils.whisperx_alignment import (
    WhisperXAligner,
    WhisperXAlignmentDeadline,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedSynthesisGroup:
    """Store one generated coherent group on the pre-speed sample axis."""

    waveform: np.ndarray
    sample_rate: int
    units: tuple[SubtitleUnit, ...]
    source_action_index: int
    start_sample: int


@dataclass(frozen=True)
class GeneratedSynthesisResult:
    """Store generated audio chunks before subtitle alignment and final speed."""

    waveform: np.ndarray
    sample_rate: int
    groups: tuple[GeneratedSynthesisGroup, ...]


def _unit_from_markup(markup: ConnectMarkup, sentence_id: int) -> SubtitleUnit:
    """Convert one parsed markup item into the normalized text contract."""
    alignment_units = tuple(character for character in markup.alignment_text if not character.isspace())
    return SubtitleUnit(
        sentence_id=sentence_id,
        display_text=markup.subtitle_text,
        synthesis_text=markup.synthesis_text,
        alignment_text=markup.alignment_text,
        alignment_units=alignment_units,
        synthesis_range=(0, len(markup.synthesis_text)),
        alignment_unit_range=(0, len(alignment_units)),
        alignment_source_offsets=tuple(
            offset for character, offset in zip(markup.alignment_text, markup.alignment_source_offsets)
            if not character.isspace()
        ),
        connect_ranges=tuple((item.start, item.end) for item in markup.ranges),
    )


def build_subtitle_units(
    text: str,
    max_char_len: int,
    sentence_id_start: int = 1,
    markup_version: str | None = None,
) -> tuple[SubtitleUnit, ...]:
    """Split one text action into complete normalized subtitle units."""
    markup = parse_connect_markup(text, markup_version)
    if markup.ranges or "[" in text:
        parts = split_connect_markup(markup, max_char_len)
        return tuple(
            _unit_from_markup(part, sentence_id_start + index)
            for index, part in enumerate(parts)
        )
    raw_parts = [
        item["text"]
        for item in _split_plain_subtitles(text, max_char_len)
    ]
    return tuple(
        _unit_from_markup(parse_connect_markup(part, markup_version), sentence_id_start + index)
        for index, part in enumerate(raw_parts)
    )


def _split_plain_subtitles(text: str, max_char_len: int) -> list[dict[str, str]]:
    """Split ordinary text at punctuation while preserving display units."""
    import re

    parts = [part.strip() for part in re.findall(r".*?(?:[,，。：；!?！？;]|$)", text, re.S) if part.strip()]
    result: list[dict[str, str]] = []
    for part in parts or [text.strip()]:
        if len(part) <= max_char_len:
            result.append({"text": part})
            continue
        for offset in range(0, len(part), max_char_len):
            result.append({"text": part[offset : offset + max_char_len]})
    return result


def _to_connect_markup(group: CoherentSynthesisGroup) -> ConnectMarkup:
    """Rebuild connect metadata after coherent units are merged."""
    return ConnectMarkup(
        synthesis_text=group.synthesis_text,
        subtitle_text=group.display_text,
        ranges=tuple(ConnectRange(start, end) for start, end in group.connect_ranges),
        alignment_text=group.alignment_text,
        alignment_source_offsets=group.alignment_source_offsets,
    )


def generate_coherent_actions(
    actions: Sequence[tuple[str, str | float]],
    model: OmniVoice,
    voice: str | None,
    ref_text: str | None,
    instruct: str | None,
    language: str | None,
    max_char_len: int,
    max_tokens: int,
    num_step: int,
    guidance_scale: float,
    denoise: bool,
    t_shift: float,
    layer_penalty_factor: float,
    position_temperature: float,
    class_temperature: float,
    markup_version: str | None,
    connect_options: ConnectRuntimeOptions,
) -> GeneratedSynthesisResult:
    """Generate all coherent groups before any WhisperX alignment."""
    sample_rate = model.sampling_rate
    chunks: list[np.ndarray] = []
    groups: list[GeneratedSynthesisGroup] = []
    current_sample = 0
    sentence_id = 1
    tokenizer = getattr(model, "text_tokenizer", None) or getattr(model, "tokenizer", None)
    for action_index, (action_type, value) in enumerate(actions):
        if action_type == "pause":
            pause_samples = round(float(value) * sample_rate)
            chunks.append(np.zeros(pause_samples, dtype=np.float32))
            current_sample += pause_samples
            continue
        if not isinstance(value, str):
            raise ValueError("Text synthesis actions must contain strings")
        units = build_subtitle_units(value, max_char_len, sentence_id, markup_version)
        sentence_id += len(units)
        coherent_groups = build_coherent_synthesis_groups(
            units,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            max_synthesis_chars=max_char_len * 3,
        )
        for group_index, coherent_group in enumerate(coherent_groups, 1):
            group_started = time.perf_counter()
            generation_kwargs = {
                "text": coherent_group.synthesis_text,
                "language": language,
                "ref_audio": voice,
                "ref_text": ref_text,
                "instruct": instruct,
                "num_step": num_step,
                "guidance_scale": guidance_scale,
                "denoise": denoise,
                "t_shift": t_shift,
                "layer_penalty_factor": layer_penalty_factor,
                "position_temperature": position_temperature,
                "class_temperature": class_temperature,
                "postprocess_output": False,
                "audio_chunk_duration": 0.0,
                "audio_chunk_threshold": 0.0,
                "pad_duration": 0.0,
                "fade_duration": 0.0,
            }
            if coherent_group.connect_ranges:
                markup = _to_connect_markup(coherent_group)
                runtime = connect_options
                if runtime.debug_dir is not None:
                    runtime = replace(
                        runtime,
                        debug_dir=str(
                            Path(runtime.debug_dir)
                            / f"action_{action_index + 1:03d}"
                            / f"group_{group_index:03d}"
                        ),
                    )
                aligner = FunASRAligner(runtime.aligner_device, runtime.aligner_model)
                waveform = select_connect_waveform(
                    markup,
                    sample_rate,
                    lambda: model.generate(**generation_kwargs)[0],
                    aligner,
                    runtime,
                    ConnectProcessingOptions(maximum_shorten_ms=runtime.maximum_shorten_ms)
                    if runtime.processing == "conservative"
                    else None,
                ).waveform
            else:
                waveform = model.generate(**generation_kwargs)[0]
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            groups.append(
                GeneratedSynthesisGroup(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    units=coherent_group.units,
                    source_action_index=action_index,
                    start_sample=current_sample,
                )
            )
            chunks.append(waveform)
            current_sample += waveform.shape[-1]
            logger.info(
                "[timing] coherent group action=%d group=%d connect=%s duration=%.3fs samples=%d",
                action_index + 1,
                group_index,
                bool(coherent_group.connect_ranges),
                time.perf_counter() - group_started,
                waveform.shape[-1],
            )
    audio = np.concatenate(chunks) if chunks else np.zeros(sample_rate, dtype=np.float32)
    return GeneratedSynthesisResult(audio, sample_rate, tuple(groups))


def resolve_generated_group_timings(
    result: GeneratedSynthesisResult,
    aligner: WhisperXAligner | None,
    deadline: WhisperXAlignmentDeadline | None,
) -> list[dict[str, object]]:
    """Resolve every generated group after TTS has been released from GPU."""
    timestamps: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="omnivoice-group-align-") as temp_dir:
        for index, group in enumerate(result.groups, 1):
            if len(group.units) == 1:
                timings = [(0, group.waveform.shape[-1])]
            else:
                resolution = None
                if aligner is not None and deadline is not None:
                    audio_path = Path(temp_dir) / f"group-{index}.wav"
                    sf.write(audio_path, group.waveform, group.sample_rate)
                    try:
                        characters = aligner.align(
                            str(audio_path),
                            "".join(unit.alignment_text for unit in group.units),
                            0.0,
                            group.waveform.shape[-1] / group.sample_rate,
                            deadline,
                        )
                        resolution = resolve_sentence_timings(
                            tuple(unit.alignment_text for unit in group.units),
                            characters,
                            group.sample_rate,
                            group.waveform.shape[-1],
                            0,
                            0,
                        )
                    except Exception:
                        resolution = None
                if resolution is None:
                    resolution = resolve_sentence_timings(
                        tuple(unit.alignment_text for unit in group.units),
                        (),
                        group.sample_rate,
                        group.waveform.shape[-1],
                        0,
                        0,
                    )
                timings = [
                    (item.start_sample, item.end_sample) for item in resolution.timings
                ]
            for unit, (start, end) in zip(group.units, timings):
                timestamps.append(
                    {
                        "start": (group.start_sample + start) / result.sample_rate,
                        "end": (group.start_sample + end) / result.sample_rate,
                        "text": unit.display_text,
                    }
                )
    return timestamps


__all__ = [
    "GeneratedSynthesisGroup",
    "GeneratedSynthesisResult",
    "build_subtitle_units",
    "generate_coherent_actions",
    "resolve_generated_group_timings",
]
