#!/usr/bin/env python3
"""Advanced inference CLI for OmniVoice with text markup support.

Provides index-tts compatible features on top of OmniVoice:

Text markup syntax:
    [pause:X]             Insert X seconds of silence (e.g. [pause:1.5])
    [汉字|拼音]           Override pronunciation (synthesis uses pinyin)
    [replace:原文|替换词]  Substitute text for synthesis, keep original for subtitles
    [connect:文字]        Join words to prevent inter-word pauses

Subtitle output:
    --srt             Generate SRT subtitle file
    --json_subtitle   Generate JSON subtitle file with audio metadata
    --max_char_len    Maximum character length per subtitle line

Usage:
    # Basic voice cloning with markup
    omnivoice-infer-advanced \\
        --model k2-fsa/OmniVoice \\
        -t "大家好，[pause:1] 我们开始上课。" \\
        -v speaker.wav \\
        -o output.wav \\
        --srt output.srt \\
        --max_char_len 40

    # With speed control
    omnivoice-infer-advanced \\
        --model k2-fsa/OmniVoice \\
        -t "[connect:学堂在线]使用[connect:雨课堂]" \\
        -v speaker.wav --ref_text "参考音频的文本" \\
        -o output.mp3 --speed 1.2

    # Voice design mode (no reference audio)
    omnivoice-infer-advanced \\
        --model k2-fsa/OmniVoice \\
        -t "Hello world[pause:0.5] How are you?" \\
        --instruct "female, warm tone" \\
        -o output.wav
"""

import argparse
from dataclasses import replace
import json
import logging
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.audio import load_waveform
from omnivoice.utils.audio_segment_regenerator import (
    RegenerationOptions,
    ReplacementRequest,
    SubtitleItem,
    load_subtitles,
    parse_replacements,
    regenerate_audio_segments,
    resolve_cli_mode,
    validate_regeneration_inputs,
    validate_regeneration_paths,
)
from omnivoice.utils.common import get_best_device, str2bool
from omnivoice.utils.connect_candidate_pipeline import (
    ConnectRuntimeOptions,
    FunASRAligner,
    select_connect_waveform,
)
from omnivoice.utils.connect_markup import parse_connect_markup, split_connect_markup
from omnivoice.utils.connect_waveform_processor import ConnectProcessingOptions


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text markup parsing
# ---------------------------------------------------------------------------


def parse_text_actions(text, newline_pause=0.1, markup_version=None):
    """Parse text containing [pause:X] markup into a list of typed actions.

    Splits the input text on ``[pause:X]`` markers and returns an interleaved
    list of ``('text', ...)`` and ``('pause', ...)`` tuples.

    Newline characters (``\n``) are converted to pause markers before parsing.

    Args:
        text: Raw input text, possibly containing ``[pause:X]`` markers and newlines.
        newline_pause: Duration in seconds for pauses inserted at newlines (default 0.3).

    Returns:
        List of tuples: ``('text', str)`` for speech segments and
        ``('pause', float)`` for silence durations.

    Example:
        >>> parse_text_actions("Hello[pause:1]World")
        [('text', 'Hello'), ('pause', 1.0), ('text', 'World')]
        >>> parse_text_actions("Line1\nLine2")
        [('text', 'Line1'), ('pause', 0.3), ('text', 'Line2')]
    """
    is_new = (markup_version == "26071300") or (not markup_version and "\x01" in text)
    # 将换行符转换为 pause 标记
    # 注意: shell 双引号中的 \n 是字面量反斜杠+n，不是真正的换行符
    # 需要同时处理两种情况
    if newline_pause > 0:
        pause_tag = f"\x01[pause:{newline_pause}\x01]" if is_new else f"[pause:{newline_pause}]"
        # 字面量 \n (反斜杠+n) → pause
        text = text.replace("\\n", pause_tag)
        # 真正的换行符 → pause
        text = text.replace("\n", pause_tag)
    else:
        text = text.replace("\\n", " ")
        text = text.replace("\n", " ")

    if is_new:
        pattern = r"\x01\[pause:(\d+(?:\.\d+)?)\x01\]"
    else:
        pattern = r"\[pause:(\d+(?:\.\d+)?)\]"
    parts = re.split(pattern, text)

    actions = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                actions.append(("text", part))
        else:
            actions.append(("pause", float(part)))
    return actions


def clean_text_for_synthesis(text, markup_version=None):
    """Clean markup tags from text, producing TTS-friendly input.

    Strips all custom markup so the text can be fed directly to the model:
    - ``[replace:原文|替换词]`` → ``替换词``
    - ``[汉字|拼音]`` → ``拼音`` (自动转大写以匹配模型训练格式)
    - ``[connect:文字]`` → ``文字``
    - ``\n`` → 空格 (换行符替换为空格)

    Args:
        text: Text with markup tags.

    Returns:
        Cleaned text suitable for speech synthesis.
    """
    # 将换行符替换为空格
    text = text.replace("\n", " ").replace("\r", "")
    is_new = (markup_version == "26071300") or (not markup_version and "\x01" in text)
    if is_new:
        # \x01[replace:原文\x01|替换词\x01] -> 替换词
        text = re.sub(
            r"\x01\[replace:([^\x01]+)\x01\|([^\x01]+)\x01\]", lambda m: f" {m.group(2)} ", text
        )
        # \x01[汉字\x01|拼音\x01] -> 拼音 (转大写)
        text = re.sub(
            r"\x01\[([^\x01]+)\x01\|([^\x01]+)\x01\]", lambda m: f" {m.group(2).upper()} ", text
        )
        # \x01[connect:文字\x01] -> 文字
        text = re.sub(r"\x01\[connect:([^\x01\]]+)\x01\]", r"\1", text)
    else:
        # [replace:原文|替换词] -> 替换词
        text = re.sub(
            r"\[replace:([^|\]]+)\|([^\]]+)\]", lambda m: f" {m.group(2)} ", text
        )
        # [汉字|拼音] -> 拼音 (转大写以匹配模型训练时的格式)
        text = re.sub(r"\[([^|\]]+)\|([^\]]+)\]", lambda m: f" {m.group(2).upper()} ", text)
        # [connect:文字] -> 文字
        text = re.sub(r"\[connect:(.*?)\]", r"\1", text)
    return text


def clean_text_for_subtitles(text, markup_version=None):
    """Clean markup tags from text, producing display-friendly subtitle text.

    Preserves the original (user-visible) text by resolving markup to the
    source form rather than the substitution form:
    - ``[replace:原文|替换词]`` → ``原文``
    - ``[汉字|拼音]`` → ``汉字``
    - ``[connect:文字]`` → ``文字``

    Args:
        text: Text with markup tags.

    Returns:
        Cleaned text suitable for display in subtitles.
    """
    is_new = (markup_version == "26071300") or (not markup_version and "\x01" in text)
    if is_new:
        # \x01[replace:原文\x01|替换词\x01] -> 原文
        text = re.sub(r"\x01\[replace:([^\x01]+)\x01\|[^\x01]+\x01\]", r"\1", text)
        # \x01[汉字\x01|拼音\x01] -> 汉字
        text = re.sub(r"\x01\[([^\x01]+)\x01\|[^\x01]+\x01\]", r"\1", text)
        # \x01[connect:文字\x01] -> 文字
        text = re.sub(r"\x01\[connect:([^\x01\]]+)\x01\]", r"\1", text)
    else:
        # [replace:原文|替换词] -> 原文
        text = re.sub(r"\[replace:([^|\]]+)\|([^\]]+)\]", r"\1", text)
        # [汉字|拼音] -> 汉字
        text = re.sub(r"\[([^|\]]+)\|([^\]]+)\]", r"\1", text)
        # [connect:文字] -> 文字
        text = re.sub(r"\[connect:(.*?)\]", r"\1", text)
    return text


# ---------------------------------------------------------------------------
# SRT time formatting
# ---------------------------------------------------------------------------


def format_seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format ``HH:MM:SS,mmm``.

    Args:
        seconds: Time in seconds (float or int).

    Returns:
        Formatted SRT time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis -= 1000
        secs += 1
    if secs >= 60:
        secs -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(timestamps):
    """Generate SRT subtitle content from a list of timestamped text items.

    Args:
        timestamps: List of dicts with keys ``start`` (float, seconds),
            ``end`` (float, seconds), and ``text`` (str).

    Returns:
        Complete SRT file content as a string.
    """
    srt_lines = []
    for i, item in enumerate(timestamps, 1):
        start_str = format_seconds_to_srt_time(item["start"])
        end_str = format_seconds_to_srt_time(item["end"])
        text = item["text"]
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(f"{text}\n")
    return "\n".join(srt_lines)


# ---------------------------------------------------------------------------
# Subtitle splitting
# ---------------------------------------------------------------------------


def split_by_max_length(text, max_len):
    """Split text into chunks no longer than ``max_len`` characters.

    Uses jieba word segmentation when available for natural break points.
    Falls back to space-splitting (for English) or character splitting.

    Args:
        text: Input text to split.
        max_len: Maximum character length per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_len:
        return [text]

    try:
        import jieba

        jieba.setLogLevel(20)
        words = []
        raw_tokens = jieba.lcut(text)
        # Merge sequential English/numeric/dot/hyphen tokens to protect
        # filenames, decimals, etc.
        temp_eng = ""
        for token in raw_tokens:
            if token.strip() and all(
                c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
                for c in token
            ):
                temp_eng += token
            else:
                if temp_eng:
                    words.append(temp_eng)
                    temp_eng = ""
                words.append(token)
        if temp_eng:
            words.append(temp_eng)
    except ImportError:
        if " " in text:
            words = text.split(" ")
            new_words = []
            for i, w in enumerate(words):
                if i > 0:
                    new_words.append(" ")
                new_words.append(w)
            words = new_words
        else:
            words = list(text)

    sub_texts = []
    current_line = ""
    for word in words:
        if not word:
            continue
        if len(current_line) + len(word) <= max_len:
            current_line += word
        else:
            if current_line:
                sub_texts.append(current_line)
            if len(word) > max_len:
                for i in range(0, len(word), max_len):
                    sub_texts.append(word[i : i + max_len])
                current_line = ""
            else:
                current_line = word
    if current_line:
        sub_texts.append(current_line)

    return [line.strip() for line in sub_texts if line.strip()]


def split_subtitle_item(item, max_char_len=80):
    """Split a subtitle item into shorter segments at punctuation boundaries.

    Splits on Chinese/English punctuation marks (commas, periods, colons,
    semicolons, exclamation/question marks). Then applies max_char_len
    splitting to any remaining long segments. Time is distributed
    proportionally by character count.

    Only splits on English ``.`` and ``:`` when followed by whitespace or
    end-of-string, to avoid breaking file names (``gpt.pth``) or decimals
    (``1.5``).

    Args:
        item: Dict with ``start``, ``end``, and ``text`` keys.
        max_char_len: Maximum character length per output segment.

    Returns:
        List of subtitle item dicts with adjusted timing.
    """
    text = item["text"]
    start = item["start"]
    end = item["end"]
    duration = end - start

    # Split on punctuation; English . and : only when followed by whitespace/EOL
    parts = re.split(
        r"([,，。：；!?\n！？;]|\.(?:\s|$)|:(?:\s|$))", text
    )

    sub_texts = []
    current_part = ""
    for part in parts:
        if not part:
            continue
        is_punc = part in [
            ",",
            "，",
            "。",
            "：",
            "；",
            "!",
            "?",
            "\n",
            "！",
            "？",
            ";",
        ] or part.startswith(".") or part.startswith(":")
        if is_punc:
            current_part += part
            sub_texts.append(current_part.strip())
            current_part = ""
        else:
            if current_part:
                sub_texts.append(current_part.strip())
            current_part = part
    if current_part:
        sub_texts.append(current_part.strip())

    sub_texts = [t for t in sub_texts if t]

    # Apply maximum character length restriction to each segment
    final_sub_texts = []
    for t in sub_texts:
        if len(t) > max_char_len:
            final_sub_texts.extend(split_by_max_length(t, max_char_len))
        else:
            final_sub_texts.append(t)

    if len(final_sub_texts) <= 1:
        if final_sub_texts:
            item["text"] = final_sub_texts[0]
        return [item]

    lengths = [len(t) for t in final_sub_texts]
    total_length = sum(lengths)

    if total_length == 0:
        return [item]

    sub_items = []
    accumulated_time = start
    for t, length in zip(final_sub_texts, lengths):
        part_duration = duration * (length / total_length)
        sub_items.append(
            {
                "start": round(accumulated_time, 3),
                "end": round(accumulated_time + part_duration, 3),
                "text": t,
            }
        )
        accumulated_time += part_duration

    return sub_items


def split_all_timestamps(timestamps, max_char_len=80):
    """Apply subtitle splitting to all timestamp entries.

    Args:
        timestamps: List of subtitle item dicts.
        max_char_len: Maximum character length per output segment.

    Returns:
        New list with long segments split into shorter ones.
    """
    new_timestamps = []
    for item in timestamps:
        new_timestamps.extend(split_subtitle_item(item, max_char_len=max_char_len))
    return new_timestamps


# ---------------------------------------------------------------------------
# Audio concatenation with segment-by-segment synthesis
# ---------------------------------------------------------------------------


def concatenate_audio_actions(
    actions,
    model,
    voice,
    ref_text,
    instruct,
    language,
    speed,
    max_char_len,
    num_step,
    guidance_scale,
    denoise,
    t_shift,
    layer_penalty_factor,
    position_temperature,
    class_temperature,
    markup_version=None,
    connect_options=None,
):
    """Synthesize text segment by segment, inserting silence for pauses.

    Processes a list of actions (text segments and pauses) by calling the
    model for each text segment, inserting silence arrays for pauses, and
    concatenating the results. Disables postprocessing on individual
    segments to prevent unwanted silence padding between concatenated parts.

    Args:
        actions: List of ``('text', str)`` and ``('pause', float)`` tuples
            from :func:`parse_text_actions`.
        model: Loaded OmniVoice model instance.
        voice: Path to speaker reference audio, or None.
        ref_text: Transcript of reference audio, or None.
        instruct: Voice design instruction, or None.
        language: Language name or code, or None.
        speed: Speed factor (applied as post-processing, not during synthesis).
        max_char_len: Max characters per subtitle line.
        num_step: Number of decoding steps.
        guidance_scale: CFG scale.
        denoise: Whether to use denoise token.
        t_shift: Time shift parameter.
        layer_penalty_factor: Layer penalty factor.
        position_temperature: Position temperature.
        class_temperature: Class temperature.

    Returns:
        Tuple of ``(audio, sample_rate, timestamps)`` where ``audio`` is a
        1-D numpy array, ``sample_rate`` is an int, and ``timestamps`` is a
        list of subtitle dicts.
    """
    sample_rate = model.sampling_rate
    expanded_actions = []
    for action_type, value in actions:
        if action_type != "text":
            expanded_actions.append((action_type, value))
            continue
        markup = parse_connect_markup(value, markup_version)
        if markup.ranges:
            expanded_actions.extend(("connect_text", item) for item in split_connect_markup(markup, max_char_len))
        elif "[" in value:
            forced_parts = split_connect_markup(markup, max_char_len)
            if len(forced_parts) > 1:
                expanded_actions.extend(("forced_text", item) for item in forced_parts)
            else:
                expanded_actions.append((action_type, value))
        elif "[" not in value:
            plain_parts = split_subtitle_item(
                {"start": 0.0, "end": 1.0, "text": value},
                max_char_len=max_char_len,
            )
            if len(plain_parts) > 1:
                expanded_actions.extend(
                    ("plain_text", item["text"])
                    for item in plain_parts
                )
            else:
                expanded_actions.append((action_type, value))
        else:
            expanded_actions.append((action_type, value))
    audio_chunks = []
    current_time = 0.0
    all_timestamps = []

    for idx, action in enumerate(expanded_actions):
        action_type, val = action

        if action_type in {"text", "plain_text", "forced_text", "connect_text"}:
            markup = val if action_type in {"forced_text", "connect_text"} else parse_connect_markup(val, markup_version)
            synth_text = markup.synthesis_text
            sub_text = markup.subtitle_text
            if (
                connect_options is not None
                and connect_options.max_forced_segment_tokens is not None
                and len(synth_text) > connect_options.max_forced_segment_tokens
            ):
                raise ValueError(
                    "forced segment exceeds configured token limit: "
                    f"{len(synth_text)} > {connect_options.max_forced_segment_tokens}"
                )

            logger.info(f"Synthesizing segment {idx}: {synth_text[:60]}...")

            # Generate audio for this segment (no postprocessing to avoid
            # padding that would interfere with concatenation)
            generation_kwargs = dict(
                text=synth_text,
                language=language,
                ref_audio=voice,
                ref_text=ref_text,
                instruct=instruct,
                num_step=num_step,
                guidance_scale=guidance_scale,
                denoise=denoise,
                t_shift=t_shift,
                layer_penalty_factor=layer_penalty_factor,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                postprocess_output=False,
                audio_chunk_duration=0.0,
                audio_chunk_threshold=0.0,
                pad_duration=0.0,
                fade_duration=0.0,
            )
            if markup.ranges:
                runtime = connect_options or ConnectRuntimeOptions()
                action_runtime = replace(
                    runtime,
                    speech_speed=speed,
                    debug_dir=(
                        str(Path(runtime.debug_dir) / f"action_{idx + 1:03d}" / "segment_001")
                        if runtime.debug_dir
                        else None
                    ),
                )
                aligner = FunASRAligner(action_runtime.aligner_device, action_runtime.aligner_model)
                segment_audio = select_connect_waveform(
                    markup, sample_rate,
                    lambda: model.generate(**generation_kwargs)[0],
                    aligner, action_runtime,
                    ConnectProcessingOptions(maximum_shorten_ms=action_runtime.maximum_shorten_ms)
                    if action_runtime.processing == "conservative" else None,
                ).waveform
            else:
                segment_audio = model.generate(**generation_kwargs)[0]
            duration = len(segment_audio) / sample_rate

            # A connect candidate's final sample count is its only trustworthy
            # subtitle duration. Do not re-split it by character proportions.
            if sub_text.strip():
                seg_subtitles = (
                    [{"start": 0.0, "end": duration, "text": sub_text}]
                    if markup.ranges or action_type in {"plain_text", "forced_text"}
                    else split_subtitle_item(
                        {"start": 0.0, "end": duration, "text": sub_text},
                        max_char_len=max_char_len,
                    )
                )
                for sub in seg_subtitles:
                    all_timestamps.append(
                        {
                            "start": round(current_time + sub["start"], 3),
                            "end": round(current_time + sub["end"], 3),
                            "text": sub["text"],
                        }
                    )

            audio_chunks.append(segment_audio)
            current_time += duration

        elif action_type == "pause":
            pause_duration = val
            num_samples = int(sample_rate * pause_duration)
            silence = np.zeros(num_samples, dtype=np.float32)
            audio_chunks.append(silence)
            current_time += pause_duration

    # Concatenate all audio chunks
    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
    else:
        final_audio = np.zeros(sample_rate, dtype=np.float32)

    # Apply post-processing speed change (librosa time-stretch preserves pitch)
    if speed != 1.0:
        original_frames = len(final_audio)
        final_audio = change_audio_speed(final_audio, speed)
        if all_timestamps:
            actual_scale = len(final_audio) / original_frames
            for sub in all_timestamps:
                sub["start"] = round(sub["start"] * actual_scale, 3)
                sub["end"] = round(sub["end"] * actual_scale, 3)

    return final_audio, sample_rate, all_timestamps


def change_audio_speed(audio, speed):
    """Adjust audio playback speed using librosa time-stretch.

    Changes speed without altering pitch. Timestamps should be adjusted
    separately by dividing by the speed factor.

    Args:
        audio: 1-D numpy float32 array.
        speed: Speed factor (>1 = faster, <1 = slower).

    Returns:
        Speed-adjusted 1-D numpy array.
    """
    if speed == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed)


# ---------------------------------------------------------------------------
# Audio saving
# ---------------------------------------------------------------------------


def save_audio(audio, sample_rate, output_path):
    """Save audio and return the actual published path.

    Supports WAV, FLAC, OGG (via soundfile) and MP3 (via pydub + ffmpeg).

    Args:
        audio: 1-D numpy float32 array.
        sample_rate: Audio sample rate in Hz.
        output_path: Output file path. Format determined by extension.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp3":
        try:
            from pydub import AudioSegment

            audio_int = (audio * 32768).clip(-32768, 32767).astype(np.int16)
            segment = AudioSegment(
                data=audio_int.tobytes(),
                sample_width=2,
                frame_rate=sample_rate,
                channels=1,
            )
            segment.export(output_path, format="mp3")
        except ImportError:
            logger.warning(
                "pydub not available for MP3 export, falling back to WAV format"
            )
            output_path = os.path.splitext(output_path)[0] + ".wav"
            sf.write(output_path, audio, sample_rate)
    else:
        sf.write(output_path, audio, sample_rate)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for the advanced inference CLI."""
    parser = argparse.ArgumentParser(
        description="OmniVoice markup inference and audio segment regeneration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core arguments ---
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="Text to synthesize (supports [pause], [replace], [connect] markup)",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default=None,
        help="Path to speaker reference audio for voice cloning (.wav)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output audio path (.wav, .flac, .ogg, .mp3)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Audio playback speed factor (e.g. 1.2 for faster, 0.8 for slower)",
    )
    parser.add_argument(
        "--source_audio",
        type=str,
        default=None,
        help="Read-only source audio for subtitle-aligned segment regeneration",
    )
    parser.add_argument(
        "--source_subtitle",
        type=str,
        default=None,
        help="Read-only source SRT or JSON subtitle for segment regeneration",
    )
    parser.add_argument(
        "--regenerate_segments",
        type=str,
        default=None,
        help="JSON array of [HH:MM:SS.mmm start, end, replacement text] items",
    )

    # --- Voice mode ---
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Transcript of the reference audio (required for voice cloning "
        "unless ASR model is available)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Style instruction for voice design (e.g. 'female, warm, British')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language name (e.g. 'Chinese') or code (e.g. 'zh')",
    )

    # --- Model ---
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (auto-detected if not specified)",
    )

    # --- Generation parameters ---
    parser.add_argument(
        "--num_step", type=int, default=32, help="Number of decoding steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--denoise", type=str2bool, default=True, help="Use denoise token"
    )
    parser.add_argument("--t_shift", type=float, default=0.1, help="Time shift")
    parser.add_argument(
        "--layer_penalty_factor",
        type=float,
        default=5.0,
        help="Layer penalty factor",
    )
    parser.add_argument(
        "--position_temperature",
        type=float,
        default=5.0,
        help="Position selection temperature",
    )
    parser.add_argument(
        "--class_temperature",
        type=float,
        default=0.0,
        help="Class token sampling temperature (0 = greedy)",
    )

    # --- Subtitle arguments ---
    parser.add_argument(
        "--srt", type=str, default=None, help="Output path for SRT subtitle file"
    )
    parser.add_argument(
        "--json_subtitle",
        type=str,
        default=None,
        help="Output path for JSON subtitle file",
    )
    parser.add_argument(
        "--max_char_len",
        type=int,
        default=80,
        help="Max character length per subtitle line (40 for shorter lines)",
    )
    parser.add_argument(
        "--markup_version",
        type=str,
        default=None,
        help="Markup format version (e.g. '26071300')",
    )
    parser.add_argument("--connect_candidates", type=int, default=3, help="Connect candidate count (1-5)")
    parser.add_argument("--connect_processing", choices=("conservative", "off"), default="conservative", help="Connect waveform processing mode")
    parser.add_argument("--connect_max_shorten_ms", type=float, default=120.0, help="Maximum shortening per connect boundary")
    parser.add_argument("--connect_aligner_device", default="cpu", help="FunASR fa-zh device")
    parser.add_argument("--connect_debug_dir", default=None, help="Persistent connect candidate evidence directory")
    parser.add_argument("--connect_aligner_model", default="fa-zh", help="FunASR forced-alignment model")
    parser.add_argument("--connect_max_gap_ms", type=float, default=None, help="Optional maximum aligned connect gap")
    parser.add_argument("--connect_seed", type=int, default=None, help="Optional base seed for reproducible connect candidates")
    parser.add_argument("--max_forced_segment_tokens", type=int, default=None, help="Optional maximum cleaned-token length per forced segment")
    parser.add_argument("--boundary_silence", type=float, default=0.3, help="Silence added to both final output boundaries")

    return parser


def _prepare_regeneration(
    args: argparse.Namespace,
) -> tuple[RegenerationOptions, list[ReplacementRequest]]:
    """Parse and statically validate regeneration inputs before model loading."""
    options = RegenerationOptions(
        source_audio=args.source_audio,
        source_subtitle=args.source_subtitle,
        output_audio=args.output,
        output_srt=args.srt,
        output_json_subtitle=args.json_subtitle,
        max_char_len=args.max_char_len,
    )
    requests = parse_replacements(args.regenerate_segments)
    validate_regeneration_paths(options)
    source_waveform, source_sample_rate = load_waveform(options.source_audio)
    source_duration = source_waveform.shape[-1] / source_sample_rate
    subtitles = load_subtitles(options.source_subtitle)
    validate_regeneration_inputs(subtitles, requests, source_duration)
    return options, requests


def _split_replacement_subtitles(
    text: str, duration: float, max_char_len: int
) -> list[SubtitleItem]:
    """Adapt the existing advanced CLI subtitle splitter to normalized items."""
    markup = parse_connect_markup(text)
    if markup.ranges:
        return [SubtitleItem(0.0, duration, markup.subtitle_text)]
    items = split_subtitle_item(
        {"start": 0.0, "end": duration, "text": text},
        max_char_len=max_char_len,
    )
    return [
        SubtitleItem(item["start"], item["end"], item["text"]) for item in items
    ]


def _run_regeneration(
    args: argparse.Namespace,
    model: OmniVoice,
    options: RegenerationOptions,
    requests: list[ReplacementRequest],
) -> None:
    """Regenerate requested clips with the loaded OmniVoice model."""
    replacement_index = 0

    def generate_replacement_segment(text: str, output_path: str) -> None:
        """Synthesize one independent replacement clip through the markup pipeline."""
        nonlocal replacement_index
        replacement_index += 1
        actions = parse_text_actions(text, markup_version=args.markup_version)
        audio, sample_rate, _ = concatenate_audio_actions(
            actions=actions,
            model=model,
            voice=args.voice,
            ref_text=args.ref_text,
            instruct=args.instruct,
            language=args.language,
            speed=args.speed,
            max_char_len=args.max_char_len,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            denoise=args.denoise,
            t_shift=args.t_shift,
            layer_penalty_factor=args.layer_penalty_factor,
            position_temperature=args.position_temperature,
            class_temperature=args.class_temperature,
            markup_version=args.markup_version,
            connect_options=ConnectRuntimeOptions(
                args.connect_candidates, args.connect_processing,
                args.connect_aligner_device, args.connect_max_shorten_ms,
                str(Path(args.connect_debug_dir) / f"replacement_{replacement_index:03d}")
                if args.connect_debug_dir else None,
                args.connect_aligner_model, args.connect_max_gap_ms, args.speed, args.connect_seed, args.max_forced_segment_tokens,
            ),
        )
        sf.write(output_path, audio, sample_rate)

    def clean_replacement_subtitle(text: str) -> str:
        """Resolve markup to the user-visible replacement subtitle text."""
        actions = parse_text_actions(text, markup_version=args.markup_version)
        return "".join(
            parse_connect_markup(value, args.markup_version).subtitle_text
            for action_type, value in actions
            if action_type == "text"
        )

    result = regenerate_audio_segments(
        options=options,
        requests=requests,
        segment_generator=generate_replacement_segment,
        subtitle_cleaner=clean_replacement_subtitle,
        subtitle_splitter=_split_replacement_subtitles,
    )
    logger.info(
        "Segment regeneration completed: %.3fs, %d subtitle item(s)",
        result.total_seconds,
        len(result.subtitles),
    )


def _validate_connect_debug_path(
    debug_dir: str,
    paths: tuple[str | None, ...],
) -> None:
    """Reject debug directories that collide with formal inputs or outputs."""
    debug_path = Path(debug_dir).resolve()
    if debug_path.exists():
        raise ValueError(f"Connect debug directory already exists: {debug_dir}")
    resolved_paths = {
        Path(value).resolve()
        for value in paths
        if value
    }
    if debug_path in resolved_paths:
        raise ValueError("Connect debug directory must not match an input or output path")


def _resolve_model_token_limit(model: OmniVoice) -> int | None:
    """Read a trustworthy text-token limit exposed by the loaded model."""
    candidates = (
        getattr(model, "tokenizer", None),
        getattr(model, "text_tokenizer", None),
        getattr(model, "processor", None),
    )
    for candidate in candidates:
        config = getattr(candidate, "config", candidate)
        value = getattr(config, "model_max_length", None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
    return None


def main():
    """Entry point for the advanced inference CLI."""
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    parser = get_parser()
    args = parser.parse_args()
    if args.connect_debug_dir is None:
        output_path = Path(args.output)
        args.connect_debug_dir = str(
            output_path.parent / f"{output_path.stem}_connect_debug"
        )
    try:
        _validate_connect_debug_path(
            args.connect_debug_dir,
            (
                args.output,
                args.srt,
                args.json_subtitle,
                args.source_audio,
                args.source_subtitle,
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        mode = resolve_cli_mode(
            args.text,
            args.source_audio,
            args.source_subtitle,
            args.regenerate_segments,
            args.srt,
            args.json_subtitle,
        )
        regeneration_inputs = (
            _prepare_regeneration(args) if mode == "regeneration" else None
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    device = args.device or get_best_device()
    logger.info(f"Loading OmniVoice from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )
    model_token_limit = _resolve_model_token_limit(model)
    if args.max_forced_segment_tokens is None and model_token_limit is not None:
        args.max_forced_segment_tokens = model_token_limit

    # Auto-load ASR if voice cloning is requested without ref_text
    if args.voice and not args.ref_text:
        try:
            model.load_asr_model()
            logger.info("ASR model loaded for auto-transcription of reference audio")
        except Exception as e:
            logger.error(
                f"Failed to load ASR model: {e}\n"
                "Please provide --ref_text for voice cloning."
            )
            sys.exit(1)

    operation_name = "segment regeneration" if mode == "regeneration" else "synthesis"
    logger.info("Starting %s...", operation_name)
    logger.info(f"   Speaker Voice: {args.voice or '(none)'}")
    if args.ref_text:
        logger.info(f"   Ref Text: {args.ref_text}")
    if args.instruct:
        logger.info(f"   Voice Design: {args.instruct}")
    if args.language:
        logger.info(f"   Language: {args.language}")

    if mode == "regeneration":
        if regeneration_inputs is None:
            raise RuntimeError("Regeneration inputs were not prepared")
        options, requests = regeneration_inputs
        try:
            _run_regeneration(args, model, options, requests)
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("Segment regeneration failed: %s", error)
            sys.exit(1)
        return

    logger.info(f"   Text: {args.text}")

    # Parse markup into actions
    actions = parse_text_actions(args.text, markup_version=args.markup_version)
    logger.info(f"   Parsed {len(actions)} action(s): {actions}")

    # Synthesize segment by segment
    audio, sample_rate, timestamps = concatenate_audio_actions(
        actions=actions,
        model=model,
        voice=args.voice,
        ref_text=args.ref_text,
        instruct=args.instruct,
        language=args.language,
        speed=args.speed,
        max_char_len=args.max_char_len,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        denoise=args.denoise,
        t_shift=args.t_shift,
        layer_penalty_factor=args.layer_penalty_factor,
        position_temperature=args.position_temperature,
        class_temperature=args.class_temperature,
        markup_version=args.markup_version,
        connect_options=ConnectRuntimeOptions(args.connect_candidates, args.connect_processing, args.connect_aligner_device, args.connect_max_shorten_ms, args.connect_debug_dir, args.connect_aligner_model, args.connect_max_gap_ms, args.speed, args.connect_seed, args.max_forced_segment_tokens),
    )

    if args.boundary_silence < 0:
        parser.error("--boundary_silence must not be negative")
    if args.boundary_silence > 0:
        padding = np.zeros(round(args.boundary_silence * sample_rate), dtype=np.float32)
        audio = np.concatenate((padding, audio, padding))
        for timestamp in timestamps:
            timestamp["start"] = round(timestamp["start"] + args.boundary_silence, 3)
            timestamp["end"] = round(timestamp["end"] + args.boundary_silence, 3)

    # Save audio
    actual_output = save_audio(audio, sample_rate, args.output)
    final_waveform, final_sample_rate = load_waveform(actual_output)
    final_duration = final_waveform.shape[-1] / final_sample_rate
    if timestamps and timestamps[-1]["end"] > final_duration + 0.01:
        parser.error("Final subtitle timeline exceeds encoded audio duration")
    logger.info(f"Audio saved to: {actual_output}")

    # Generate SRT subtitles
    need_subtitles = bool(args.srt or args.json_subtitle)
    if need_subtitles and timestamps:
        logger.info("Generated subtitles:")
        for ts in timestamps:
            logger.info(
                f"   [{ts['start']:.3f}s -> {ts['end']:.3f}s] {ts['text']}"
            )

        if args.srt:
            srt_content = generate_srt(timestamps)
            srt_dir = os.path.dirname(args.srt)
            if srt_dir:
                os.makedirs(srt_dir, exist_ok=True)
            with open(args.srt, "w", encoding="utf-8") as f:
                f.write(srt_content)
            logger.info(f"SRT subtitle saved to: {args.srt}")

        if args.json_subtitle:
            total_seconds = round(final_duration, 2)
            json_data = {
                "audio_file": os.path.basename(actual_output),
                "sample_rate": final_sample_rate,
                "total_seconds": total_seconds,
                "sentences": timestamps,
            }
            json_dir = os.path.dirname(args.json_subtitle)
            if json_dir:
                os.makedirs(json_dir, exist_ok=True)
            with open(args.json_subtitle, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON subtitle saved to: {args.json_subtitle}")


if __name__ == "__main__":
    main()
