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
import json
import logging
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device, str2bool


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
    audio_chunks = []
    current_time = 0.0
    all_timestamps = []

    for idx, action in enumerate(actions):
        action_type, val = action

        if action_type == "text":
            synth_text = clean_text_for_synthesis(val, markup_version=markup_version)
            sub_text = clean_text_for_subtitles(val, markup_version=markup_version)

            logger.info(f"Synthesizing segment {idx}: {synth_text[:60]}...")

            # Generate audio for this segment (no postprocessing to avoid
            # padding that would interfere with concatenation)
            audios = model.generate(
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
            )

            segment_audio = audios[0]
            duration = len(segment_audio) / sample_rate

            # Build subtitle entries for this segment
            if sub_text.strip():
                seg_subtitles = split_subtitle_item(
                    {"start": 0.0, "end": duration, "text": sub_text},
                    max_char_len=max_char_len,
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
        final_audio = change_audio_speed(final_audio, speed)
        if all_timestamps:
            for sub in all_timestamps:
                sub["start"] = round(sub["start"] / speed, 3)
                sub["end"] = round(sub["end"] / speed, 3)

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
    """Save audio to file, inferring format from extension.

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for the advanced inference CLI."""
    parser = argparse.ArgumentParser(
        description="OmniVoice advanced inference with text markup support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Core arguments ---
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
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

    return parser


def main():
    """Entry point for the advanced inference CLI."""
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logger.info(f"Loading OmniVoice from {args.model} on {device} ...")
    model = OmniVoice.from_pretrained(
        args.model, device_map=device, dtype=torch.float16
    )

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

    logger.info(f"Starting synthesis...")
    logger.info(f"   Text: {args.text}")
    logger.info(f"   Speaker Voice: {args.voice or '(none)'}")
    if args.ref_text:
        logger.info(f"   Ref Text: {args.ref_text}")
    if args.instruct:
        logger.info(f"   Voice Design: {args.instruct}")
    if args.language:
        logger.info(f"   Language: {args.language}")

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
    )

    # Save audio
    save_audio(audio, sample_rate, args.output)
    logger.info(f"Audio saved to: {args.output}")

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
            total_seconds = round(len(audio) / sample_rate, 2)
            json_data = {
                "audio_file": os.path.basename(args.output),
                "sample_rate": sample_rate,
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
