"""Pure parsing, validation, and timeline logic for segment regeneration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


BOUNDARY_TOLERANCE_SECONDS = 0.01
CLI_TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
SRT_TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


@dataclass(frozen=True)
class SubtitleItem:
    """Represent one normalized subtitle item in seconds."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ReplacementRequest:
    """Describe one subtitle-aligned range to synthesize again."""

    start: float
    end: float
    text: str | list[str]
    speech_speed: float = 1.0


SubtitleSplitter = Callable[[str, float, int], list[SubtitleItem]]


def parse_cli_timestamp(value: str) -> float:
    """Parse a strict ``HH:MM:SS.mmm`` timestamp into seconds."""
    match = CLI_TIMESTAMP_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Timestamp must use HH:MM:SS.mmm format: {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Timestamp contains an invalid minute or second: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _parse_srt_timestamp(value: str) -> float:
    """Parse an SRT timestamp using comma or dot milliseconds."""
    match = SRT_TIMESTAMP_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_replacements(
    value: str,
    legacy_speed: float = 1.0,
) -> list[ReplacementRequest]:
    """Parse IndexTTS replacement quadruples with legacy triple support."""
    if not math.isfinite(legacy_speed) or legacy_speed <= 0:
        raise ValueError("Legacy replacement speed must be positive and finite")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Replacement JSON is invalid: {error.msg}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("Replacement array must contain at least one item")

    requests: list[ReplacementRequest] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, list) or len(item) not in (3, 4):
            raise ValueError(
                f"Replacement #{index} must contain three legacy or four IndexTTS values"
            )
        start_value, end_value, text_value = item[:3]
        if not isinstance(start_value, str) or not isinstance(end_value, str):
            raise ValueError(f"Replacement #{index} start and end must be strings")
        if isinstance(text_value, str):
            text: str | list[str] = text_value.strip()
            if not text:
                raise ValueError(f"Replacement #{index} text must not be blank")
        elif isinstance(text_value, list) and text_value and all(
            isinstance(part, str) and part.strip() for part in text_value
        ):
            text = [part.strip() for part in text_value]
        else:
            raise ValueError(f"Replacement #{index} text must be a string or non-empty string array")
        speech_speed = legacy_speed
        if len(item) == 4:
            raw_speed = item[3]
            if isinstance(raw_speed, bool) or not isinstance(raw_speed, (int, float)):
                raise ValueError(f"Replacement #{index} speechSpeed must be numeric")
            speech_speed = float(raw_speed)
            if not math.isfinite(speech_speed) or speech_speed <= 0:
                raise ValueError(f"Replacement #{index} speechSpeed must be positive and finite")
        requests.append(
            ReplacementRequest(
                start=parse_cli_timestamp(start_value),
                end=parse_cli_timestamp(end_value),
                text=text,
                speech_speed=speech_speed,
            )
        )
    return requests


def _covered_subtitle_indexes(
    subtitles: Sequence[SubtitleItem],
    request: ReplacementRequest,
) -> list[int]:
    """Return source subtitle indexes touched by a replacement range."""
    return [
        index
        for index, item in enumerate(subtitles)
        if item.end > request.start + BOUNDARY_TOLERANCE_SECONDS
        and item.start < request.end - BOUNDARY_TOLERANCE_SECONDS
    ]


def _text_items_for_request(
    subtitles: Sequence[SubtitleItem],
    request: ReplacementRequest,
) -> list[str]:
    """Normalize one request's text to one display item per covered subtitle."""
    covered = _covered_subtitle_indexes(subtitles, request)
    if not covered:
        raise ValueError("Replacement request does not cover any subtitle")
    if isinstance(request.text, list):
        if len(request.text) != len(covered):
            raise ValueError("Replacement text array must match covered subtitle count")
        return request.text
    if len(covered) == 1:
        return [request.text]
    pieces = [
        part.strip()
        for part in re.findall(r".*?(?:[。！？!?；;]|$)", request.text, re.S)
        if part.strip()
    ]
    if len(pieces) != len(covered):
        raise ValueError("Multi-subtitle replacement text must provide one sentence per subtitle")
    return pieces


def normalize_replacement_requests(
    subtitles: Sequence[SubtitleItem],
    requests: Sequence[ReplacementRequest],
) -> list[ReplacementRequest]:
    """Merge adjacent complete requests with identical per-request speed."""
    validate_subtitles(subtitles)
    if not requests:
        raise ValueError("Replacement array must contain at least one item")

    normalized: list[ReplacementRequest] = []
    for request in requests:
        text_items = _text_items_for_request(subtitles, request)
        current = ReplacementRequest(
            request.start,
            request.end,
            text_items if len(text_items) > 1 else text_items[0],
            request.speech_speed,
        )
        if normalized:
            previous = normalized[-1]
            previous_indexes = _covered_subtitle_indexes(subtitles, previous)
            current_indexes = _covered_subtitle_indexes(subtitles, current)
            if (
                previous_indexes
                and current_indexes
                and previous_indexes[-1] + 1 == current_indexes[0]
                and math.isclose(previous.end, current.start, abs_tol=BOUNDARY_TOLERANCE_SECONDS)
            ):
                if not math.isclose(previous.speech_speed, current.speech_speed):
                    raise ValueError("Adjacent replacement requests must use the same speech speed")
                previous_items = _text_items_for_request(subtitles, previous)
                merged_items = previous_items + text_items
                normalized[-1] = ReplacementRequest(
                    previous.start,
                    current.end,
                    merged_items,
                    previous.speech_speed,
                )
                continue
        normalized.append(current)
    return normalized


def load_subtitles(path: str) -> list[SubtitleItem]:
    """Load and normalize subtitle items from an SRT or JSON file."""
    subtitle_path = Path(path)
    suffix = subtitle_path.suffix.lower()
    if suffix == ".srt":
        items = _load_srt(subtitle_path)
    elif suffix == ".json":
        items = _load_json_subtitles(subtitle_path)
    else:
        raise ValueError(f"Unsupported subtitle format: {subtitle_path.suffix}")
    validate_subtitles(items)
    return items


def _load_srt(path: Path) -> list[SubtitleItem]:
    """Load subtitle items from an SRT file."""
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", content.strip()) if content.strip() else []
    items: list[SubtitleItem] = []
    for block_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        if len(lines) < 2 or "-->" not in lines[0]:
            raise ValueError(f"Invalid SRT block #{block_index}")
        start_value, end_value = (part.strip() for part in lines[0].split("-->", 1))
        text = "\n".join(lines[1:]).strip()
        items.append(
            SubtitleItem(
                start=_parse_srt_timestamp(start_value),
                end=_parse_srt_timestamp(end_value),
                text=text,
            )
        )
    return items


def _load_json_subtitles(path: Path) -> list[SubtitleItem]:
    """Load subtitle items from the standard JSON sentences structure."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Subtitle JSON is invalid: {error.msg}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("sentences"), list):
        raise ValueError("Subtitle JSON must contain a sentences array")
    items: list[SubtitleItem] = []
    for index, item in enumerate(payload["sentences"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"Subtitle #{index} must be an object")
        start = item.get("start")
        end = item.get("end")
        text = item.get("text")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            raise ValueError(f"Subtitle #{index} start must be numeric")
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            raise ValueError(f"Subtitle #{index} end must be numeric")
        if not isinstance(text, str):
            raise ValueError(f"Subtitle #{index} text must be a string")
        items.append(SubtitleItem(float(start), float(end), text.strip()))
    return items


def validate_subtitles(subtitles: Sequence[SubtitleItem]) -> None:
    """Validate a normalized subtitle timeline."""
    if not subtitles:
        raise ValueError("Subtitle timeline must contain at least one item")
    previous_end = 0.0
    for index, item in enumerate(subtitles, 1):
        if not math.isfinite(item.start) or not math.isfinite(item.end):
            raise ValueError(f"Subtitle #{index} times must be finite")
        if item.start < 0 or item.start >= item.end:
            raise ValueError(f"Subtitle #{index} must satisfy 0 <= start < end")
        if not item.text:
            raise ValueError(f"Subtitle #{index} text must not be blank")
        if index > 1 and item.start < previous_end:
            raise ValueError(f"Subtitle #{index} overlaps the previous subtitle")
        previous_end = item.end


def validate_regeneration_inputs(
    subtitles: Sequence[SubtitleItem],
    requests: Sequence[ReplacementRequest],
    source_duration: float,
    tolerance: float = BOUNDARY_TOLERANCE_SECONDS,
) -> None:
    """Validate replacement ordering, ranges, and full subtitle boundaries."""
    validate_subtitles(subtitles)
    if not requests:
        raise ValueError("Replacement array must contain at least one item")
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("Source audio duration must be a positive finite value")

    previous_end = -1.0
    for index, request in enumerate(requests, 1):
        if not math.isfinite(request.start) or not math.isfinite(request.end):
            raise ValueError(f"Replacement #{index} times must be finite")
        if request.start < 0 or request.start >= request.end:
            raise ValueError(f"Replacement #{index} must satisfy 0 <= start < end")
        if isinstance(request.text, str):
            has_text = bool(request.text.strip())
        else:
            has_text = bool(request.text) and all(item.strip() for item in request.text)
        if not has_text:
            raise ValueError(f"Replacement #{index} text must not be blank")
        if not math.isfinite(request.speech_speed) or request.speech_speed <= 0:
            raise ValueError(f"Replacement #{index} speech speed must be positive and finite")
        if request.end > source_duration + tolerance:
            raise ValueError(f"Replacement #{index} exceeds source audio duration")
        if request.start < previous_end:
            raise ValueError(f"Replacement #{index} overlaps replacement #{index - 1}")
        previous_end = request.end

        covered = [
            item
            for item in subtitles
            if item.end > request.start + tolerance
            and item.start < request.end - tolerance
        ]
        aligned = (
            covered
            and abs(covered[0].start - request.start) <= tolerance
            and abs(covered[-1].end - request.end) <= tolerance
            and all(
                item.start >= request.start - tolerance
                and item.end <= request.end + tolerance
                for item in covered
            )
        )
        if not aligned:
            raise ValueError(
                f"Replacement #{index} must align with complete subtitle boundaries"
            )


def rebuild_subtitle_timeline(
    subtitles: Sequence[SubtitleItem],
    requests: Sequence[ReplacementRequest],
    generated_durations: Sequence[float],
    display_texts: Sequence[str],
    subtitle_splitter: SubtitleSplitter,
    max_char_len: int,
    overlap_durations: Sequence[tuple[float, float]] | None = None,
) -> list[SubtitleItem]:
    """Replace covered subtitles and shift later items by cumulative deltas."""
    if not (len(requests) == len(generated_durations) == len(display_texts)):
        raise ValueError("Replacement metadata lengths must match")
    if overlap_durations is None:
        overlap_durations = [(0.0, 0.0)] * len(requests)
    if len(overlap_durations) != len(requests):
        raise ValueError("Replacement overlap metadata lengths must match")

    result: list[SubtitleItem] = []
    subtitle_index = 0
    cumulative_offset = 0.0
    for request, duration, display_text, (left_overlap, right_overlap) in zip(
        requests, generated_durations, display_texts, overlap_durations
    ):
        while (
            subtitle_index < len(subtitles)
            and subtitles[subtitle_index].end <= request.start + BOUNDARY_TOLERANCE_SECONDS
        ):
            item = subtitles[subtitle_index]
            result.append(
                SubtitleItem(
                    item.start + cumulative_offset,
                    item.end + cumulative_offset,
                    item.text,
                )
            )
            subtitle_index += 1

        while (
            subtitle_index < len(subtitles)
            and subtitles[subtitle_index].start < request.end - BOUNDARY_TOLERANCE_SECONDS
        ):
            subtitle_index += 1

        effective_duration = duration - left_overlap - right_overlap
        if effective_duration <= 0:
            raise ValueError("Replacement overlaps consume the generated subtitle window")
        replacement_start = request.start + cumulative_offset
        for item in subtitle_splitter(display_text, effective_duration, max_char_len):
            result.append(
                SubtitleItem(
                    replacement_start + item.start,
                    replacement_start + item.end,
                    item.text,
                )
            )
        cumulative_offset += effective_duration - (request.end - request.start)

    for item in subtitles[subtitle_index:]:
        result.append(
            SubtitleItem(
                item.start + cumulative_offset,
                item.end + cumulative_offset,
                item.text,
            )
        )
    validate_subtitles(result)
    return result


def resolve_cli_mode(
    text: str | None,
    source_audio: str | None,
    source_subtitle: str | None,
    regenerate_segments: str | None,
    srt: str | None,
    json_subtitle: str | None,
) -> str:
    """Resolve normal synthesis or segment regeneration CLI mode."""
    regeneration_values = (source_audio, source_subtitle, regenerate_segments)
    regeneration_requested = any(value is not None for value in regeneration_values)
    if regeneration_requested and text is not None:
        raise ValueError("--text and segment regeneration inputs cannot be used together")
    if regeneration_requested:
        if not all(value is not None for value in regeneration_values):
            raise ValueError(
                "Segment regeneration requires --source_audio, --source_subtitle, "
                "and --regenerate_segments"
            )
        if not (srt or json_subtitle):
            raise ValueError("Segment regeneration requires at least one subtitle output")
        return "regeneration"
    if text is None:
        raise ValueError("Normal synthesis requires --text")
    return "synthesis"
