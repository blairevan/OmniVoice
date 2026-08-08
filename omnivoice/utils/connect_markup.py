"""Parse connect markup while preserving its exact synthesis offsets."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ConnectRange:
    """Represent an inclusive-exclusive connected range in synthesis text."""

    start: int
    end: int


@dataclass(frozen=True)
class ConnectMarkup:
    """Return cleaned text and the ranges eligible for connect processing."""

    synthesis_text: str
    subtitle_text: str
    ranges: tuple[ConnectRange, ...]
    alignment_text: str = ""
    alignment_source_offsets: tuple[int | None, ...] = ()
    raw_text: str = ""

    @property
    def connect_spans(self) -> tuple[ConnectRange, ...]:
        """Expose IndexTTS-compatible forced-segment span naming."""
        return self.ranges


def parse_connect_markup(text: str, markup_version: str | None = None) -> ConnectMarkup:
    """Resolve supported markup and record Chinese connect payload offsets."""
    is_new = markup_version == "26071300" or (markup_version is None and "\x01" in text)
    connect_pattern = (
        re.compile(r"\x01\[connect:([^\x01\]]+)\x01\]")
        if is_new
        else re.compile(r"\[connect:([^\]]+)\]")
    )
    synthesis_parts: list[str] = []
    subtitle_parts: list[str] = []
    ranges: list[ConnectRange] = []
    cursor = 0
    for match in connect_pattern.finditer(text):
        prefix = text[cursor:match.start()]
        synthesis_parts.append(_clean(prefix, is_new, False))
        subtitle_parts.append(_clean(prefix, is_new, True))
        payload = match.group(1)
        if len(payload) < 2 or not all("\u3400" <= char <= "\u9fff" for char in payload):
            raise ValueError("connect must contain at least two Chinese characters")
        start = len("".join(synthesis_parts))
        synthesis_parts.append(payload)
        subtitle_parts.append(payload)
        ranges.append(ConnectRange(start, start + len(payload)))
        cursor = match.end()
    tail = text[cursor:]
    synthesis_parts.append(_clean(tail, is_new, False))
    subtitle_parts.append(_clean(tail, is_new, True))
    prefix = "\x01[connect:" if is_new else "[connect:"
    if text.count(prefix) != len(ranges):
        raise ValueError("Invalid or unclosed connect markup")
    synthesis_text = "".join(synthesis_parts).strip()
    subtitle_text = "".join(subtitle_parts).strip()
    alignment_text, offsets = _alignment_mapping(text, is_new, synthesis_text)
    return ConnectMarkup(synthesis_text, subtitle_text, tuple(ranges), alignment_text, offsets, text)


def connect_boundaries(ranges: tuple[ConnectRange, ...]) -> tuple[tuple[int, int], ...]:
    """Return only adjacent character offsets within individual connect ranges."""
    return tuple((index, index + 1) for item in ranges for index in range(item.start, item.end - 1))


def split_connect_markup(markup: ConnectMarkup, max_char_len: int) -> tuple[ConnectMarkup, ...]:
    """Pre-split connect text without splitting a connect range or timing unit."""
    if max_char_len <= 0:
        raise ValueError("max_char_len must be positive")
    if len(markup.synthesis_text) != len(markup.subtitle_text) and markup.raw_text:
        return _split_raw_markup(markup.raw_text, max_char_len)
    protected = {index for item in markup.ranges for index in range(item.start + 1, item.end)}
    boundaries = [0]
    cursor = 0
    text = markup.synthesis_text
    while cursor < len(text):
        limit = min(len(text), cursor + max_char_len)
        candidates = [index for index in range(cursor + 1, limit + 1) if index not in protected]
        if not candidates:
            raise ValueError("connect range exceeds max_char_len and cannot be split")
        punctuation = [index for index in candidates if text[index - 1] in "，。；！？,.!?;"]
        connect_ends = [item.end for item in markup.ranges if item.end in candidates]
        cursor = punctuation[-1] if punctuation else (connect_ends[-1] if connect_ends else candidates[-1])
        boundaries.append(cursor)
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        ranges = tuple(
            ConnectRange(item.start - start, item.end - start)
            for item in markup.ranges
            if start <= item.start and item.end <= end
        )
        alignment = [
            (char, offset)
            for char, offset in zip(markup.alignment_text, markup.alignment_source_offsets)
            if offset is not None and start <= offset < end
        ]
        result.append(ConnectMarkup(
            text[start:end], markup.subtitle_text[start:end], ranges,
            "".join(item[0] for item in alignment),
            tuple(item[1] - start for item in alignment), "",
        ))
    return tuple(result)


def _split_raw_markup(text: str, max_char_len: int) -> tuple[ConnectMarkup, ...]:
    """Split raw markup only between atomic tags, using display-text length."""
    marker = re.compile(r"\x01\[.*?\x01\]" if "\x01" in text else r"\[(?:replace:[^\]]+|connect:[^\]]+|[^|\]]+\|[^\]]+)\]")
    tokens: list[str] = []; cursor = 0
    for match in marker.finditer(text):
        tokens.extend(text[cursor:match.start()]); tokens.append(match.group(0)); cursor = match.end()
    tokens.extend(text[cursor:])
    groups: list[list[str]] = []; current: list[str] = []; current_length = 0
    boundaries = set("，。；！？,.!?;")
    for token in tokens:
        display_length = len(parse_connect_markup(token).subtitle_text)
        if current and current_length + display_length > max_char_len:
            groups.append(current); current = []; current_length = 0
        current.append(token); current_length += display_length
        if token in boundaries:
            groups.append(current); current = []; current_length = 0
    if current: groups.append(current)
    return tuple(parse_connect_markup("".join(group)) for group in groups)


# Compatibility name used by the IndexTTS forced-segment contract.
ForcedSegment = ConnectMarkup


def _clean(value: str, is_new: bool, subtitle: bool) -> str:
    """Resolve legacy markup outside connect tags for synthesis or display."""
    if is_new:
        if subtitle:
            value = re.sub(r"\x01\[replace:([^\x01]+)\x01\|[^\x01]+\x01\]", r"\1", value)
            value = re.sub(r"\x01\[([^\x01]+)\x01\|[^\x01]+\x01\]", r"\1", value)
        else:
            value = re.sub(r"\x01\[replace:[^\x01]+\x01\|([^\x01]+)\x01\]", r" \1 ", value)
            value = re.sub(r"\x01\[[^\x01]+\x01\|([^\x01]+)\x01\]", r" \1 ", value)
    elif subtitle:
        value = re.sub(r"\[replace:([^|\]]+)\|[^\]]+\]", r"\1", value)
        value = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", value)
    else:
        value = re.sub(r"\[replace:[^|\]]+\|([^\]]+)\]", r" \1 ", value)
        value = re.sub(r"\[[^|\]]+\|([^\]]+)\]", r" \1 ", value)
    return value.replace("\n", " ").replace("\r", "")


def _alignment_mapping(text: str, is_new: bool, synthesis_text: str) -> tuple[str, tuple[int | None, ...]]:
    """Map aligned Chinese text back to synthesis offsets, preserving pinyin overrides."""
    token_pattern = re.compile(r"\x01\[.*?\x01\]" if is_new else r"\[(?:replace:[^\]]+|connect:[^\]]+|[^|\]]+\|[^\]]+)\]")
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for match in token_pattern.finditer(text):
        parts.append((text[cursor:match.start()], False)); parts.append((match.group(0), True)); cursor = match.end()
    parts.append((text[cursor:], False))
    characters: list[str] = []; offsets: list[int | None] = []; synthesis_cursor = 0; synthesis_parts: list[str] = []
    for raw, atomic in parts:
        connect_match = re.fullmatch(r"\x01\[connect:([^\x01\]]+)\x01\]" if is_new else r"\[connect:([^\]]+)\]", raw)
        synth = connect_match.group(1) if connect_match else _clean(raw, is_new, False)
        sub = connect_match.group(1) if connect_match else _clean(raw, is_new, True)
        synthesis_parts.append(synth)
        pronunciation = atomic and "|" in raw and "replace:" not in raw and "connect:" not in raw
        if pronunciation:
            for char in sub:
                if "\u3400" <= char <= "\u9fff": characters.append(char); offsets.append(None)
        else:
            for index, char in enumerate(synth):
                if "\u3400" <= char <= "\u9fff": characters.append(char); offsets.append(synthesis_cursor + index)
        synthesis_cursor += len(synth)
    raw_synthesis = "".join(synthesis_parts)
    leading = len(raw_synthesis) - len(raw_synthesis.lstrip())
    adjusted = tuple(None if item is None else item - leading for item in offsets)
    return "".join(characters), adjusted
