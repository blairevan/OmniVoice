"""Tests for subtitle-aligned audio segment regeneration."""

import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from omnivoice.utils.sentence_timing import SentenceTiming, TimingResolution
from omnivoice.utils.audio_segment_regenerator import (
    RegenerationOptions,
    ReplacementRequest,
    SubtitleItem,
    load_subtitles,
    parse_cli_timestamp,
    parse_replacements,
    normalize_replacement_requests,
    rebuild_subtitle_timeline,
    regenerate_audio_segments,
    resolve_cli_mode,
    validate_regeneration_inputs,
    validate_regeneration_paths,
    _choose_overlap_lengths,
    _overlap_add,
    _write_audio,
)


class AudioWriteTests(unittest.TestCase):
    """Verify ordinary audio publication does not import MP3-only helpers."""

    def test_wav_write_does_not_require_pydub(self) -> None:
        """Keep WAV regeneration usable when pydub is unavailable."""
        real_import = builtins.__import__

        def fail_pydub(name, *args, **kwargs):
            if name == "pydub" or name.startswith("pydub."):
                raise ImportError("pydub unavailable")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.wav"
            with patch("builtins.__import__", side_effect=fail_pydub):
                _write_audio(
                    output,
                    np.zeros((1, 100), dtype=np.float32),
                    1_000,
                )
            self.assertTrue(output.is_file())


class ParserTests(unittest.TestCase):
    """Verify timestamp, replacement, and subtitle parsing."""

    def test_parse_cli_timestamp_accepts_strict_milliseconds(self) -> None:
        """Parse a strict CLI timestamp into seconds."""
        self.assertAlmostEqual(3661.009, parse_cli_timestamp("01:01:01.009"))

    def test_parse_cli_timestamp_rejects_ambiguous_format(self) -> None:
        """Reject timestamps without exactly three millisecond digits."""
        with self.assertRaisesRegex(ValueError, "HH:MM:SS.mmm"):
            parse_cli_timestamp("00:00:04.5")

    def test_parse_replacements_accepts_multiple_items(self) -> None:
        """Parse multiple replacement triples in their original order."""
        requests = parse_replacements(
            '[["00:00:04.509","00:00:06.655","第一句；"],'
            '["00:00:19.537","00:00:21.899","第二句，"]]'
        )
        self.assertEqual(2, len(requests))
        self.assertAlmostEqual(4.509, requests[0].start)
        self.assertEqual("第二句，", requests[1].text)

    def test_parse_replacements_accepts_index_tts_four_item_contract(self) -> None:
        """Parse per-request speech speed and one text item per subtitle."""
        requests = parse_replacements(
            '[["00:00:04.509","00:00:06.655",["新的句子。"],1.1]]'
        )
        self.assertEqual(["新的句子。"], requests[0].text)
        self.assertAlmostEqual(1.1, requests[0].speech_speed)

    def test_parse_replacements_rejects_invalid_speech_speed(self) -> None:
        """Reject missing or non-positive per-request speech speed."""
        with self.assertRaisesRegex(ValueError, "speechSpeed"):
            parse_replacements(
                '[["00:00:01.000","00:00:02.000","new",0]]'
            )

    def test_normalize_merges_adjacent_requests_with_matching_speed(self) -> None:
        """Merge complete adjacent requests before independent TTS generation."""
        subtitles = [
            SubtitleItem(0.0, 1.0, "旧一。"),
            SubtitleItem(1.0, 2.0, "旧二。"),
            SubtitleItem(3.0, 4.0, "旧三。"),
        ]
        requests = parse_replacements(
            '[["00:00:00.000","00:00:01.000","新一。",1.1],'
            '["00:00:01.000","00:00:02.000","新二。",1.1],'
            '["00:00:03.000","00:00:04.000","新三。",0.9]]'
        )

        normalized = normalize_replacement_requests(subtitles, requests)

        self.assertEqual(2, len(normalized))
        self.assertEqual(["新一。", "新二。"], normalized[0].text)
        self.assertAlmostEqual(1.1, normalized[0].speech_speed)
        self.assertEqual("新三。", normalized[1].text)

    def test_normalize_rejects_adjacent_requests_with_different_speed(self) -> None:
        """Do not fake coherent speech when adjacent requests use different speed."""
        subtitles = [SubtitleItem(0.0, 1.0, "旧一。"), SubtitleItem(1.0, 2.0, "旧二。")]
        requests = parse_replacements(
            '[["00:00:00.000","00:00:01.000","新一。",1.0],'
            '["00:00:01.000","00:00:02.000","新二。",1.1]]'
        )

        with self.assertRaisesRegex(ValueError, "speech speed"):
            normalize_replacement_requests(subtitles, requests)

    def test_parse_replacements_rejects_blank_text(self) -> None:
        """Reject a replacement whose new script is blank."""
        with self.assertRaisesRegex(ValueError, "Replacement #1"):
            parse_replacements('[["00:00:01.000","00:00:02.000","  "]]')

    def test_load_srt_supports_multiline_text(self) -> None:
        """Load SRT blocks and retain multiline subtitle text."""
        content = (
            "1\n00:00:00,000 --> 00:00:01,500\n第一行\n第二行\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n结束\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.srt"
            path.write_text(content, encoding="utf-8")
            subtitles = load_subtitles(str(path))
        self.assertEqual("第一行\n第二行", subtitles[0].text)

    def test_load_json_supports_sentences_shape(self) -> None:
        """Load the JSON subtitle sentences structure."""
        payload = {
            "audio_file": "source.wav",
            "sample_rate": 16000,
            "total_seconds": 3.0,
            "sentences": [
                {"start": 0.0, "end": 1.5, "text": "第一句。"},
                {"start": 2.0, "end": 3.0, "text": "第二句。"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            subtitles = load_subtitles(str(path))
        self.assertEqual("第二句。", subtitles[1].text)


class TimelineTests(unittest.TestCase):
    """Verify range validation and cumulative subtitle offsets."""

    def setUp(self) -> None:
        """Create a timeline with two replaceable subtitle intervals."""
        self.subtitles = [
            SubtitleItem(0.0, 4.0, "开场。"),
            SubtitleItem(4.509, 6.655, "旧句子一。"),
            SubtitleItem(7.0, 10.0, "中间。"),
            SubtitleItem(19.537, 21.899, "旧句子二。"),
            SubtitleItem(22.0, 25.0, "结尾。"),
        ]

    @staticmethod
    def splitter(text: str, duration: float, max_char_len: int) -> list[SubtitleItem]:
        """Return one subtitle spanning the generated clip."""
        del max_char_len
        return [SubtitleItem(0.0, duration, text)]

    def test_validate_rejects_partial_subtitle_coverage(self) -> None:
        """Reject a replacement that starts inside an existing subtitle."""
        requests = [ReplacementRequest(5.0, 6.655, "新句子。")]
        with self.assertRaisesRegex(ValueError, "complete subtitle boundaries"):
            validate_regeneration_inputs(self.subtitles, requests, 25.0)

    def test_validate_rejects_overlapping_replacements(self) -> None:
        """Reject overlapping replacement requests before synthesis."""
        requests = [
            ReplacementRequest(4.509, 6.655, "一。"),
            ReplacementRequest(6.5, 10.0, "二。"),
        ]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_regeneration_inputs(self.subtitles, requests, 25.0)

    def test_rebuild_applies_cumulative_duration_offsets(self) -> None:
        """Shift later subtitles by all preceding duration differences."""
        requests = [
            ReplacementRequest(4.509, 6.655, "新句子一。"),
            ReplacementRequest(19.537, 21.899, "新句子二。"),
        ]
        rebuilt = rebuild_subtitle_timeline(
            self.subtitles,
            requests,
            generated_durations=[3.0, 1.0],
            display_texts=["新句子一。", "新句子二。"],
            subtitle_splitter=self.splitter,
            max_char_len=80,
        )
        first_delta = 3.0 - (6.655 - 4.509)
        total_delta = first_delta + 1.0 - (21.899 - 19.537)
        self.assertAlmostEqual(4.509, rebuilt[1].start)
        self.assertAlmostEqual(7.0 + first_delta, rebuilt[2].start)
        self.assertAlmostEqual(22.0 + total_delta, rebuilt[4].start)

    def test_rebuild_rejects_invalid_splitter_output(self) -> None:
        """Reject a subtitle splitter that creates a zero-duration item."""

        def invalid_splitter(
            text: str, duration: float, max_char_len: int
        ) -> list[SubtitleItem]:
            """Return an intentionally invalid item for validation coverage."""
            del duration, max_char_len
            return [SubtitleItem(0.0, 0.0, text)]

        with self.assertRaisesRegex(ValueError, "start < end"):
            rebuild_subtitle_timeline(
                self.subtitles,
                [ReplacementRequest(4.509, 6.655, "新句子。")],
                generated_durations=[1.0],
                display_texts=["新句子。"],
                subtitle_splitter=invalid_splitter,
                max_char_len=80,
            )


class CliModeTests(unittest.TestCase):
    """Verify mutually exclusive advanced CLI modes."""

    def test_advanced_parser_accepts_regeneration_without_text(self) -> None:
        """Allow regeneration arguments without the normal synthesis text flag."""
        from omnivoice.cli.cli_infer import get_parser

        args = get_parser().parse_args(
            [
                "--source_audio",
                "source.wav",
                "--source_subtitle",
                "source.srt",
                "--regenerate_segments",
                '[["00:00:00.000","00:00:01.000","new"]]',
                "--srt",
                "output.srt",
                "--output",
                "output.wav",
            ]
        )
        self.assertIsNone(args.text)
        self.assertEqual("source.wav", args.source_audio)

    def test_resolve_cli_mode_accepts_regeneration(self) -> None:
        """Select regeneration when all required source inputs are present."""
        self.assertEqual(
            "regeneration",
            resolve_cli_mode(
                None,
                "source.wav",
                "source.srt",
                "[]",
                "output.srt",
                None,
            ),
        )

    def test_resolve_cli_mode_rejects_mixed_modes(self) -> None:
        """Reject normal synthesis text together with regeneration inputs."""
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_cli_mode(
                "hello",
                "source.wav",
                "source.srt",
                "[]",
                "output.srt",
                None,
            )

    def test_resolve_cli_mode_requires_subtitle_output(self) -> None:
        """Require at least one rebuilt subtitle output."""
        with self.assertRaisesRegex(ValueError, "subtitle output"):
            resolve_cli_mode(None, "source.wav", "source.srt", "[]", None, None)


class PathValidationTests(unittest.TestCase):
    """Verify that source and output files are protected before inference."""

    @staticmethod
    def create_source_files(temp_dir: str) -> tuple[Path, Path]:
        """Create minimal files sufficient for path-only validation."""
        source_audio = Path(temp_dir) / "source.wav"
        source_audio.write_bytes(b"path validation only")
        source_subtitle = Path(temp_dir) / "source.srt"
        source_subtitle.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nold\n",
            encoding="utf-8",
        )
        return source_audio, source_subtitle

    def test_validate_paths_rejects_existing_output(self) -> None:
        """Refuse to overwrite an existing regeneration output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio, source_subtitle = self.create_source_files(temp_dir)
            output_audio = Path(temp_dir) / "output.wav"
            output_audio.write_bytes(b"existing")
            options = RegenerationOptions(
                str(source_audio),
                str(source_subtitle),
                str(output_audio),
                str(Path(temp_dir) / "output.srt"),
                None,
                80,
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                validate_regeneration_paths(options)

    def test_validate_paths_rejects_source_output_collision(self) -> None:
        """Refuse to publish audio over the source audio file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio, source_subtitle = self.create_source_files(temp_dir)
            options = RegenerationOptions(
                str(source_audio),
                str(source_subtitle),
                str(source_audio),
                str(Path(temp_dir) / "output.srt"),
                None,
                80,
            )
            with self.assertRaisesRegex(ValueError, "must not match"):
                validate_regeneration_paths(options)


class AudioRegenerationTests(unittest.TestCase):
    """Verify waveform splicing without initializing OmniVoice."""

    @staticmethod
    def cleaner(text: str) -> str:
        """Return display text unchanged for orchestration tests."""
        return text

    @staticmethod
    def splitter(text: str, duration: float, max_char_len: int) -> list[SubtitleItem]:
        """Return one subtitle spanning one generated replacement clip."""
        del max_char_len
        return [SubtitleItem(0.0, duration, text)]

    def test_overlap_add_uses_equal_power_and_actual_samples(self) -> None:
        """Use 15ms overlap and close the output sample count exactly."""
        overlaps = _choose_overlap_lengths((100, 100), 15)
        self.assertEqual((15,), overlaps)
        left = np.ones((1, 100), dtype=np.float32)
        right = np.ones((1, 100), dtype=np.float32)
        result = _overlap_add(left, right, overlaps[0])
        self.assertEqual(185, result.shape[-1])
        self.assertTrue(np.isfinite(result).all())

    def test_regenerate_splices_clip_and_writes_matching_subtitles(self) -> None:
        """Replace one second with two seconds and shift the following subtitle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio = Path(temp_dir) / "source.wav"
            source_subtitle = Path(temp_dir) / "source.srt"
            output_audio = Path(temp_dir) / "output.wav"
            output_srt = Path(temp_dir) / "output.srt"
            output_json = Path(temp_dir) / "output.json"
            sf.write(source_audio, np.zeros((16000 * 4, 2), dtype=np.float32), 16000)
            source_audio_bytes = source_audio.read_bytes()
            source_subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nold\n\n"
                "2\n00:00:01,000 --> 00:00:04,000\ntail\n",
                encoding="utf-8",
            )

            def fake_segment_generator(request: ReplacementRequest, output_path: str) -> None:
                """Write a deterministic two-second mono replacement WAV."""
                self.assertEqual("new", request.text)
                sf.write(
                    output_path,
                    np.ones(22050 * 2, dtype=np.float32) * 0.01,
                    22050,
                )

            result = regenerate_audio_segments(
                options=RegenerationOptions(
                    str(source_audio),
                    str(source_subtitle),
                    str(output_audio),
                    str(output_srt),
                    str(output_json),
                    80,
                ),
                requests=[ReplacementRequest(0.0, 1.0, "new")],
                segment_generator=fake_segment_generator,
                subtitle_cleaner=self.cleaner,
                subtitle_splitter=self.splitter,
            )

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            audio_info = sf.info(output_audio)
            self.assertEqual(source_audio_bytes, source_audio.read_bytes())
            self.assertAlmostEqual(4.985, result.total_seconds, places=3)
            self.assertEqual(16000, audio_info.samplerate)
            self.assertEqual(2, audio_info.channels)
            self.assertEqual("new", payload["sentences"][0]["text"])
            self.assertAlmostEqual(1.985, payload["sentences"][1]["start"], places=3)
            self.assertIn(
                "00:00:01,985 --> 00:00:04,985",
                output_srt.read_text(encoding="utf-8"),
            )

    def test_regeneration_resolves_whisperx_against_alignment_text(self) -> None:
        """Keep display text separate from the transcript supplied to WhisperX."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio = Path(temp_dir) / "source.wav"
            source_subtitle = Path(temp_dir) / "source.srt"
            output_audio = Path(temp_dir) / "output.wav"
            output_srt = Path(temp_dir) / "output.srt"
            sf.write(source_audio, np.zeros(16000 * 2, dtype=np.float32), 16000)
            source_subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nold one\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\nold two\n",
                encoding="utf-8",
            )
            captured: list[tuple[str, ...]] = []

            def generator(request: ReplacementRequest, output_path: str) -> None:
                """Write one replacement waveform for the coherent source range."""
                del request
                sf.write(output_path, np.ones(16000 * 2, dtype=np.float32), 16000)

            def resolver(
                audio_path: str,
                alignment_texts: tuple[str, ...],
                generated_samples: int,
                sample_rate: int,
            ) -> TimingResolution:
                """Capture the alignment layer instead of visible subtitle text."""
                del audio_path, generated_samples, sample_rate
                captured.append(alignment_texts)
                return TimingResolution(
                    (SentenceTiming(0, 16000), SentenceTiming(16000, 32000)),
                    "whisperx_alignment",
                    1.0,
                    None,
                )

            result = regenerate_audio_segments(
                options=RegenerationOptions(
                    str(source_audio),
                    str(source_subtitle),
                    str(output_audio),
                    str(output_srt),
                    None,
                    80,
                ),
                requests=[
                    ReplacementRequest(
                        0.0,
                        2.0,
                        ["visible-one", "visible-two"],
                    )
                ],
                segment_generator=generator,
                subtitle_cleaner=self.cleaner,
                subtitle_splitter=self.splitter,
                alignment_resolver=resolver,
                alignment_text_resolver=lambda text: f"align:{text}",
            )

            self.assertEqual(
                [["align:visible-one", "align:visible-two"]],
                captured,
            )
            self.assertEqual(
                ["visible-one", "visible-two"],
                [item.text for item in result.subtitles],
            )

    def test_generator_failure_does_not_publish_outputs(self) -> None:
        """Leave no formal output when replacement synthesis raises an error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_audio = Path(temp_dir) / "source.wav"
            source_subtitle = Path(temp_dir) / "source.srt"
            output_audio = Path(temp_dir) / "output.wav"
            output_srt = Path(temp_dir) / "output.srt"
            sf.write(source_audio, np.zeros(16000, dtype=np.float32), 16000)
            source_subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nold\n",
                encoding="utf-8",
            )

            def failing_generator(request: ReplacementRequest, output_path: str) -> None:
                """Simulate a TTS failure before writing a segment."""
                del request, output_path
                raise RuntimeError("synthetic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                regenerate_audio_segments(
                    options=RegenerationOptions(
                        str(source_audio),
                        str(source_subtitle),
                        str(output_audio),
                        str(output_srt),
                        None,
                        80,
                    ),
                    requests=[ReplacementRequest(0.0, 1.0, "new")],
                    segment_generator=failing_generator,
                    subtitle_cleaner=self.cleaner,
                    subtitle_splitter=self.splitter,
                )

            self.assertFalse(output_audio.exists())
            self.assertFalse(output_srt.exists())


if __name__ == "__main__":
    unittest.main()
