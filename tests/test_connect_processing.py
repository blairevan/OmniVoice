"""Unit tests for deterministic connect parsing and waveform editing."""

import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

from omnivoice.utils.connect_candidate_selector import CharacterTimestamp, internal_boundaries
from omnivoice.utils.connect_markup import parse_connect_markup, split_connect_markup
from omnivoice.utils.connect_waveform_processor import (
    ConnectProcessingOptions,
    process_connect_waveform,
)
from omnivoice.utils.connect_candidate_pipeline import (
    ConnectRuntimeOptions,
    _normalize_funasr_timestamps,
    _resolve_local_funasr_model_path,
    select_connect_waveform,
)


class ConnectMarkupTests(unittest.TestCase):
    """Verify connect ranges do not extend into neighboring text."""

    def test_connect_range_excludes_following_text(self) -> None:
        """Map only adjacent characters inside the connect tag."""
        markup = parse_connect_markup("[connect:课程]门户")
        timestamps = (
            CharacterTimestamp("课", 0, 80),
            CharacterTimestamp("程", 100, 180),
            CharacterTimestamp("门", 200, 280),
            CharacterTimestamp("户", 300, 380),
        )
        boundaries = internal_boundaries(markup, timestamps)
        self.assertEqual(1, len(boundaries))
        self.assertEqual(("课", "程"), (boundaries[0][0].character, boundaries[0][1].character))

    def test_connect_replacement_subtitle_uses_full_real_duration(self) -> None:
        """Avoid proportional splitting after a candidate changes its duration."""
        from omnivoice.cli.cli_infer import _split_replacement_subtitles
        subtitles = _split_replacement_subtitles("[connect:课程门户首页]副屏组件。", 1.234, 2)
        self.assertEqual(1, len(subtitles))
        self.assertAlmostEqual(1.234, subtitles[0].end)

    def test_pre_split_never_breaks_connect_range(self) -> None:
        """Keep an atomic connect range together while limiting surrounding text."""
        units = split_connect_markup(parse_connect_markup("前言。[connect:课程门户]后续。"), 5)
        self.assertEqual(["前言。", "课程门户", "后续。"], [item.synthesis_text for item in units])
        self.assertEqual(1, len(units[1].ranges))

    def test_pre_split_preserves_replace_markup(self) -> None:
        """Split at markup token boundaries even when display and speech differ."""
        units = split_connect_markup(parse_connect_markup("[replace:展示词|合成词][connect:课程门户]结束"), 4)
        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(any(item.ranges for item in units))

    def test_markup_without_connect_splits_at_punctuation(self) -> None:
        """Use the same punctuation forced-segment rule for ordinary markup."""
        units = split_connect_markup(
            parse_connect_markup("[replace:展示|展示]第一句。第二句。"),
            80,
        )
        self.assertEqual([item.subtitle_text for item in units], ["展示第一句。", "第二句。"])

    def test_pronunciation_markup_uses_display_characters_for_alignment(self) -> None:
        """Align pronunciation overrides with their Chinese display characters."""
        markup = parse_connect_markup("[汉字|HAN ZI][connect:课程]")
        self.assertEqual("汉字课程", markup.alignment_text)
        self.assertEqual((None, None, 7, 8), markup.alignment_source_offsets)


class ConnectWaveformTests(unittest.TestCase):
    """Verify only a long low-energy gap can be shortened."""

    def test_processor_shortens_low_energy_gap(self) -> None:
        """Remove part of silence while retaining a residual gap."""
        sample_rate = 1_000
        waveform = np.concatenate((np.ones(100, dtype=np.float32) * .2, np.zeros(160, dtype=np.float32), np.ones(100, dtype=np.float32) * .2))
        result = process_connect_waveform(
            waveform, sample_rate,
            [(CharacterTimestamp("课", 0, 100), CharacterTimestamp("程", 260, 360))],
            ConnectProcessingOptions(maximum_shorten_ms=80, boundary_guard_ms=20, minimum_low_energy_ms=30, residual_gap_ms=20, crossfade_ms=8),
        )
        self.assertGreater(result.total_removed_samples, 0)
        self.assertEqual(waveform.size - result.total_removed_samples, result.waveform.size)

    def test_processor_preserves_protected_gap(self) -> None:
        """Keep a gap that disappears after the two protection zones."""
        waveform = np.concatenate((np.ones(100, dtype=np.float32) * .2, np.zeros(30, dtype=np.float32), np.ones(100, dtype=np.float32) * .2))
        result = process_connect_waveform(
            waveform, 1_000,
            [(CharacterTimestamp("课", 0, 100), CharacterTimestamp("程", 130, 230))],
            ConnectProcessingOptions(),
        )
        self.assertEqual(0, result.total_removed_samples)
        self.assertTrue(np.array_equal(waveform, result.waveform))

    def test_processor_preserves_multichannel_shape_and_phase(self) -> None:
        """Edit all channels with one shared cut and retain exact shape accounting."""
        left = np.full(100, 0.2, dtype=np.float32)
        gap = np.zeros(160, dtype=np.float32)
        right = np.full(100, -0.2, dtype=np.float32)
        waveform = np.vstack((np.concatenate((left, gap, right)), np.concatenate((left * 0.5, gap, right * 0.5))))
        result = process_connect_waveform(
            waveform,
            1_000,
            [(CharacterTimestamp("课", 0, 100), CharacterTimestamp("程", 260, 360))],
            ConnectProcessingOptions(maximum_shorten_ms=80, boundary_guard_ms=20, minimum_low_energy_ms=30, residual_gap_ms=20, crossfade_ms=8),
        )
        self.assertEqual(result.waveform.shape[0], 2)
        self.assertEqual(result.waveform.shape[1], waveform.shape[1] - result.total_removed_samples)
        self.assertTrue(np.isfinite(result.waveform).all())


class ConnectPipelineTests(unittest.TestCase):
    """Verify candidate evidence and selection without downloading FunASR."""

    def test_funasr_model_path_requires_complete_local_snapshot(self) -> None:
        """Resolve a complete local snapshot and reject missing runtime models."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "configuration.json").write_text("{}", encoding="utf-8")
            (model_path / "model.pt").write_bytes(b"weights")
            self.assertEqual(
                str(model_path),
                _resolve_local_funasr_model_path(str(model_path)),
            )

        with patch.dict("os.environ", {"MODELSCOPE_CACHE": directory}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Runtime download is disabled"):
                _resolve_local_funasr_model_path("fa-zh")

    def test_pipeline_persists_original_alignment_and_selection(self) -> None:
        """Keep candidate evidence when an explicit debug directory is supplied."""
        markup = parse_connect_markup("[connect:课程]")
        waveform = np.concatenate((np.ones(100, dtype=np.float32) * .2, np.zeros(80, dtype=np.float32), np.ones(100, dtype=np.float32) * .2))
        def align(_path: str, _text: str) -> tuple[CharacterTimestamp, ...]:
            return (CharacterTimestamp("课", 0, 100), CharacterTimestamp("程", 180, 280))
        with tempfile.TemporaryDirectory() as directory:
            debug_dir = str(Path(directory) / "connect_debug")
            with self.assertLogs("omnivoice.utils.connect_candidate_pipeline", level="INFO") as captured:
                selection = select_connect_waveform(markup, 1000, lambda: waveform, align, ConnectRuntimeOptions(candidates=1, debug_dir=debug_dir), ConnectProcessingOptions())
            self.assertEqual(1, selection.candidate_index)
            self.assertTrue(any("[connect] candidate selected candidate=1" in line for line in captured.output))
            self.assertTrue(any("[connect] candidate evaluation candidate=1" in line for line in captured.output))
            self.assertTrue((Path(debug_dir) / "candidate_001_original_alignment.json").is_file())
            self.assertTrue((Path(debug_dir) / "selection.json").is_file())
            selection_data = (Path(debug_dir) / "selection.json").read_text(encoding="utf-8")
            self.assertIn('"candidateReports"', selection_data)

    def test_pipeline_skips_failed_candidate_and_keeps_later_candidates(self) -> None:
        """A single generation failure must not abort candidate selection."""
        markup = parse_connect_markup("[connect:课程]")
        waveform = np.ones(200, dtype=np.float32) * 0.2
        calls = 0

        def generate() -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("synthetic candidate failure")
            return waveform

        def align(_path: str, _text: str) -> tuple[CharacterTimestamp, ...]:
            return (CharacterTimestamp("课", 0, 80), CharacterTimestamp("程", 80, 160))

        with tempfile.TemporaryDirectory() as directory:
            debug_dir = str(Path(directory) / "connect_debug")
            selection = select_connect_waveform(
                markup,
                1_000,
                generate,
                align,
                ConnectRuntimeOptions(candidates=2, processing="off", debug_dir=debug_dir),
                None,
            )
            self.assertEqual(selection.candidate_index, 2)
            evidence = (Path(debug_dir) / "selection.json").read_text(encoding="utf-8")
            self.assertIn("synthetic candidate failure", evidence)

    def test_normalize_funasr_timestamp_value_shape(self) -> None:
        """Accept FunASR's alternate value field and preserve strict coverage."""
        result = [{"value": [[0, 80], [90, 170]]}]
        timestamps = _normalize_funasr_timestamps(result, "课程")
        self.assertEqual((timestamps[0].character, timestamps[-1].end_ms), ("课", 170.0))
