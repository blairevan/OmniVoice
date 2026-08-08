"""Pure CLI contract tests for forced-segment generation and publication."""

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from omnivoice.cli.cli_infer import (
    _validate_connect_debug_path,
    _resolve_model_token_limit,
    concatenate_audio_actions,
    get_parser,
    save_audio,
)


class _FakeModel:
    """Capture generation kwargs without loading a TTS model."""

    sampling_rate = 1_000

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return (np.ones(100, dtype=np.float32) * 0.1,)


class _TokenLimitedModel:
    """Expose a tokenizer limit for model-contract tests."""

    class tokenizer:
        model_max_length = 256


class ConnectCliTests(unittest.TestCase):
    """Verify public parser and forced-segment contracts."""

    def test_parser_exposes_conservative_defaults(self) -> None:
        """Keep three candidates and boundary silence as the safe defaults."""
        args = get_parser().parse_args(["--text", "中文"])
        self.assertEqual(args.connect_candidates, 3)
        self.assertEqual(args.connect_processing, "conservative")
        self.assertAlmostEqual(args.boundary_silence, 0.3)

    def test_forced_segment_disables_internal_postprocessing(self) -> None:
        """Every forced segment must request an exact unpadded waveform."""
        model = _FakeModel()
        concatenate_audio_actions(
            [("text", "[replace:展示词|合成词]后续文本")], model, None, None, None, "zh", 1.0,
            4, 1, 1.0, False, 0.0, 0.0, 1.0, 1.0, 1.0,
        )
        self.assertEqual(len(model.calls), 2)
        for call in model.calls:
            self.assertFalse(call["postprocess_output"])
            self.assertEqual(call["audio_chunk_duration"], 0.0)
            self.assertEqual(call["audio_chunk_threshold"], 0.0)
            self.assertEqual(call["pad_duration"], 0.0)
            self.assertEqual(call["fade_duration"], 0.0)

    def test_plain_punctuation_is_synthesized_as_separate_segments(self) -> None:
        """Keep ordinary subtitle units aligned to independently generated audio."""
        model = _FakeModel()
        _, _, timestamps = concatenate_audio_actions(
            [("text", "第一句。第二句。")], model, None, None, None, "zh", 1.0,
            80, 1, 1.0, False, 0.0, 0.0, 1.0, 1.0, 1.0,
        )
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([item["text"] for item in timestamps], ["第一句。", "第二句。"])

    def test_mp3_fallback_returns_actual_wav_path(self) -> None:
        """When pydub is unavailable, publication metadata can use the WAV path."""
        with tempfile.TemporaryDirectory() as directory:
            requested = str(Path(directory) / "result.mp3")
            real_import = builtins.__import__

            def fail_pydub(name, *args, **kwargs):
                if name == "pydub":
                    raise ImportError("test")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_pydub):
                actual = save_audio(np.zeros(100, dtype=np.float32), 1_000, requested)
            self.assertTrue(actual.endswith(".wav"))
            self.assertTrue(Path(actual).is_file())

    def test_debug_directory_cannot_collide_with_output(self) -> None:
        """Validate debug paths before model loading or synthesis."""
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "output.wav")
            with self.assertRaises(ValueError):
                _validate_connect_debug_path(output, (output, None, None, None, None))

    def test_model_token_limit_is_used_when_exposed(self) -> None:
        """Prefer a finite tokenizer limit over an implicit character guess."""
        self.assertEqual(_resolve_model_token_limit(_TokenLimitedModel()), 256)


if __name__ == "__main__":
    unittest.main()
