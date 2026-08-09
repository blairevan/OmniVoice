"""Pure CLI contract tests for forced-segment generation and publication."""

import builtins
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from omnivoice.cli.cli_infer import (
    _apply_subtitle_offset,
    _detect_active_start_seconds,
    _reset_connect_debug_dir,
    _resolve_subtitle_offset,
    _validate_connect_debug_path,
    _resolve_model_token_limit,
    concatenate_audio_actions,
    get_parser,
    save_audio,
)
from omnivoice.models.omnivoice import _resolve_model_path
from omnivoice.utils.connect_candidate_pipeline import ConnectRuntimeOptions
from omnivoice.utils.synthesis_orchestrator import (
    generate_coherent_actions,
    resolve_generated_group_timings,
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


class _CoherentFakeModel:
    """Generate a deterministic waveform for coherent orchestration tests."""

    sampling_rate = 1_000

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        """Record one model call and return one short waveform."""
        self.calls.append(kwargs)
        return (np.ones(100, dtype=np.float32) * 0.1,)


class ConnectCliTests(unittest.TestCase):
    """Verify public parser and forced-segment contracts."""

    def test_parser_exposes_conservative_defaults(self) -> None:
        """Keep three candidates and split silence defaults as the safe defaults."""
        args = get_parser().parse_args(["--text", "中文"])
        self.assertEqual(args.connect_candidates, 3)
        self.assertEqual(args.connect_processing, "conservative")
        self.assertAlmostEqual(args.leading_silence, 300.0)
        self.assertAlmostEqual(args.trailing_silence, 300.0)
        self.assertEqual(args.subtitle_offset, "auto")
        self.assertTrue(args.offline)

    def test_parser_rejects_legacy_boundary_silence(self) -> None:
        """Reject the removed legacy boundary silence option."""
        with self.assertRaises(SystemExit):
            get_parser().parse_args(["--text", "中文", "--boundary_silence", "0.3"])

    def test_parser_accepts_index_tts_silence_arguments(self) -> None:
        """Accept the queue consumer's IndexTTS-compatible timing arguments."""
        args = get_parser().parse_args(
            [
                "--text", "中文",
                "--leading_silence", "300",
                "--trailing_silence", "150",
                "--subtitle_offset", "auto",
            ]
        )
        self.assertAlmostEqual(args.leading_silence, 300.0)
        self.assertAlmostEqual(args.trailing_silence, 150.0)

    def test_parser_can_explicitly_enable_hub_access(self) -> None:
        """Keep an explicit opt-out for environments that permit downloads."""
        args = get_parser().parse_args(["--text", "中文", "--no-offline"])
        self.assertFalse(args.offline)

    def test_model_path_resolution_uses_local_hub_cache_offline(self) -> None:
        """Pass local_files_only to HuggingFace instead of waiting on the network."""
        with patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}, clear=False), patch(
            "huggingface_hub.snapshot_download", return_value="/cached/model"
        ) as download:
            resolved = _resolve_model_path("k2-fsa/OmniVoice")

        self.assertEqual("/cached/model", resolved)
        download.assert_called_once_with("k2-fsa/OmniVoice", local_files_only=True)

    def test_subtitle_offset_matches_index_tts_semantics(self) -> None:
        """Shift the first end and later bounds without moving the first start."""
        audio = np.concatenate((np.zeros(100), np.ones(100))).astype(np.float32)
        self.assertAlmostEqual(0.1, _detect_active_start_seconds(audio, 1000))
        self.assertAlmostEqual(0.1, _resolve_subtitle_offset("auto", 0.1))
        timestamps = [
            {"start": 0.3, "end": 1.0, "text": "一"},
            {"start": 1.0, "end": 1.5, "text": "二"},
        ]
        _apply_subtitle_offset(timestamps, 0.1)
        self.assertEqual(0.3, timestamps[0]["start"])
        self.assertEqual(0.9, timestamps[0]["end"])
        self.assertEqual(0.9, timestamps[1]["start"])
        self.assertEqual(1.4, timestamps[1]["end"])

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

    def test_debug_directory_is_overwritten_on_retry(self) -> None:
        """Clear stale evidence instead of allocating a suffix directory."""
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "output_connect_debug"
            first.mkdir()
            (first / "stale.json").write_text("stale", encoding="utf-8")
            _reset_connect_debug_dir(str(first))
            self.assertFalse(first.exists())

    def test_model_token_limit_is_used_when_exposed(self) -> None:
        """Prefer a finite tokenizer limit over an implicit character guess."""
        self.assertEqual(_resolve_model_token_limit(_TokenLimitedModel()), 256)

    def test_connect_candidate_receives_group_speed_for_evaluation(self) -> None:
        """Preserve the caller speed for connect gap evaluation before final stretch."""
        model = _CoherentFakeModel()
        options = ConnectRuntimeOptions(processing="off", speech_speed=1.2)
        with patch(
            "omnivoice.utils.synthesis_orchestrator.FunASRAligner"
        ), patch(
            "omnivoice.utils.synthesis_orchestrator.select_connect_waveform",
            return_value=SimpleNamespace(waveform=np.ones(100, dtype=np.float32)),
        ) as selector:
            generate_coherent_actions(
                actions=[("text", "[connect:你好]。")],
                model=model,
                voice=None,
                ref_text=None,
                instruct=None,
                language="zh",
                max_char_len=80,
                max_tokens=120,
                num_step=4,
                guidance_scale=2.0,
                denoise=True,
                t_shift=0.1,
                layer_penalty_factor=5.0,
                position_temperature=5.0,
                class_temperature=0.0,
                markup_version=None,
                connect_options=options,
            )

        self.assertAlmostEqual(1.2, selector.call_args.args[4].speech_speed)

    def test_normal_generation_merges_adjacent_sentences_into_one_call(self) -> None:
        """Preserve sentence context while retaining two subtitle units."""
        model = _CoherentFakeModel()
        result = generate_coherent_actions(
            actions=[("text", "第一句。第二句。")],
            model=model,
            voice=None,
            ref_text=None,
            instruct=None,
            language="zh",
            max_char_len=80,
            max_tokens=120,
            num_step=4,
            guidance_scale=2.0,
            denoise=True,
            t_shift=0.1,
            layer_penalty_factor=5.0,
            position_temperature=5.0,
            class_temperature=0.0,
            markup_version=None,
            connect_options=ConnectRuntimeOptions(processing="off"),
        )

        timestamps = resolve_generated_group_timings(result, None, None)

        self.assertEqual(1, len(model.calls))
        self.assertEqual("第一句。第二句。", model.calls[0]["text"])
        self.assertEqual(["第一句。", "第二句。"], [item["text"] for item in timestamps])


if __name__ == "__main__":
    unittest.main()
