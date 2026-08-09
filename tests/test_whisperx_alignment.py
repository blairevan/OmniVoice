"""Tests for the isolated shared WhisperX adapter."""

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from omnivoice.utils.whisperx_alignment import (
    WhisperXAligner,
    WhisperXAlignmentDeadline,
    WhisperXAlignmentError,
)


class WhisperXAlignmentTests(unittest.TestCase):
    """Verify command construction, deadlines, and output validation."""

    def create_runtime(self, directory: str) -> tuple[Path, Path]:
        """Create a minimal isolated runtime and local model directory."""
        runtime = Path(directory) / "whisperx"
        runtime.mkdir()
        (runtime / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
        (runtime / "align.py").write_text("# test runtime\n", encoding="utf-8")
        model = Path(directory) / "model"
        model.mkdir()
        return runtime, model

    def test_align_uses_offline_runtime_and_remaining_deadline(self) -> None:
        """Run the exact shared-runtime command with offline environment flags."""
        with tempfile.TemporaryDirectory() as directory:
            runtime, model = self.create_runtime(directory)
            audio_path = Path(directory) / "input.wav"
            audio_path.write_bytes(b"audio")
            observed: dict[str, object] = {}

            def run(command, **kwargs):
                """Write a valid alignment payload to the requested output."""
                observed["command"] = command
                observed.update(kwargs)
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        {"segments": [{"chars": [{"char": "你", "start": 0.1, "end": 0.2}]}]}
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            deadline = WhisperXAlignmentDeadline.from_timeout_seconds(10)
            aligner = WhisperXAligner(str(runtime), "zh", "cuda", str(model))
            with patch("subprocess.run", side_effect=run):
                result = aligner.align(str(audio_path), "你", 0.0, 1.0, deadline)

            self.assertEqual("你", result[0].character)
            command = observed["command"]
            self.assertIn("--project", command)
            self.assertIn(str(runtime.resolve()), command)
            self.assertIn(str(model.resolve()), command)
            self.assertEqual("cuda", command[command.index("--device") + 1])
            environment = observed["env"]
            self.assertEqual("1", environment["HF_HUB_OFFLINE"])
            self.assertEqual("1", environment["TRANSFORMERS_OFFLINE"])
            self.assertLessEqual(observed["timeout"], 10)

    def test_deadline_exhaustion_does_not_spawn_child(self) -> None:
        """Reject an expired task before invoking the external runtime."""
        deadline = WhisperXAlignmentDeadline(
            started_at=time.monotonic() - 2.0,
            timeout_seconds=1.0,
        )
        aligner = WhisperXAligner("/missing", "zh", "cuda", "/missing-model")
        with patch("subprocess.run") as run:
            with self.assertRaisesRegex(WhisperXAlignmentError, "deadline"):
                aligner.align("audio.wav", "你", 0.0, 1.0, deadline)
        run.assert_not_called()

    def test_invalid_output_is_rejected(self) -> None:
        """Reject malformed character alignment output."""
        with tempfile.TemporaryDirectory() as directory:
            runtime, model = self.create_runtime(directory)

            def run(command, **_kwargs):
                """Write an invalid response."""
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_text('{"segments":[{}]}', encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            aligner = WhisperXAligner(str(runtime), "zh", "cuda", str(model))
            with patch("subprocess.run", side_effect=run):
                with self.assertRaisesRegex(WhisperXAlignmentError, "chars"):
                    aligner.align(
                        str(Path(directory) / "audio.wav"),
                        "你",
                        0.0,
                        1.0,
                        WhisperXAlignmentDeadline.from_timeout_seconds(10),
                    )


if __name__ == "__main__":
    unittest.main()
