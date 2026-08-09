"""Behavior tests for OmniVoice runtime YAML configuration."""

import tempfile
import unittest
from pathlib import Path

from omnivoice.utils.runtime_config import (
    RuntimeConfigError,
    WhisperXRuntimeConfig,
    load_whisperx_runtime_config,
)


class RuntimeConfigTests(unittest.TestCase):
    """Verify strict, side-effect-free WhisperX configuration loading."""

    def setUp(self) -> None:
        """Create one isolated directory for each configuration fixture."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write(self, content: str) -> Path:
        """Write one YAML fixture and return its path."""
        path = self.root / "config.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_absolute_whisperx_paths(self) -> None:
        """Load both approved server paths from a valid YAML mapping."""
        path = self._write(
            "whisperx:\n"
            "  runtime_dir: /root/digital_human/whisperx\n"
            "  model_dir: /root/digital_human/whisperx/models/zh\n"
        )

        self.assertEqual(
            load_whisperx_runtime_config(path),
            WhisperXRuntimeConfig(
                runtime_dir="/root/digital_human/whisperx",
                model_dir="/root/digital_human/whisperx/models/zh",
            ),
        )

    def test_missing_file_returns_empty_config(self) -> None:
        """Keep generated-timing fallback when the YAML file is absent."""
        missing = self.root / "missing.yaml"

        self.assertEqual(
            load_whisperx_runtime_config(missing),
            WhisperXRuntimeConfig(),
        )

    def test_missing_whisperx_section_returns_empty_config(self) -> None:
        """Ignore an unrelated valid root mapping without inventing paths."""
        path = self._write("logging:\n  level: info\n")

        self.assertEqual(
            load_whisperx_runtime_config(path),
            WhisperXRuntimeConfig(),
        )

    def test_malformed_yaml_is_rejected(self) -> None:
        """Reject syntax errors instead of silently disabling alignment."""
        path = self._write("whisperx: [\n")

        with self.assertRaisesRegex(RuntimeConfigError, "invalid YAML"):
            load_whisperx_runtime_config(path)

    def test_non_mapping_root_is_rejected(self) -> None:
        """Reject a root sequence because it cannot contain named sections."""
        path = self._write("- whisperx\n")

        with self.assertRaisesRegex(RuntimeConfigError, "root must be a mapping"):
            load_whisperx_runtime_config(path)

    def test_non_mapping_whisperx_section_is_rejected(self) -> None:
        """Reject a scalar WhisperX section before reading path fields."""
        path = self._write("whisperx: enabled\n")

        with self.assertRaisesRegex(RuntimeConfigError, "whisperx must be a mapping"):
            load_whisperx_runtime_config(path)

    def test_blank_path_is_rejected(self) -> None:
        """Reject blank configured paths that would disable alignment later."""
        path = self._write("whisperx:\n  runtime_dir: '  '\n")

        with self.assertRaisesRegex(RuntimeConfigError, "runtime_dir must be a non-empty string"):
            load_whisperx_runtime_config(path)

    def test_non_string_path_is_rejected(self) -> None:
        """Reject implicit YAML scalar conversion for filesystem paths."""
        path = self._write("whisperx:\n  model_dir: 123\n")

        with self.assertRaisesRegex(RuntimeConfigError, "model_dir must be a non-empty string"):
            load_whisperx_runtime_config(path)


if __name__ == "__main__":
    unittest.main()
