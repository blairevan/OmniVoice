"""Tests for offline model cache validation and manifest generation."""

import json
import tempfile
import unittest
from pathlib import Path

from omnivoice.utils.offline_cache import (
    CacheComponent,
    build_manifest,
    validate_component,
)
from scripts.prepare_offline_cache import (
    _download_huggingface,
    parse_args,
    prepare_cache,
)


class OfflineCacheValidationTests(unittest.TestCase):
    """Verify pure cache checks without downloading real models."""

    def make_component(self, root: Path) -> CacheComponent:
        """Create the component contract used by the temporary fixtures."""
        return CacheComponent(
            name="test-model",
            source="test/source",
            path=root,
            required_files=("config.json", "tokenizer_config.json"),
            weight_patterns=("*.safetensors", "*.bin"),
        )

    def test_complete_component_is_valid(self) -> None:
        """Accept required files and recursive model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "model.safetensors").write_bytes(b"weights")

            result = validate_component(self.make_component(root))

        self.assertTrue(result.ok)
        self.assertEqual(3, result.file_count)
        self.assertEqual(("nested/model.safetensors",), result.weight_files)
        self.assertEqual((), result.missing)

    def test_missing_required_file_is_reported(self) -> None:
        """Report missing required files without raising an exception."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.bin").write_bytes(b"weights")

            result = validate_component(self.make_component(root))

        self.assertFalse(result.ok)
        self.assertIn("tokenizer_config.json", result.missing)

    def test_missing_weight_file_is_reported(self) -> None:
        """Reject a directory that has metadata but no model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            result = validate_component(self.make_component(root))

        self.assertFalse(result.ok)
        self.assertEqual("no weight files matched configured patterns", result.error)

    def test_manifest_records_global_failure(self) -> None:
        """Expose component status and aggregate status in JSON-safe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = validate_component(self.make_component(root))

        manifest = build_manifest((result,))

        self.assertFalse(manifest["ok"])
        self.assertEqual("test-model", manifest["components"][0]["name"])
        json.dumps(manifest)

    def test_huggingface_download_uses_local_only_flag(self) -> None:
        """Pass the offline flag through to a fake HuggingFace downloader."""
        calls: list[dict[str, object]] = []

        def fake_downloader(**kwargs: object) -> str:
            calls.append(kwargs)
            return "/tmp/cached-model"

        result = _download_huggingface("test/model", True, fake_downloader)

        self.assertEqual(Path("/tmp/cached-model"), result)
        self.assertEqual(
            [{"repo_id": "test/model", "local_files_only": True}], calls
        )

    def test_check_only_writes_manifest_only_when_all_components_are_valid(self) -> None:
        """Validate all four local components and refuse a failed manifest."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            omnivoice = root / "omnivoice"
            audio_tokenizer = root / "audio_tokenizer"
            funasr = root / "funasr"
            whisperx = root / "whisperx"
            for directory in (omnivoice, audio_tokenizer, funasr, whisperx):
                directory.mkdir()
            for name in ("config.json", "tokenizer_config.json"):
                (omnivoice / name).write_text("{}", encoding="utf-8")
            (omnivoice / "model.safetensors").write_bytes(b"weights")
            for name in ("config.json", "preprocessor_config.json"):
                (audio_tokenizer / name).write_text("{}", encoding="utf-8")
            (audio_tokenizer / "model.safetensors").write_bytes(b"weights")
            (funasr / "model.pt").write_bytes(b"weights")
            for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
                (whisperx / name).write_text("{}", encoding="utf-8")
            (whisperx / "model.bin").write_bytes(b"weights")
            manifest_path = root / "manifest.json"
            args = parse_args(
                [
                    "--check-only",
                    "--omnivoice-path", str(omnivoice),
                    "--audio-tokenizer-path", str(audio_tokenizer),
                    "--funasr-model-path", str(funasr),
                    "--whisperx-model-path", str(whisperx),
                    "--output-manifest", str(manifest_path),
                ]
            )

            self.assertEqual(0, prepare_cache(args))
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(json.loads(manifest_path.read_text())["ok"])

            (funasr / "model.pt").unlink()
            failed_manifest = root / "failed.json"
            args = parse_args(
                [
                    "--check-only",
                    "--omnivoice-path", str(omnivoice),
                    "--audio-tokenizer-path", str(audio_tokenizer),
                    "--funasr-model-path", str(funasr),
                    "--whisperx-model-path", str(whisperx),
                    "--output-manifest", str(failed_manifest),
                ]
            )

            self.assertEqual(1, prepare_cache(args))
            self.assertFalse(failed_manifest.exists())


if __name__ == "__main__":
    unittest.main()
