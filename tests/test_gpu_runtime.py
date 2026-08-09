"""Tests for explicit TTS GPU runtime release."""

import unittest
from unittest.mock import patch

from omnivoice.utils.gpu_runtime import release_tts_gpu_runtime


class _Component:
    """Record device moves for lifecycle tests."""

    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.device = "cuda"

    def to(self, device: str) -> None:
        """Move the fake component to a device."""
        self.events.append(f"{self.name}:{device}")
        self.device = device


class _Model(_Component):
    """Expose OmniVoice-owned child components."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(events, "model")
        self.audio_tokenizer = _Component(events, "audio_tokenizer")
        self._asr_pipe = _Component(events, "asr")


class GpuRuntimeTests(unittest.TestCase):
    """Verify release ordering and failure result semantics."""

    def test_release_moves_all_components_before_clearing_cache(self) -> None:
        """Release child components before the parent and CUDA cache."""
        events: list[str] = []
        model = _Model(events)
        with patch("omnivoice.utils.gpu_runtime.torch.cuda.is_available", return_value=True), patch(
            "omnivoice.utils.gpu_runtime.torch.cuda.memory_allocated", return_value=0
        ), patch("omnivoice.utils.gpu_runtime.torch.cuda.synchronize") as synchronize, patch(
            "omnivoice.utils.gpu_runtime.torch.cuda.empty_cache"
        ) as empty_cache:
            result = release_tts_gpu_runtime(model)

        self.assertTrue(result.released)
        self.assertEqual(
            ["audio_tokenizer:cpu", "asr:cpu", "model:cpu"],
            events,
        )
        self.assertEqual(2, synchronize.call_count)
        empty_cache.assert_called_once()

    def test_cuda_unavailable_blocks_whisperx_startup(self) -> None:
        """Treat absent CUDA as a downgrade condition rather than a successful release."""
        model = _Model([])
        with patch(
            "omnivoice.utils.gpu_runtime.torch.cuda.is_available", return_value=False
        ):
            result = release_tts_gpu_runtime(model)

        self.assertFalse(result.released)
        self.assertIn("unavailable", result.reason or "")

    def test_release_reports_residual_memory_without_raising(self) -> None:
        """Return a diagnostic failure when CUDA memory remains allocated."""
        model = _Model([])
        with patch("omnivoice.utils.gpu_runtime.torch.cuda.is_available", return_value=True), patch(
            "omnivoice.utils.gpu_runtime.torch.cuda.memory_allocated", return_value=200_000_000
        ), patch("omnivoice.utils.gpu_runtime.torch.cuda.synchronize"), patch(
            "omnivoice.utils.gpu_runtime.torch.cuda.empty_cache"
        ):
            result = release_tts_gpu_runtime(model)

        self.assertFalse(result.released)
        self.assertIn("allocated", result.reason or "")


if __name__ == "__main__":
    unittest.main()
