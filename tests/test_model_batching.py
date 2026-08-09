"""Batch input normalization tests for OmniVoice inference."""

import unittest
from unittest.mock import patch

import torch

from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt


class _DurationEstimator:
    """Return a deterministic token estimate without loading model assets."""

    def estimate_duration(self, *_args, **_kwargs):
        return 25


class OmniVoiceBatchInputTests(unittest.TestCase):
    """Verify scalar/list voice-clone inputs broadcast across a batch."""

    def _model(self) -> OmniVoice:
        model = OmniVoice.__new__(OmniVoice)
        torch.nn.Module.__init__(model)
        model.duration_estimator = _DurationEstimator()
        return model

    def test_scalar_ref_text_broadcasts_to_each_ref_audio(self) -> None:
        """Do not silently reuse the first prompt when ref_text is scalar."""
        model = self._model()
        calls: list[tuple[object, object]] = []

        def create_prompt(_self, ref_audio, ref_text=None, preprocess_prompt=True):
            del preprocess_prompt
            calls.append((ref_audio, ref_text))
            return VoiceClonePrompt(torch.zeros((8, 1), dtype=torch.long), ref_text, 0.1)

        with patch.object(OmniVoice, "create_voice_clone_prompt", create_prompt):
            task = model._preprocess_all(
                text=["第一条", "第二条"],
                ref_audio=["voice-a.wav", "voice-b.wav"],
                ref_text="同一参考文本",
            )

        self.assertEqual(
            calls,
            [("voice-a.wav", "同一参考文本"), ("voice-b.wav", "同一参考文本")],
        )
        self.assertEqual(task.ref_texts, ["同一参考文本", "同一参考文本"])

    def test_scalar_ref_audio_broadcasts_to_each_ref_text(self) -> None:
        """Allow one reference audio to pair with per-item reference texts."""
        model = self._model()
        calls: list[tuple[object, object]] = []

        def create_prompt(_self, ref_audio, ref_text=None, preprocess_prompt=True):
            del preprocess_prompt
            calls.append((ref_audio, ref_text))
            return VoiceClonePrompt(torch.zeros((8, 1), dtype=torch.long), ref_text, 0.1)

        with patch.object(OmniVoice, "create_voice_clone_prompt", create_prompt):
            task = model._preprocess_all(
                text=["第一条", "第二条"],
                ref_audio="shared.wav",
                ref_text=["参考一", "参考二"],
            )

        self.assertEqual(
            calls,
            [("shared.wav", "参考一"), ("shared.wav", "参考二")],
        )
        self.assertEqual(task.ref_texts, ["参考一", "参考二"])


if __name__ == "__main__":
    unittest.main()
