"""Explicitly release OmniVoice GPU components before WhisperX CUDA."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuReleaseResult:
    """Describe whether the TTS runtime released enough GPU memory."""

    released: bool
    allocated_bytes_after: int
    reason: str | None


def _move_component_to_cpu(component: object, name: str) -> None:
    """Move one model-owned component to CPU when it exposes a device API."""
    if component is None:
        return
    mover = getattr(component, "to", None)
    if not callable(mover):
        return
    try:
        mover("cpu")
    except Exception as error:
        raise RuntimeError(f"Unable to move {name} to CPU") from error


def release_tts_gpu_runtime(
    model: object,
    maximum_allocated_bytes: int = 134_217_728,
) -> GpuReleaseResult:
    """Release TTS components and verify process-local CUDA allocation."""
    if maximum_allocated_bytes < 0:
        raise ValueError("maximum_allocated_bytes must not be negative")
    if not torch.cuda.is_available():
        return GpuReleaseResult(False, 0, "CUDA is unavailable for WhisperX alignment")

    try:
        torch.cuda.synchronize()
        _move_component_to_cpu(getattr(model, "audio_tokenizer", None), "audio_tokenizer")
        _move_component_to_cpu(getattr(model, "_asr_pipe", None), "ASR pipeline")
        _move_component_to_cpu(model, "OmniVoice model")
        if hasattr(model, "_asr_pipe"):
            setattr(model, "_asr_pipe", None)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = int(torch.cuda.memory_allocated())
    except Exception as error:
        return GpuReleaseResult(False, int(torch.cuda.memory_allocated()), str(error))

    if allocated > maximum_allocated_bytes:
        return GpuReleaseResult(
            False,
            allocated,
            f"CUDA memory remains allocated: {allocated} bytes",
        )
    return GpuReleaseResult(True, allocated, None)


__all__ = ["GpuReleaseResult", "release_tts_gpu_runtime"]
