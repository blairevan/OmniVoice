"""Run Chinese connect candidates with optional FunASR alignment."""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib
import math
from typing import Callable
import tempfile
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from omnivoice.utils.connect_markup import ForcedSegment
from omnivoice.utils.connect_candidate_selector import CharacterTimestamp, validate_alignment, internal_boundaries, CandidateScore
from omnivoice.utils.connect_waveform_processor import ConnectProcessingOptions, measure_low_energy_pauses, process_connect_waveform

Aligner = Callable[[str, str], tuple[CharacterTimestamp, ...]]
Generator = Callable[[], np.ndarray]

@dataclass(frozen=True)
class ConnectRuntimeOptions:
    """Describe public connect runtime switches."""
    candidates: int = 3
    processing: str = "conservative"
    aligner_device: str = "cpu"
    maximum_shorten_ms: float = 120.0
    debug_dir: str | None = None
    aligner_model: str = "fa-zh"
    max_gap_ms: float | None = None
    speech_speed: float = 1.0
    seed: int | None = None
    max_forced_segment_tokens: int | None = None

@dataclass(frozen=True)
class WaveformSelection:
    """Return selected waveform and its deterministic provenance."""
    waveform: np.ndarray
    candidate_index: int
    version: str
    removed_samples: int

class FunASRAligner:
    """Lazily load fa-zh only when a connect action needs alignment."""
    def __init__(self, device: str, model: str = "fa-zh") -> None: self._device, self._model_name, self._model = device, model, None
    def __call__(self, audio_path: str, text: str) -> tuple[CharacterTimestamp, ...]:
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as error:
                raise RuntimeError("connect requires FunASR; install omnivoice[connect]") from error
            self._model = AutoModel(
                model=self._model_name,
                model_revision="v2.0.4",
                device=self._device,
                disable_update=True,
            )
        waveform, sample_rate = sf.read(audio_path, always_2d=True)
        mono = waveform.mean(axis=1).astype(np.float32)
        if sample_rate != 16000:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000)
        with tempfile.NamedTemporaryFile(prefix="omnivoice-fa-zh-", suffix=".wav") as alignment_file:
            sf.write(alignment_file.name, mono, 16000)
            alignment_text = " ".join(char for char in text if "\u3400" <= char <= "\u9fff")
            result = self._model.generate(input=(alignment_file.name, alignment_text), data_type=("sound", "text"))
        return _normalize_funasr_timestamps(result, text)


def _normalize_funasr_timestamps(
    result: object,
    transcript: str,
) -> tuple[CharacterTimestamp, ...]:
    """Normalize FunASR timestamp/value output to strict CJK character bounds."""
    if not isinstance(result, list) or not result or not isinstance(result[0], dict):
        raise ValueError("FunASR timestamp output is missing a result object")
    raw_items = result[0].get("timestamp")
    if raw_items is None:
        raw_items = result[0].get("value")
    characters = [char for char in transcript if "\u3400" <= char <= "\u9fff"]
    if not isinstance(raw_items, list) or len(raw_items) != len(characters):
        raise ValueError("FunASR timestamp output does not cover the normalized transcript")
    timestamps: list[CharacterTimestamp] = []
    previous_end = 0.0
    for character, raw_item in zip(characters, raw_items):
        if not isinstance(raw_item, (list, tuple)) or len(raw_item) < 2:
            raise ValueError("FunASR timestamp output contains an invalid item")
        start_ms = float(raw_item[-2])
        end_ms = float(raw_item[-1])
        if not math.isfinite(start_ms) or not math.isfinite(end_ms) or start_ms < previous_end or end_ms <= start_ms:
            raise ValueError("FunASR timestamp output is not monotonic")
        timestamps.append(CharacterTimestamp(character, start_ms, end_ms))
        previous_end = end_ms
    return tuple(timestamps)


def _waveform_quality_reasons(
    waveform: np.ndarray,
    sample_rate: int,
    duration_median: float,
    rms_median: float,
) -> list[str]:
    """Return hard audio-quality rejection reasons for one candidate."""
    duration = waveform.shape[-1] / sample_rate
    peak = float(np.max(np.abs(waveform)))
    rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(waveform)))), 1e-12))
    clipped = float(np.mean(np.abs(waveform) >= 0.999))
    reasons: list[str] = []
    if not np.isfinite(waveform).all():
        reasons.append("non_finite")
    if peak < 1e-4 or rms_db < -100.0:
        reasons.append("silent")
    if clipped > 0.001:
        reasons.append("clipping")
    if abs(duration - duration_median) / duration_median > 0.20:
        reasons.append("duration_outlier")
    if abs(rms_db - rms_median) > 6.0:
        reasons.append("rms_outlier")
    return reasons


def select_connect_waveform(markup: ForcedSegment, sample_rate: int, generate: Generator, align: Aligner, options: ConnectRuntimeOptions, processing_options: ConnectProcessingOptions | None) -> WaveformSelection:
    """Generate originals, retain safe processed variants, and select the best."""
    if not 1 <= options.candidates <= 5 or options.processing not in {"conservative", "off"}:
        raise ValueError("connect candidates must be between 1 and 5 and processing must be conservative or off")
    if not np.isfinite(options.speech_speed) or options.speech_speed <= 0:
        raise ValueError("connect speech speed must be positive and finite")
    if options.max_forced_segment_tokens is not None and options.max_forced_segment_tokens <= 0:
        raise ValueError("max forced segment tokens must be positive")
    if options.max_forced_segment_tokens is not None and len(markup.synthesis_text) > options.max_forced_segment_tokens:
        raise ValueError("forced segment exceeds configured token limit")
    generated_waveforms: dict[int, np.ndarray] = {}
    generation_errors: dict[int, str] = {}
    for index in range(1, options.candidates + 1):
        try:
            if options.seed is None:
                original = generate().astype(np.float32, copy=False)
            else:
                try:
                    import torch
                    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(options.seed + index - 1)
                        if cuda_devices:
                            torch.cuda.manual_seed_all(options.seed + index - 1)
                        original = generate().astype(np.float32, copy=False)
                except ImportError:
                    original = generate().astype(np.float32, copy=False)
            if original.ndim not in {1, 2} or original.size == 0 or not np.isfinite(original).all():
                raise ValueError("generated waveform is empty, non-finite, or has an invalid rank")
            generated_waveforms[index] = original
        except Exception as error:
            generation_errors[index] = str(error)
    if not generated_waveforms:
        raise RuntimeError("connect generated no valid candidate waveform")
    original_durations = [waveform.shape[-1] / sample_rate for waveform in generated_waveforms.values()]
    original_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(waveform)))), 1e-12)) for waveform in generated_waveforms.values()]
    duration_median = float(np.median(original_durations))
    rms_median = float(np.median(original_rms))
    versions: list[tuple[CandidateScore, np.ndarray, int]] = []
    candidate_reports: list[dict[str, object]] = []
    all_original_waveforms: list[np.ndarray] = []
    original_hashes: set[str] = set()
    if options.debug_dir is not None and Path(options.debug_dir).exists():
        raise FileExistsError(f"connect debug directory already exists: {options.debug_dir}")
    debug_context = (
        nullcontext(options.debug_dir)
        if options.debug_dir is not None
        else tempfile.TemporaryDirectory(prefix="omnivoice-connect-")
    )
    with debug_context as directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        for index in range(1, options.candidates + 1):
            report: dict[str, object] = {
                "candidateIndex": index,
                "seed": None if options.seed is None else options.seed + index - 1,
                "accepted": False,
                "rejections": [],
            }
            if index in generation_errors:
                report["generationError"] = generation_errors[index]
                report["rejections"].append("generation")
                candidate_reports.append(report)
                continue
            original = generated_waveforms[index]
            report["originalSha256"] = hashlib.sha256(original.tobytes()).hexdigest()
            report["originalDurationMs"] = original.shape[-1] * 1000 / sample_rate
            path = Path(directory) / f"candidate_{index}_original.wav"
            sf.write(path, original.T if original.ndim == 2 else original, sample_rate)
            try:
                alignment = validate_alignment(markup, align(str(path), markup.alignment_text or markup.synthesis_text), original.shape[-1] * 1000 / sample_rate)
                boundaries = internal_boundaries(markup, alignment)
                alignment_error = None
            except Exception as error:
                alignment, boundaries, alignment_error = (), (), str(error)
            (Path(directory) / f"candidate_{index:03d}_original_alignment.json").write_text(
                json.dumps({"candidateIndex": index, "version": "original", "error": alignment_error, "timestamps": [item.__dict__ for item in alignment]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if alignment_error is not None:
                report["alignmentError"] = alignment_error
                report["rejections"].append("alignment")
                candidate_reports.append(report)
                continue
            waveform_hash = str(report["originalSha256"])
            if waveform_hash in original_hashes:
                report["rejections"].append("duplicate_original")
                candidate_reports.append(report)
                continue
            original_hashes.add(waveform_hash)
            gaps = [max(0.0, right.start_ms-left.end_ms) / options.speech_speed for left, right in boundaries]
            if options.max_gap_ms is not None and max(gaps, default=0.0) > options.max_gap_ms:
                report["maxGapMs"] = max(gaps, default=0.0)
                report["rejections"].append("max_gap")
                candidate_reports.append(report)
                continue
            quality_reasons = _waveform_quality_reasons(original, sample_rate, duration_median, rms_median)
            if quality_reasons:
                report["rejections"].extend(f"original_{reason}" for reason in quality_reasons)
                candidate_reports.append(report)
                continue
            pause_max, pause_total = measure_low_energy_pauses(original, sample_rate, boundaries, processing_options) if processing_options else (0.0, 0.0)
            versions.append((CandidateScore(index, "original", pause_max, pause_total, max(gaps, default=0), sum(gaps), 0), original, 0))
            if options.processing == "conservative":
                if processing_options is None:
                    raise RuntimeError("conservative processing options are required")
                processed = process_connect_waveform(original, sample_rate, boundaries, processing_options)
                (Path(directory) / f"candidate_{index:03d}_edits.json").write_text(
                    json.dumps({"candidateIndex": index, "totalRemovedSamples": processed.total_removed_samples, "edits": [item.__dict__ for item in processed.edits]}, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if processed.total_removed_samples:
                    try:
                        processed_path = Path(directory) / f"candidate_{index}_processed.wav"; sf.write(processed_path, processed.waveform.T if processed.waveform.ndim == 2 else processed.waveform, sample_rate)
                        processed_alignment = validate_alignment(
                            markup,
                            align(str(processed_path), markup.alignment_text or markup.synthesis_text),
                            processed.waveform.shape[-1] * 1000 / sample_rate,
                        )
                        processed_boundaries = internal_boundaries(markup, processed_alignment)
                        (Path(directory) / f"candidate_{index:03d}_processed_alignment.json").write_text(
                            json.dumps({"candidateIndex": index, "version": "processed", "error": None, "timestamps": [item.__dict__ for item in processed_alignment]}, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        processed_gaps = [
                            max(0.0, right.start_ms - left.end_ms) / options.speech_speed
                            for left, right in processed_boundaries
                        ]
                        pause_max, pause_total = measure_low_energy_pauses(processed.waveform, sample_rate, processed_boundaries, processing_options)
                        versions.append((
                            CandidateScore(
                                index, "processed", pause_max,
                                pause_total, max(processed_gaps, default=0),
                                sum(processed_gaps), processed.total_removed_samples,
                            ), processed.waveform, processed.total_removed_samples,
                        ))
                        report["processedAccepted"] = True
                        report["processedMaxGapMs"] = max(processed_gaps, default=0.0)
                        report["processedTotalGapMs"] = sum(processed_gaps)
                    except Exception as error:
                        report.setdefault("processedRejections", []).append(str(error))
                        (Path(directory) / f"candidate_{index:03d}_processed_alignment.json").write_text(
                            json.dumps({"candidateIndex": index, "version": "processed", "error": str(error), "timestamps": []}, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                else:
                    report["processedRejections"] = ["no_safe_low_energy_edit"]
            report["acceptedOriginal"] = True
            report["originalMaxGapMs"] = max(gaps, default=0.0)
            report["originalTotalGapMs"] = sum(gaps)
            candidate_reports.append(report)
    originals = [item for item in versions if item[0].version == "original"]
    if not originals:
        details = []
        for report in candidate_reports:
            rejection = report.get("rejections", [])
            if rejection:
                detail = f"candidate {report.get('candidateIndex')}: {', '.join(str(item) for item in rejection)}"
                if report.get("alignmentError"):
                    detail += f" ({report['alignmentError']})"
                details.append(detail)
        suffix = "; ".join(details) if details else "no candidate evidence"
        raise RuntimeError(f"connect has no valid original candidate: {suffix}")
    accepted_versions = []
    for item in versions:
        waveform = item[1]
        duration = waveform.shape[-1] / sample_rate
        peak = float(np.max(np.abs(waveform)))
        rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(waveform)))), 1e-12))
        clipped = float(np.mean(np.abs(waveform) >= 0.999))
        reasons: list[str] = []
        if not np.isfinite(waveform).all():
            reasons.append("non_finite")
        if peak < 1e-4 or rms_db < -100.0:
            reasons.append("silent")
        if clipped > 0.001:
            reasons.append("clipping")
        if reasons:
            for candidate_report in candidate_reports:
                if candidate_report["candidateIndex"] == item[0].candidate_index:
                    candidate_report["rejections"].extend(f"{item[0].version}_{reason}" for reason in reasons)
            continue
        if abs(duration - duration_median) / duration_median > 0.20:
            reasons.append("duration_outlier")
        if abs(rms_db - rms_median) > 6.0:
            reasons.append("rms_outlier")
        if reasons:
            for candidate_report in candidate_reports:
                if candidate_report["candidateIndex"] == item[0].candidate_index:
                    candidate_report["rejections"].extend(f"{item[0].version}_{reason}" for reason in reasons)
            continue
        for candidate_report in candidate_reports:
            if candidate_report["candidateIndex"] == item[0].candidate_index:
                candidate_report["accepted"] = True
        accepted_versions.append(item)
    versions = accepted_versions
    if not versions:
        raise RuntimeError("connect has no acceptable candidate after audio quality validation")
    score, waveform, removed = min(versions, key=lambda item: item[0].key())
    if options.debug_dir is not None:
        (Path(options.debug_dir) / "selection.json").write_text(
            json.dumps({
                "selectedCandidateIndex": score.candidate_index,
                "selectedVersion": score.version,
                "totalRemovedSamples": removed,
                "synthesisText": markup.synthesis_text,
                "subtitleText": markup.subtitle_text,
                "alignmentText": markup.alignment_text,
                "connectRanges": [item.__dict__ for item in markup.ranges],
                "processingOptions": processing_options.__dict__ if processing_options else None,
                "seed": options.seed,
                "candidateSeeds": [None if options.seed is None else options.seed + index - 1 for index in range(1, options.candidates + 1)],
                "candidates": [item[0].__dict__ for item in versions],
                "candidateReports": candidate_reports,
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return WaveformSelection(waveform, score.candidate_index, score.version, removed)
