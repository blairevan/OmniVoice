# OmniVoice Connect Full Parity Implementation Plan

> **For agentic workers:** Execute each task with tests before proceeding. Do not commit in this working tree.

**Goal:** Replace the current partial `connect` implementation with the full IndexTTS-equivalent contract.

**Architecture:** Use one `ForcedSegment` model across normal synthesis and source-audio regeneration. Keep parsing, candidate evaluation, waveform processing, runtime orchestration and timeline publication independent and testable.

## Task 1: Replace Markup parsing with token-aware forced segments

**Files:** `omnivoice/utils/connect_markup.py`, `tests/test_connect_markup.py`

- [x] Implement standard and 26071300 tokenization for plain text, pause, replace, pronunciation and connect tags.
- [x] Preserve `alignment_text` and nullable source offsets for pronunciation overrides.
- [x] Pre-split only between tokens; never split connect; support unequal synthesis/display lengths.
- [x] Keep pause exclusively in `parse_text_actions`; ForcedSegment contains text only.
- [x] Allow one oversized connect token and reject only when the explicit CLI limit is exceeded.
- [x] Obtain a finite model-exposed tokenizer/config limit when available; otherwise validate explicit `--max_forced_segment_tokens` only.
- [x] Route subtitle cleaning through the unified parsed segment data.
- [x] Pre-split ordinary punctuation/length units and long markup text while retaining the single-candidate fast path for units without connect.
- [x] Test mixed replace/pronunciation/connect and display-length limits.

## Task 2: Port pure candidate quality evaluation

**Files:** `omnivoice/utils/connect_candidate_selector.py`, `tests/test_connect_candidate_selector.py`

- [x] Define typed candidate scores and aligned connect gaps.
- [x] Compute original-only duration/RMS medians before quality validation.
- [x] Reject silence, clipping, non-finite audio, alignment overflow, duration/RMS outliers and hard max gap violations.
- [x] Score accepted versions with speed-corrected gaps and original tie-break.
- [x] Add candidate seed derivation, waveform hashing, duplicate-candidate removal and rejection evidence; generation failures skip only the failed candidate.
- [x] Expose `--connect_seed` and isolate CPU/CUDA RNG state; server-side model acceptance still remains pending.

## Task 3: Port guarded multichannel waveform processing

**Files:** `omnivoice/utils/connect_waveform_processor.py`, `tests/test_connect_waveform_processor.py`

- [ ] Implement short-window RMS run detection and local voiced reference calculation.
- [ ] Implement guarded zero-crossing selection and equal-power multichannel splice.
- [ ] Test no edit for voiced/short gaps, right-to-left multiple boundaries, sample accounting, finite samples and channel preservation.

## Task 4: Rebuild candidate pipeline and debug evidence

**Files:** `omnivoice/utils/connect_candidate_pipeline.py`, `tests/test_connect_candidate_pipeline.py`

- [ ] Generate all originals first and persist every original alignment result.
- [ ] Create a temporary mono alignment WAV with explicit sample-rate policy while retaining original channel layout for final waveform operations.
- [ ] Collect metrics for every original before alignment/max-gap filtering; calculate original-only medians once.
- [ ] Process only accepted originals; realign and quality-check processed versions.
- [x] Persist every failure/rejection in per-candidate alignment evidence and `selection.json`.
- [ ] Create exact `replacement/action/segment` paths; test candidate failure does not abort other candidates.

## Task 5: Adapt CLI and both timeline paths

**Files:** `omnivoice/cli/cli_infer.py`, `tests/test_connect_cli_integration.py`, `tests/test_audio_segment_regenerator.py`

- [x] Pass forced segments and speech speed to one pipeline in normal and regeneration modes.
- [x] Disable OmniVoice internal automatic audio chunking for every forced segment (`audio_chunk_duration=0`).
- [x] Pass `postprocess_output=False`, zero padding and zero fade; covered by a fake-model test.
- [x] Derive one subtitle per selected forced segment from actual samples.
- [x] Apply boundary silence only to full output and synchronize SRT/JSON.
- [x] Re-open final encoded audio and reject a timeline beyond actual duration.
- [x] Scale timestamps by the actual post-speed frame ratio before boundary silence.
- [x] Make audio publication return the actual output path and cover MP3 fallback.
- [x] Force `boundary_silence=0` for regeneration replacement invocation.
- [x] Reject an existing explicit connect debug directory and write collision-safe evidence paths.
- [x] Validate connect CLI options and explicit forced-segment limits before generation.

## Task 6: Documentation and acceptance

**Files:** `README.md`, `docs/connect_conservative_waveform_processing_zh.md`

- [x] Document optional dependency installation, CLI contract, debug layout and rollback rules.
- [x] Add the `connect` optional dependency group and lock update; FunASR remains a lazy import.
- [x] Run full local unit, compile, CLI help, lock and diff checks.
- [x] Record GPU/FunASR/listening validation as server-side pending work.
- [x] Add tests for multichannel processing, candidate evidence, actual forced kwargs, MP3 fallback and parser defaults; real model acceptance remains intentionally unrun here.
