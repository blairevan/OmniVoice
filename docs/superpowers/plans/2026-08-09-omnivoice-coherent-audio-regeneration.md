# OmniVoice Coherent Audio and Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring OmniVoice normal synthesis and subtitle-aligned regeneration to parity with the approved IndexTTS coherence design, including coherent sentence groups, shared offline WhisperX alignment, natural-duration timelines, and 15ms overlap-add splicing.

**Architecture:** Keep OmniVoice markup and FunASR `connect` processing as the model-specific adapter. Add pure timeline/grouping, sentence timing, GPU lifecycle, and synthesis-orchestration modules, a subprocess-based WhisperX adapter for the shared `/opt/app/aining/digital_human/whisperx` runtime, and a sample-index splicer. TTS generation runs to completion first; the TTS model is then moved off GPU and only afterward are all multi-sentence groups aligned with WhisperX CUDA under one task-wide deadline.

**Tech Stack:** Python 3.10+, NumPy, Torch, SoundFile, Librosa, pydub, `unittest`, `uv`, shared WhisperX 3.7.4 runtime.

## Global Constraints

- Do not add WhisperX to OmniVoice's default dependencies; invoke `/opt/app/aining/digital_human/whisperx` with `uv run --project`.
- WhisperX defaults are runtime `/opt/app/aining/digital_human/whisperx`, model `/opt/app/aining/digital_human/whisperx/models/zh`, language `zh`, and device `cuda`; use local model files, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
- The 10800-second WhisperX timeout is a task-wide monotonic deadline. Every subprocess receives only the remaining task time; once exhausted, all pending groups use subtitle timing fallback.
- WhisperX alignment coverage uses the approved 95% threshold. Match normalized alignment units with a deterministic order-preserving sequence matcher that handles repeated characters; interpolate only missing runs bounded by reliable neighbors, and force whole-group fallback when any sentence has no reliable unit.
- WhisperX CUDA OOM, timeout, process failure, or inability to release TTS GPU memory falls back to pronunciation-weight subtitle timing and never retries on CPU.
- The formal regeneration request is `[start, end, text|string[], speechSpeed]`; accept the legacy three-item request only as an explicit compatibility parse.
- Adjacent regeneration requests are normalized into one coherent group only when their complete subtitle coverage is adjacent and their per-request `speechSpeed` values match. Voice, style, language, and model parameters are task-global immutable CLI values.
- A regeneration request's `speechSpeed` applies exactly once as post-synthesis group-level time stretch. Do not pass it to `model.generate()`, and do not apply the global `--speed` a second time.
- Internal timing uses integer sample indices; SRT/JSON conversion happens only at serialization.
- Preserve current `connect` FunASR candidate selection and conservative waveform processing; do not use its internal boundaries as visible subtitle boundaries.
- Preserve unrelated worktree changes and do not run destructive Git commands. Do not create commits unless the user explicitly requests them.
- Pure tests and fake TTS do not count as GPU, codec, or listening acceptance.

---

## File Map

Create the following focused modules:

- `omnivoice/utils/coherent_timeline.py`: normalized subtitle units, three text layers, coherent group packing, and group merging.
- `omnivoice/utils/sentence_timing.py`: WhisperX character-to-sentence mapping, 95% coverage, interpolation, and pronunciation-weight fallback.
- `omnivoice/utils/whisperx_alignment.py`: shared-runtime subprocess adapter, offline environment, local model preflight, and timeout handling.
- `omnivoice/utils/gpu_runtime.py`: explicit TTS model/audio-tokenizer release before WhisperX CUDA.
- `omnivoice/utils/synthesis_orchestrator.py`: generated-group records, two-phase TTS/alignment flow, and final subtitle assembly.

Modify these existing modules:

- `omnivoice/utils/segment_timeline.py`: four-item request parsing, speech speed, adjacent-group normalization, and integer-sample subtitle validation.
- `omnivoice/utils/audio_segment_regenerator.py`: all-group generation, 15ms overlap-add, sentence timing, group metadata, and safe publication.
- `omnivoice/cli/cli_infer.py`: CLI options, argument validation, thin invocation of the orchestrator, and final boundary-silence behavior.
- `omnivoice/utils/connect_markup.py`: expose the exact mappings needed by coherent groups without changing current markup semantics.
- `tests/test_audio_segment_regenerator.py`: update existing contracts and add multi-group/overlap tests.
- `tests/test_connect_cli_integration.py`: preserve connect and normal CLI contracts while adding coherent-group assertions.

Create these tests:

- `tests/test_coherent_timeline.py`
- `tests/test_sentence_timing.py`
- `tests/test_whisperx_alignment.py`
- `tests/test_gpu_runtime.py`

Update documentation:

- `README.md`: document the four-item regeneration request and WhisperX options.
- `docs/coherent_audio_regeneration_zh.md`: document runtime requirements, two-phase GPU lifecycle, fallback semantics, and server acceptance commands.

## Task 1: Lock the normalized text and coherent-group contracts

**Files:**
- Create: `omnivoice/utils/coherent_timeline.py`
- Modify: `omnivoice/utils/connect_markup.py`
- Test: `tests/test_coherent_timeline.py`

**Interfaces:**
- `SubtitleUnit`: frozen data object containing `sentence_id`, `display_text`, `synthesis_text`, `alignment_text`, `alignment_units`, `explicit_pause_ms`, and mapping ranges.
- `CoherentSynthesisGroup`: frozen data object containing ordered `units`, merged `synthesis_text`, merged `alignment_text`, merged offsets, and connect ranges.
- `build_coherent_synthesis_groups(units, tokenizer, max_tokens, max_synthesis_chars) -> tuple[CoherentSynthesisGroup, ...]`.
- `merge_coherent_units(units) -> CoherentSynthesisGroup`.

- [ ] **Step 1: Write failing mapping tests**

  Cover ordinary Chinese sentences, `replace`, pronunciation markup, `connect`, numbers/percentages, explicit pauses, and the invariant that display text is never used as the synthesis index space. Assert that all ranges are half-open and that sentence IDs remain ordered and unique.

- [ ] **Step 2: Run the focused tests and verify failure**

  Run: `uv run python -m unittest tests.test_coherent_timeline -v`

  Expected: FAIL because the new module and contracts are not implemented.

- [ ] **Step 3: Implement normalized units and group merging**

  Reuse `parse_connect_markup()` for markup semantics. Preserve separate display/synthesis/alignment strings and source offsets. Merge adjacent units by concatenating their mapped fields and offsetting all ranges; never silently insert user-visible text. Keep `connect` spans in the merged group so the existing FunASR selector can process them.

- [ ] **Step 4: Implement tokenizer-bounded grouping**

  Pack adjacent units until adding the next unit would exceed `max_tokens` when a tokenizer is available, otherwise `max_synthesis_chars`. Split only at unit boundaries. Raise a descriptive error when one unit itself exceeds the limit; do not truncate.

- [ ] **Step 5: Run the focused tests and verify success**

  Run: `uv run python -m unittest tests.test_coherent_timeline -v`

  Expected: PASS, including tests proving that a group contains one merged synthesis string and that connect spans do not become visible subtitle splits.

## Task 2: Align the regeneration request and group-normalization contracts

**Files:**
- Modify: `omnivoice/utils/segment_timeline.py`
- Modify: `omnivoice/cli/cli_infer.py`
- Test: `tests/test_audio_segment_regenerator.py`

**Interfaces:**
- `ReplacementRequest(start: float, end: float, text: str | list[str], speech_speed: float)`.
- `parse_replacements(value: str, legacy_speed: float | None = None) -> list[ReplacementRequest]`.
- `normalize_replacement_requests(subtitles, requests) -> list[ReplacementRequest]`.

- [ ] **Step 1: Add failing protocol tests**

  Assert that the formal parser accepts `[["00:00:01.000", "00:00:03.000", ["句一", "句二"], 1.1]]`, rejects missing or non-positive `speechSpeed`, rejects a text array whose length differs from covered subtitles, and maps the legacy three-item request to the supplied global speed.

- [ ] **Step 2: Add failing adjacency tests**

  Create three ordered subtitles and assert that adjacent requests with identical `speechSpeed` normalize into one request with a text array; non-adjacent requests remain separate; adjacent requests with different speed are rejected. Voice/style/language are task-global and therefore are not compared per request.

- [ ] **Step 3: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_audio_segment_regenerator.ParserTests tests.test_audio_segment_regenerator.TimelineTests -v`

  Expected: FAIL on the new four-item and normalization assertions.

- [ ] **Step 4: Implement strict parsing and normalization**

  Keep strict `HH:MM:SS.mmm` parsing and complete subtitle-boundary validation. Preserve original request order for source-axis cutting. Store the request's `speech_speed`; use the global speed only to populate legacy three-item input. Merge only complete adjacent subtitle coverage with equal `speech_speed`, preserving replacement texts in source order.

- [ ] **Step 5: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_audio_segment_regenerator.ParserTests tests.test_audio_segment_regenerator.TimelineTests -v`

  Expected: PASS, with old three-item tests retained as compatibility coverage.

## Task 3: Add the shared offline WhisperX adapter

**Files:**
- Create: `omnivoice/utils/whisperx_alignment.py`
- Test: `tests/test_whisperx_alignment.py`
- Modify: `omnivoice/cli/cli_infer.py`

**Interfaces:**
- `AlignedCharacter(character: str, start: float, end: float)`.
- `WhisperXAligner(runtime_dir, language, device, model_name)`.
- `WhisperXAlignmentDeadline.from_timeout_seconds(10800) -> WhisperXAlignmentDeadline`.
- `WhisperXAligner.align(audio_path, text, start, end, deadline) -> tuple[AlignedCharacter, ...]`.

- [ ] **Step 1: Write failing adapter tests**

  Mock `time.monotonic` and `subprocess.run`. Assert the exact command includes `uv run --project`, the shared `align.py`, local model path, language, CUDA device, and interval; assert `timeout` equals the deadline's remaining seconds rather than a fresh 10800 seconds. Assert the subprocess environment contains both offline variables. Add tests for deadline exhaustion before spawning, malformed JSON, missing `segments[0].chars`, non-zero exit, timeout, and missing local runtime/model paths.

- [ ] **Step 2: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_whisperx_alignment -v`

  Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the adapter**

  Preflight `runtime_dir/pyproject.toml`, `runtime_dir/align.py`, and `model_name` before invoking `uv`. Pass an explicit environment copy with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Before each child process, obtain the deadline's positive remaining seconds and use it as `subprocess.run(..., check=True, capture_output=True, text=True, timeout=remaining_seconds)`; if no time remains, raise a typed timeout error without spawning. Convert timeout and process failures into typed alignment errors that the caller records and downgrades. Parse only valid finite `char/start/end` entries and reject an output that does not have exactly one segment.

- [ ] **Step 4: Add CLI options and lazy construction**

  Add `--whisperx_runtime_dir` defaulting to `/opt/app/aining/digital_human/whisperx`, `--whisperx_language` defaulting to `zh`, `--whisperx_device` restricted to `cuda`, `--whisperx_model` defaulting to `/opt/app/aining/digital_human/whisperx/models/zh`, and `--whisperx_timeout_seconds` defaulting to `10800`. Construct the adapter and task deadline only when SRT or JSON output is requested and at least one multi-sentence group is pending.

- [ ] **Step 5: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_whisperx_alignment -v`

  Expected: PASS without importing WhisperX into the OmniVoice process.

## Task 4: Implement sentence timing and the 95% fallback contract

**Files:**
- Create: `omnivoice/utils/sentence_timing.py`
- Test: `tests/test_sentence_timing.py`

**Interfaces:**
- `SentenceTiming(start_sample: int, end_sample: int)`.
- `build_sentence_boundaries(sentence_alignment_texts, characters, sample_rate, group_start, group_end) -> tuple[SentenceTiming, ...]`.
- `build_weighted_sentence_timings(sentence_texts, generated_samples, left_overlap_samples, right_overlap_samples) -> tuple[SentenceTiming, ...]`.
- `resolve_sentence_timings(..., alignment_resolver, coverage_threshold=0.95) -> TimingResolution`.

- [ ] **Step 1: Write failing timing tests**

  Cover full alignment, 95% coverage with monotonic interpolation, coverage below 95%, one sentence with no reliable character, empty/duplicate/out-of-order characters, one-sentence short circuit, overlap clipping, and exact integer-sample closure.

- [ ] **Step 2: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_sentence_timing -v`

  Expected: FAIL because the module is not implemented.

- [ ] **Step 3: Implement strict character mapping and coverage**

  Tokenize both expected and returned values into normalized alignment units, then run a deterministic longest-common-subsequence-style order-preserving matcher that records each expected unit's optional aligned timestamp and works for repeated characters. Compute `reliable_timed_units / total_alignable_units`; interpolate only interior missing runs bounded by valid left/right timestamps when total coverage is at least 0.95. Force whole-group fallback if any sentence has no reliable unit or if the interpolated result is non-monotonic. Never mix a partially aligned sentence with weighted timing from another rule.

- [ ] **Step 4: Implement weighted fallback**

  Count language-appropriate pronunciation units, add explicit pause and punctuation weights, reserve the non-overlap window, and assign the final boundary to the last sentence so rounding closes exactly. Return `timing_method`, `alignment_coverage`, and `fallback_reason` with the timing result.

- [ ] **Step 5: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_sentence_timing -v`

  Expected: PASS for alignment, interpolation, fallback, single-sentence, and sample-closure cases.

## Task 5: Add explicit TTS GPU release before WhisperX CUDA

**Files:**
- Create: `omnivoice/utils/gpu_runtime.py`
- Test: `tests/test_gpu_runtime.py`
- Modify: `omnivoice/cli/cli_infer.py`

**Interfaces:**
- `release_tts_gpu_runtime(model, maximum_allocated_bytes=134217728) -> GpuReleaseResult`.
- `GpuReleaseResult(released: bool, allocated_bytes_after: int, reason: str | None)`.

- [ ] **Step 1: Write failing lifecycle tests**

  Use fake model, audio tokenizer, and ASR components. Assert that model and tokenizer receive `.to("cpu")` or are deleted, `gc.collect()` is called, and `torch.cuda.empty_cache()` is called only after component release. Mock `torch.cuda.memory_allocated()` and assert that allocations above 134217728 bytes return `released=False`; assert component-release failure also returns `released=False` with a reason, without raising away already-generated audio.

- [ ] **Step 2: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_gpu_runtime -v`

  Expected: FAIL because the lifecycle helper does not exist.

- [ ] **Step 3: Implement release and preflight**

  Release `audio_tokenizer`, optional `_asr_pipe`, and the OmniVoice model in that order. Do not claim that `empty_cache()` alone releases model memory. Synchronize CUDA before release, run garbage collection afterward, and inspect process-local `torch.cuda.memory_allocated()`. Return `released=True` only when tracked components no longer report CUDA and allocated memory is at most 134217728 bytes; otherwise return `released=False` with a diagnostic reason.

- [ ] **Step 4: Integrate two-phase orchestration**

  Change normal synthesis and regeneration to collect all generated group WAVs and mappings first. If no multi-sentence group needs alignment, skip GPU release entirely. Otherwise call `release_tts_gpu_runtime(model)` after the last TTS call. When it returns `released=False`, do not start WhisperX CUDA; record `tts_gpu_release_failed` for every pending group and resolve all of them with weighted timing. When it returns `released=True`, run pending WhisperX alignments with CUDA under the single task deadline. On WhisperX OOM/timeout/process error, retain generated audio and resolve only subtitle timing via weighted fallback. Do not restore the TTS model in the same task.

- [ ] **Step 5: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_gpu_runtime tests.test_whisperx_alignment -v`

  Expected: PASS, including the assertion that WhisperX is not invoked before the release hook.

## Task 6: Replace regeneration fades with IndexTTS-style overlap-add

**Files:**
- Modify: `omnivoice/utils/audio_segment_regenerator.py`
- Test: `tests/test_audio_segment_regenerator.py`

**Interfaces:**
- `_choose_overlap_lengths(piece_lengths: Sequence[int], target_samples: int) -> tuple[int, ...]`.
- `_overlap_add(left: np.ndarray, right: np.ndarray, overlap_samples: int) -> np.ndarray`.
- `regenerate_audio_segments(..., alignment_resolver, group_generator) -> RegenerationResult`.

- [ ] **Step 1: Write failing splice tests**

  Assert that two source/replacement boundaries use 15ms equal-power overlap with `fade_out**2 + fade_in**2 == 1` within floating-point tolerance, short pieces reduce overlap safely, output sample count equals piece sums minus actual overlaps, mixed samples are finite, and source slices are taken from original sample positions even after earlier groups change duration.

- [ ] **Step 2: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_audio_segment_regenerator.AudioRegenerationTests -v`

  Expected: FAIL because the current implementation fades pieces and concatenates them without overlap.

- [ ] **Step 3: Implement sample-level overlap-add**

  Port the IndexTTS algorithm to NumPy channels-first arrays: build retained-source and generated-replacement pieces on the original source axis, choose actual overlaps capped by adjacent piece lengths, use equal-power windows, and concatenate with overlap subtraction. Record each piece start and both overlap lengths.

- [ ] **Step 4: Rebuild sentence subtitles on the output axis**

  For each replacement group, use single-sentence duration, WhisperX timing, or weighted timing. Clip group subtitle windows to exclude left/right overlap. Compute `group_delta_samples = generated - original - left_overlap - right_overlap`, then shift only later source subtitles by cumulative delta.

- [ ] **Step 5: Add group metadata and safe publication**

  Emit `sentence_ids`, timing method, coverage, fallback reason, generated samples, overlap samples, and delta samples in JSON. Re-read the encoded output before publishing and validate the final integer-sample timeline.

- [ ] **Step 6: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_audio_segment_regenerator -v`

  Expected: PASS for one group, multiple groups, natural duration changes, overlap closure, subtitle shifts, and failure-without-publication.

## Task 7: Integrate coherent normal synthesis and preserve connect behavior

**Files:**
- Create: `omnivoice/utils/synthesis_orchestrator.py`
- Modify: `omnivoice/cli/cli_infer.py`
- Modify: `omnivoice/utils/connect_markup.py`
- Modify: `tests/test_connect_cli_integration.py`
- Modify: `tests/test_connect_processing.py`

**Interfaces:**
- `GeneratedSynthesisGroup(waveform, sample_rate, units, source_action_index, timing_request)`.
- `generate_coherent_actions(...) -> tuple[GeneratedSynthesisGroup, ...]`.
- `resolve_generated_group_timings(groups, alignment_session) -> tuple[SubtitleItem, ...]`.

- [ ] **Step 1: Add failing integration tests**

  With a fake OmniVoice model, assert that two adjacent subtitle units result in one model generation call, one connect group retains FunASR processing, single sentences skip WhisperX, and all WhisperX calls happen after the GPU release hook. Assert the final `leading_silence` and `trailing_silence` are applied once and shift normal-output subtitles equally only for the leading padding.

- [ ] **Step 2: Run tests to verify failure**

  Run: `uv run python -m unittest tests.test_connect_cli_integration tests.test_connect_processing -v`

  Expected: FAIL on coherent grouping and two-phase lifecycle assertions.

- [ ] **Step 3: Implement normal-mode orchestration**

  Put generated-group records and two-phase orchestration in `synthesis_orchestrator.py`; keep `cli_infer.py` as argument parsing and result publication only. Replace the current per-expanded-action timestamp assignment with generated group records. Keep `pause` as an explicit audio action, preserve the existing connect candidate options, and pass merged connect markup to the existing selector. Generate all group audio first; then conditionally release the TTS runtime and resolve multi-sentence timings.

- [ ] **Step 4: Apply final boundary silence once**

  Keep `leading_silence` and `trailing_silence` at the final normal-output stage only. Add the leading sample offset to every generated subtitle timestamp, and never add either padding inside a replacement group.

- [ ] **Step 5: Run tests to verify success**

  Run: `uv run python -m unittest tests.test_connect_cli_integration tests.test_connect_processing -v`

  Expected: PASS with existing connect evidence and new coherent-group lifecycle assertions.

## Task 8: Wire regeneration CLI, documentation, and complete verification

**Files:**
- Modify: `omnivoice/cli/cli_infer.py`
- Modify: `README.md`
- Create: `docs/coherent_audio_regeneration_zh.md`
- Test: `tests/test_audio_segment_regenerator.py`

- [ ] **Step 1: Integrate request normalization and group generation**

  For each normalized request, call `model.generate()` without a speed override, then apply the request's `speech_speed` exactly once with the existing group-level time-stretch helper before reading its actual sample count. Pass the same speed to `ConnectRuntimeOptions` only for connect candidate evaluation. For a text array, preserve one display/alignment unit per covered source subtitle. Generate every replacement group before the GPU release hook, then align all multi-sentence replacements through the shared adapter.

- [ ] **Step 2: Document the public CLI contract**

  Add examples for the formal four-item request, legacy compatibility, shared WhisperX runtime/model options, `--whisperx_timeout_seconds 10800`, CUDA-only behavior, fallback metadata, and the requirement that source and output paths differ.

- [ ] **Step 3: Run the complete pure test suite**

  Run: `uv run python -m unittest discover -s tests -v`

  Expected: PASS for all existing and new tests.

- [ ] **Step 4: Run syntax, style, and diff checks**

  Run:

  ```bash
  uv run python -m py_compile \
    omnivoice/cli/cli_infer.py \
    omnivoice/utils/coherent_timeline.py \
    omnivoice/utils/sentence_timing.py \
    omnivoice/utils/whisperx_alignment.py \
    omnivoice/utils/gpu_runtime.py \
    omnivoice/utils/segment_timeline.py \
    omnivoice/utils/audio_segment_regenerator.py
  git diff --check
  ```

  Expected: no syntax errors and no whitespace errors. Use the repository's available pycodestyle/ruff command if configured; do not claim GPU validation from these checks.

- [ ] **Step 5: Prepare target-server acceptance**

  Run on the GPU server only after pure tests pass:

  ```bash
  uv run python -c 'from pathlib import Path; print(Path("/opt/app/aining/digital_human/whisperx/pyproject.toml").is_file())'
  uv run python -m unittest tests.test_whisperx_alignment -v
  uv run python omnivoice/cli/cli_infer.py \
    --text "这里是连续句子。这里是第二句。" \
    --voice /data/reference.wav \
    --ref_text "参考音频文本" \
    --output /data/omnivoice_coherent.wav \
    --srt /data/omnivoice_coherent.srt \
    --whisperx_runtime_dir /opt/app/aining/digital_human/whisperx \
    --whisperx_model /opt/app/aining/digital_human/whisperx/models/zh \
    --whisperx_language zh \
    --whisperx_device cuda \
    --whisperx_timeout_seconds 10800
  ```

  Manually record `torch.cuda.memory_allocated()` before TTS, after the TTS-release hook, and immediately before WhisperX starts; after release it must be at most 134217728 bytes or WhisperX must be skipped with `tts_gpu_release_failed`. Verify output audio is decodable, subtitles are aligned and monotonic, and the audio has no missing/repeated sentence. Run a separate local-range regeneration sample and compare the source hash before/after.

- [ ] **Step 6: Preserve the clean commit boundary**

  Do not create commits as part of plan execution. Before editing, record `git status --short` and `git diff --stat`; after all verification, present the scoped diff to the user. Create a Conventional Commit only if the user explicitly requests it.

  ```text
  No automatic staging or commit is permitted by this plan.
  ```

  Never stage or commit files until the user explicitly asks for a commit.
