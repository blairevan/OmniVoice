# OmniVoice Connect Conservative Waveform Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Chinese `[connect:...]` produce verifiable continuous speech through three candidates, FunASR `fa-zh` alignment, conservative low-energy editing, fallback, and real-duration subtitle updates.

**Architecture:** Keep markup parsing, pure scoring, waveform editing, and runtime orchestration in separate utility modules. `cli_infer.py` remains the adapter for normal synthesis and `regenerate_segments`; both routes obtain their final waveform from the same connect pipeline when a text action contains connect markup.

**Tech Stack:** Python 3.10+, NumPy, SoundFile, Librosa, OmniVoice, FunASR `fa-zh`, unittest.

## Global Constraints

- Chinese only; do not route non-Chinese text to `fa-zh`.
- Generate exactly three original candidates for each text action containing connect markup.
- Only edit adjacent characters inside a single connect range; never cross its endpoints.
- Use 20ms guards, 30ms minimum low-energy span, 20ms residual gap, 120ms maximum shortening, 8ms equal-power fade, -18dB relative and -40dBFS absolute thresholds.
- Process multiple boundaries from right to left and require exact removed-sample accounting.
- A processed candidate must pass a second alignment and quality check; otherwise retain its original version.
- Preserve current no-connect behavior and existing `regenerate_segments` natural-duration/subtitle semantics.
- FunASR loads only for a connect task. No commits are made in this working tree.

---

### Task 1: Add pure connect markup and alignment contracts

**Files:**
- Create: `omnivoice/utils/connect_markup.py`
- Create: `omnivoice/utils/connect_candidate_selector.py`
- Test: `tests/test_connect_markup.py`
- Test: `tests/test_connect_candidate_selector.py`

**Interfaces:**
- Produces `ConnectRange(start: int, end: int)`, `ConnectMarkup(synthesis_text: str, subtitle_text: str, ranges: tuple[ConnectRange, ...])`.
- Produces `CharacterTimestamp(character: str, start_ms: float, end_ms: float)`, `ConnectSelectionOptions`, and `evaluate_candidate(...)`.

- [ ] **Step 1: Write failing parser and selector tests**

```python
def test_parse_connect_markup_retains_only_internal_boundaries() -> None:
    markup = parse_connect_markup("[connect:课程]门户")
    assert markup.synthesis_text == "课程门户"
    assert markup.ranges == (ConnectRange(0, 2),)
    assert build_connect_boundaries(markup.ranges) == ((0, 1),)

def test_selector_rejects_missing_connect_timestamp() -> None:
    with pytest.raises(ValueError, match="connect character"):
        normalize_alignment("课程", [CharacterTimestamp("课", 0, 100)])
```

- [ ] **Step 2: Run focused tests and observe failures**

Run: `uv run python -m unittest tests.test_connect_markup tests.test_connect_candidate_selector -v`

Expected: import failures for the two new modules.

- [ ] **Step 3: Implement immutable parsing and scoring types**

```python
@dataclass(frozen=True)
class ConnectRange:
    start: int
    end: int

def parse_connect_markup(text: str, markup_version: str | None = None) -> ConnectMarkup:
    """Resolve markup and record inclusive-exclusive connect text offsets."""
    # Walk source markup left-to-right, append resolved synthesis/display text,
    # and record the output offset before and after each connect payload.

def build_connect_boundaries(ranges: Sequence[ConnectRange]) -> tuple[tuple[int, int], ...]:
    """Return adjacent offsets that both belong to the same connect range."""
    return tuple((index, index + 1) for item in ranges for index in range(item.start, item.end - 1))
```

Implement strict normalized alignment validation: expected Chinese characters must appear in order, timestamps must be finite, non-negative, monotonic, and within waveform duration. Make scoring pure: maximum and summed low-energy gap, maximum and summed aligned gap, removed samples, candidate index, and an `original` tie-break preference.

- [ ] **Step 4: Run focused tests**

Run: `uv run python -m unittest tests.test_connect_markup tests.test_connect_candidate_selector -v`

Expected: PASS.

### Task 2: Implement conservative waveform processing with tests

**Files:**
- Create: `omnivoice/utils/connect_waveform_processor.py`
- Test: `tests/test_connect_waveform_processor.py`

**Interfaces:**
- Consumes: channels-first `np.ndarray`, sample rate, timestamp-derived internal boundaries, `ConnectProcessingOptions`.
- Produces: `ProcessedWaveform(waveform, edits, total_removed_samples)` where edits contain a skip reason or exact original-coordinate ranges.

- [ ] **Step 1: Write deterministic fixture tests**

```python
def test_processor_removes_only_low_energy_between_guards() -> None:
    result = process_connect_waveform(waveform, 24_000, boundary, options)
    assert result.total_removed_samples <= round(0.120 * 24_000)
    assert result.waveform.shape[-1] == waveform.shape[-1] - result.total_removed_samples
    assert result.edits[0].processed is True

def test_processor_preserves_short_or_voiced_gaps() -> None:
    result = process_connect_waveform(voiced_gap, 24_000, boundary, options)
    assert result.total_removed_samples == 0
    assert result.edits[0].skip_reason == "no_continuous_low_energy_region"
```

- [ ] **Step 2: Run focused waveform tests and observe failures**

Run: `uv run python -m unittest tests.test_connect_waveform_processor -v`

Expected: import failure for `connect_waveform_processor`.

- [ ] **Step 3: Implement bounded analysis and splice logic**

```python
def process_connect_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    boundaries: Sequence[AlignedBoundary],
    options: ConnectProcessingOptions,
) -> ProcessedWaveform:
    """Remove only safe low-energy samples at internal connect boundaries."""
    # Copy channels-first samples; evaluate boundaries in descending original offset.
    # Require both relative and absolute RMS thresholds; retain residual gap.
    # Locate cut points near zero crossings, join shoulders with cos/sin fades,
    # and reject a result unless removed-frame accounting is exact.
```

Use local speech reference windows before/after each guard. Do not modify waveform if a reference, safe span, or protected region is unavailable. Keep actual channel count and dtype-compatible normalized float samples.

- [ ] **Step 4: Run focused waveform tests**

Run: `uv run python -m unittest tests.test_connect_waveform_processor -v`

Expected: PASS.

### Task 3: Add optional FunASR adapter and candidate pipeline

**Files:**
- Create: `omnivoice/utils/connect_candidate_pipeline.py`
- Modify: `pyproject.toml`
- Test: `tests/test_connect_candidate_pipeline.py`

**Interfaces:**
- Produces `FunASRAligner(device: str)` with `align(audio_path: str, text: str) -> tuple[CharacterTimestamp, ...]`.
- Produces `select_connect_waveform(generate_candidate, markup, sample_rate, options) -> WaveformSelection`.

- [ ] **Step 1: Write fake-aligner/fake-generator pipeline tests**

```python
def test_processed_alignment_failure_falls_back_to_original() -> None:
    selection = select_connect_waveform(fake_generator, markup, 24_000, options)
    assert selection.selected_version == "original"

def test_all_original_failures_stop_publication() -> None:
    with pytest.raises(RuntimeError, match="no valid original"):
        select_connect_waveform(failing_generator, markup, 24_000, options)
```

- [ ] **Step 2: Run pipeline tests and observe failures**

Run: `uv run python -m unittest tests.test_connect_candidate_pipeline -v`

Expected: import failure for `connect_candidate_pipeline`.

- [ ] **Step 3: Implement lazy alignment and original/processed orchestration**

```python
class FunASRAligner:
    """Lazily run FunASR fa-zh for one candidate WAV and expected text."""

    def align(self, audio_path: str, text: str) -> tuple[CharacterTimestamp, ...]:
        from funasr import AutoModel
        # Cache AutoModel(model="fa-zh", device=self._device) after first use.
        # Convert its millisecond result to strictly validated timestamps.

def select_connect_waveform(...):
    """Evaluate three originals and eligible processed versions deterministically."""
    # Persist original, alignment, edit, processed, and selection JSON only under debug_dir.
    # Build quality baselines from originals only and never publish a failed selection.
```

Add a `connect = ["funasr"]` optional dependency group, leaving the existing `eval` group unchanged. Catch only import/model/alignment failures at the adapter boundary and produce actionable errors describing `pip install 'omnivoice[connect]'`.

- [ ] **Step 4: Run pipeline tests**

Run: `uv run python -m unittest tests.test_connect_candidate_pipeline -v`

Expected: PASS without downloading FunASR models.

### Task 4: Integrate normal synthesis and segment regeneration

**Files:**
- Modify: `omnivoice/cli/cli_infer.py`
- Test: `tests/test_connect_cli_integration.py`

**Interfaces:**
- `concatenate_audio_actions(...)` accepts a `ConnectRuntimeOptions` object.
- `_run_regeneration(...)` passes the same options to every replacement action.

- [ ] **Step 1: Write CLI and fake-model integration tests**

```python
def test_parser_exposes_conservative_connect_defaults() -> None:
    args = get_parser().parse_args(["--text", "[connect:课程]"])
    assert args.connect_candidates == 3
    assert args.connect_processing == "conservative"

def test_regeneration_uses_selected_waveform_duration() -> None:
    # Fake pipeline returns a shortened waveform; rebuilt later subtitle start
    # equals the source timestamp plus actual replacement duration delta.
    assert rebuilt[-1].start == expected_start
```

- [ ] **Step 2: Run integration tests and observe failures**

Run: `uv run python -m unittest tests.test_connect_cli_integration -v`

Expected: parser assertion failure because the CLI flags do not yet exist.

- [ ] **Step 3: Wire the pipeline without changing no-connect behavior**

```python
if parse_connect_markup(val, args.markup_version).ranges:
    selection = select_connect_waveform(generate_one_candidate, markup, sample_rate, options)
    segment_audio = selection.waveform
else:
    segment_audio = model.generate(...)[0]
```

Add `--connect_candidates`, `--connect_processing`, `--connect_max_shorten_ms`, `--connect_aligner_device`, and `--connect_debug_dir`. Validate positive candidate/shorten values and restrict processing to `conservative`/`off`. Keep each replacement call independent so its speed and selected waveform duration flow unchanged into `regenerate_audio_segments`.

- [ ] **Step 4: Run integration and existing regeneration tests**

Run: `uv run python -m unittest tests.test_connect_cli_integration tests.test_audio_segment_regenerator -v`

Expected: PASS.

### Task 5: Document and verify all local contracts

**Files:**
- Modify: `README.md`
- Create: `docs/connect_conservative_waveform_processing_zh.md`
- Modify: `docs/superpowers/specs/2026-08-02-connect-conservative-waveform-processing-design.md`

- [ ] **Step 1: Document installation, CLI, fallback, and GPU acceptance**

Add a Chinese guide showing `uv sync --extra connect`, conservative and off CLI examples, debug artifact meanings, and the required real environment checks: actual FunASR alignment, original/processed A/B listening, no artifact check, and final SRT/JSON duration bounds.

- [ ] **Step 2: Run static and unit verification**

Run:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m py_compile omnivoice/cli/cli_infer.py omnivoice/utils/connect_markup.py omnivoice/utils/connect_candidate_selector.py omnivoice/utils/connect_waveform_processor.py omnivoice/utils/connect_candidate_pipeline.py
uv run python omnivoice/cli/cli_infer.py --help
uv lock --check
git diff --check
```

Expected: all local tests, compilation, CLI help, lock validation, and diff validation pass. Record GPU/FunASR generation and human listening as not locally verified unless actually run.

- [ ] **Step 3: Do not commit**

Leave all modifications in the working tree as required by this repository's user workflow.

## Plan Self-Review

- Spec coverage: Tasks 1-4 cover parsing, alignment, processing, fallback, debug evidence, CLI and both subtitle paths; Task 5 covers documentation and validation.
- Type consistency: all runtime code uses `ConnectMarkup`, `CharacterTimestamp`, `ConnectProcessingOptions`, and `WaveformSelection` created by Tasks 1-3.
- Scope: only the Chinese advanced CLI and its existing regeneration adapter change; core OmniVoice model code is untouched.
