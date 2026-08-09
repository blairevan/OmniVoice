# Dev 分支代码审查问题清单

> 审查时间：2026-08-09
> 比较基准：`master` vs `dev`

---

## 🔴 严重问题

### 1. 硬编码绝对路径（影响可移植性）

**文件**：`omnivoice/cli/cli_infer.py:938,941`

```python
parser.add_argument("--whisperx_runtime_dir", default="/opt/app/aining/digital_human/whisperx", ...)
parser.add_argument("--whisperx_model", default="/opt/app/aining/digital_human/whisperx/models/zh", ...)
```

两个默认值是特定机器的绝对路径，在其他环境部署会直接失败。

**建议**：使用相对路径或环境变量，或设为 `default=None` 并在运行时校验。

---

### 2. `pydub` 顶层导入导致模块加载失败

**文件**：`omnivoice/utils/audio_segment_regenerator.py:17`

```python
from pydub import AudioSegment
```

`pydub` 是可选依赖（仅用于 MP3 输出），但这里是模块级导入。如果 `pydub` 未安装，`import audio_segment_regenerator` 就会直接失败。

对比 `cli_infer.py:750` 在 `save_audio()` 函数内部做了延迟导入，处理正确。

**建议**：将 `pydub` 的导入移到 `_write_audio()` 函数内部，与 `cli_infer.py` 保持一致。

---

## 🟡 中等问题

### 3. `connect_debug_dir` 始终自动创建

**文件**：`omnivoice/cli/cli_infer.py:1212-1245`

```python
if args.connect_debug_dir is None:
    output_path = Path(args.output)
    args.connect_debug_dir = str(
        output_path.parent / f"{output_path.stem}_connect_debug"
    )
```

无论用户是否使用 connect 处理（纯文本合成或 segment regeneration 模式），都会基于输出路径自动生成 debug 目录，然后 `_prepare_connect_debug_dir` 会删除并重建。对于不需要 connect 的场景，这是不必要的磁盘操作。

**建议**：仅在 `args.connect_processing != "off"` 时自动生成 debug 目录。

---

### 4. `select_connect_waveform` 函数过长

**文件**：`omnivoice/utils/connect_candidate_pipeline.py:168-433`（约 265 行）

该函数承担了多个职责：候选生成、对齐验证、音频质量检查、波形处理、评分排名、持久化证据写入。违反单一职责原则，难以测试和维护。

**建议**：拆分为独立函数：
- `_generate_candidates()` — 候选音频生成
- `_validate_and_score_candidates()` — 验证和评分
- `_persist_evidence()` — 证据持久化

---

### 5. `measure_low_energy_pauses` 条件逻辑分散

**文件**：`connect_candidate_pipeline.py:288-289`

```python
pause_max, pause_total = measure_low_energy_pauses(
    original, sample_rate, boundaries, processing_options
) if processing_options else (0.0, 0.0)
```

当 `options.processing == "conservative"` 但 `processing_options is None` 时，第 291-292 行会抛出 `RuntimeError`。但第 288 行在 `processing_options` 为 `None` 时返回 `(0.0, 0.0)`，这发生在 `options.processing == "off"` 时。逻辑正确但条件分散，不直观。

**建议**：将 processing 分支提取为独立函数，使逻辑更清晰。

---

### 6. `coherent_timeline.py` tokenizer 类型安全性

**文件**：`omnivoice/utils/coherent_timeline.py:127-148`

`token_count` 函数依赖 `hasattr` / `callable` 检查来推断 tokenizer 类型，如果 tokenizer 返回意外的数据结构格式，可能产生错误的 token 计数。

**建议**：增加更严格的类型校验，或在 token 计数失败时抛出明确错误。

---

## 🟢 轻微问题

### 7. FunASR 模型版本硬编码

**文件**：`omnivoice/utils/connect_candidate_pipeline.py:49,101`

```python
def _resolve_local_funasr_model_path(model_name: str, model_revision: str = "v2.0.4") -> str:
...
model_revision="v2.0.4",
```

模型版本 `v2.0.4` 硬编码在函数签名和调用处。如果将来需要升级，需要修改多处源码。

**建议**：考虑通过环境变量或配置暴露。

---

## 总结

| 严重程度 | 数量 | 关键项 |
|---------|------|--------|
| 🔴 严重 | 2 | 硬编码路径、pydub 顶层导入 |
| 🟡 中等 | 4 | debug_dir 自动创建、函数过长、条件逻辑分散、tokenizer 类型安全 |
| 🟢 轻微 | 1 | 模型版本硬编码 |

**优先修复**：硬编码路径（#1）和 pydub 导入（#2）是阻塞性问题，其他环境部署会直接失败。

---

## 修复总结（2026-08-09）

本轮已对上述 7 项逐条处理。修复按实际运行行为处理。

| # | 状态 | 修复内容 |
|---|---|---|
| 1 | ✅ 已修复 | 移除 WhisperX 的机器绝对路径默认值；支持 `OMNIVOICE_WHISPERX_RUNTIME_DIR` / `OMNIVOICE_WHISPERX_MODEL` 环境变量或 CLI 显式参数。仅在确实需要多句对齐且配置完整时创建 WhisperX aligner；未配置时明确 warning 并回退到 generated timing。 |
| 2 | ✅ 已改进 | `pydub` 当前属于 `pyproject.toml` 的正式依赖，因此原审查中“可选依赖”的前提不准确。仍已移除 `audio_segment_regenerator.py` 和 `audio.py` 的顶层 `pydub` 导入，改为实际需要 MP3/静音处理时延迟加载；WAV 写入不再触发 pydub 导入。 |
| 3 | ✅ 已修复 | connect debug evidence 目录改为仅在本次调用实际包含 `[connect:...]` markup 时准备。普通文本合成、无 connect 的 segment regeneration 即使默认 `connect_processing=conservative` 也不会创建、删除 debug 目录。 |
| 4 | ✅ 已重构 | `select_connect_waveform()` 从约 265 行降至约 150 行。拆出 `_generate_candidate_waveforms()`、`_evaluate_candidate()`、`_filter_candidate_versions()`、`_persist_selection_evidence()` 等职责函数，候选生成、验证评分、质量过滤和证据持久化已分离。 |
| 5 | ✅ 已重构 | pause measurement 统一进入 `_measure_candidate_pauses()`；`conservative` 模式缺少 processing options 的校验提前到 candidate selection 入口，不再把条件分散在候选循环中。 |
| 6 | ✅ 已修复 | 新增 `_count_model_tokens()` 严格校验 tokenizer 输出。支持 `tokenize()`、`input_ids`、单 token-ID sequence；对于缺少 `input_ids`、单字符串返回多 batch、结构不一致等情况明确抛出 `TypeError`，不再用 mapping 长度等不可靠值充当 token 数。 |
| 7 | ✅ 已修复 | 增加 `DEFAULT_FUNASR_MODEL_REVISION = "v2.0.4"`，resolver 和 `AutoModel` 共用同一版本常量，源码中版本字面量只保留一处。 |

---

## 第二轮审查（2026-08-09）

上轮 7 项全部修复后重新审查，新增 2 个轻微问题。

### 🟢 轻微问题

#### 8. `_waveform_quality_reasons` 对原始波形重复检查

**文件**：`connect_candidate_pipeline.py:303, 460`

原始波形在 `_evaluate_candidate`（第 303 行）通过 `_waveform_quality_reasons` 检查后，又在 `_filter_candidate_versions`（第 460 行）被再次调用同一函数检查。每个通过的原始波形都被检查了两遍（非有限值、静音、削波、时长离群、RMS 离群各两次）。processed 版本仅在第二次检查，这是必要的。不影响正确性，但原始波形的质量检查冗余。

**建议**：在 `_filter_candidate_versions` 中跳过 `version == "original"` 的波形，或让 `_evaluate_candidate` 返回时标记已通过质量检查。

**状态**：✅ 已修复。`_filter_candidate_versions()` 对 `original` 版本直接复用 `_evaluate_candidate()` 已完成的质量检查结果，仅对 `processed` 版本执行第二阶段 `_waveform_quality_reasons()`。新增回归测试验证 original + processed 场景下质量检查只额外调用一次。

#### 9. `FunASRAligner.__init__` 单行可读性差

**文件**：`connect_candidate_pipeline.py:90`

```python
def __init__(self, device: str, model: str = "fa-zh") -> None: self._device, self._model_name, self._model = device, model, None
```

属性赋值挤在一行，与项目其他 `__init__` 方法风格不一致，降低可读性。

**建议**：展开为多行。

**状态**：✅ 已修复。`FunASRAligner.__init__()` 已展开为独立属性赋值，初始化行为不变。

---

## 修订记录

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-08-09 | v1 | 初始审查，发现 7 个问题（2 严重 / 4 中等 / 1 轻微） |
| 2026-08-09 | v2 | 7 个问题全部修复，新增修复总结；第二轮审查发现 2 个轻微问题 |
| 2026-08-09 | v3 | 第二轮 2 个轻微问题全部修复；去除原始波形重复质量检查并整理 FunASR aligner 初始化格式；新增回归测试并合并重复修复总结 |

---

## 第三轮审查（2026-08-09）

本轮对全部 9 项修复后的代码进行最终审查，未发现新问题。

### 审查范围

- `omnivoice/cli/cli_infer.py` — WhisperX 环境变量、connect debug 按需创建、`uses_connect` 提前检测
- `omnivoice/utils/connect_candidate_pipeline.py` — 函数拆分、`_filter_candidate_versions` 跳过原始波形、`FunASRAligner.__init__` 多行格式、`DEFAULT_FUNASR_MODEL_REVISION` 常量
- `omnivoice/utils/audio_segment_regenerator.py` — pydub 延迟导入
- `omnivoice/utils/audio.py` — pydub 延迟加载封装
- `omnivoice/utils/coherent_timeline.py` — `_count_model_tokens` 严格校验
- `tests/` — 新增 WhisperX 默认值、connect debug 按需创建、质量过滤、FunASR 路径等回归测试

### 验证状态

- `omnivoice/` 中 WhisperX `/opt/app/aining/digital_human/whisperx` 硬编码路径：**0 处**
- Python 源码顶层 `pydub` import：**0 处**
- `connect_candidate_pipeline.py` 中 `v2.0.4` 字面量：**1 处**（统一常量定义）
- `select_connect_waveform()`：约 **150 行**
- `_filter_candidate_versions()`：original 不重复检查

### 结论

**全部 9 个问题已修复，代码质量良好，无新增问题。**

---

## 第二轮修复总结（2026-08-09）

第二轮新增的 2 个轻微问题均已处理：

- #8：去除 original candidate 的重复质量检查，final filter 仅复检 processed candidate。
- #9：展开 `FunASRAligner.__init__()` 的单行属性赋值，提高可读性，不改变行为。
- 新增 `_filter_candidate_versions()` 回归测试，覆盖 original + processed 共存时只对 processed 做第二阶段质量检查。
- 当前 `tests/` 下测试方法总数为 **87**。

### 额外回归保护

- 保留 connect debug ownership marker 机制，避免递归删除非 OmniVoice 管理目录。
- WhisperX 未配置时不会为 alignment 提前释放 TTS GPU。
- 保留 WhisperX 环境变量/default、connect debug 按需创建、tokenizer 异常结构、WAV 无 pydub 写入等回归测试。

### 验证状态

已完成源码静态核对：

- `omnivoice/` 中 WhisperX `/opt/app/aining/digital_human/whisperx` 硬编码路径：**0 处**。
- Python 源码顶层 `pydub` import：**0 处**。
- `connect_candidate_pipeline.py` 中 `v2.0.4` 字面量：**1 处**（统一常量定义）。
- `select_connect_waveform()`：约 **150 行**。
- `_filter_candidate_versions()`：original 不再重复调用 `_waveform_quality_reasons()`。

当前 DevSpace 运行环境仍没有可用的 Python/pytest 解释器，仓库现有 `.venv` / `venv` 指向失效的 macOS Python 路径，因此本轮无法实际执行 pytest。代码与测试均已写入工作区；恢复可用 Python 环境后应执行完整 `pytest -q` 作为最终动态验证。