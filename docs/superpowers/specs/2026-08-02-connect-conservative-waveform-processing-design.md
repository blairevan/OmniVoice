# OmniVoice `connect` 保守波形处理设计

## 1. 目标与范围

将 `omnivoice-infer-advanced` 的 `[connect:文字]` 从仅清理的文本标记升级为可验证的连续发音控制。首期仅支持中文，并与 IndexTTS 的最终实现保持相同的候选、对齐、波形处理、回退和调试证据策略。

本设计不改变无 `connect` 文本的生成路径，不改变现有 `[pause]`、`[replace]`、发音标记、语速或片段重生成的语义。

## 2. 非目标

- 不把 `connect` 标记直接传给 OmniVoice 模型，也不修改模型语义 token 或扩散采样逻辑。
- 不支持中文以外语言的强制对齐。
- 不压缩标签外的停顿，不做整句时间拉伸。
- 不要求替换片段维持其源音频时长。

## 3. 总体流程

```text
解析 Markup 并记录 connect 字符范围
  -> 对含 connect 的完整文本片段生成三个 original 候选
  -> FunASR fa-zh 执行候选音频与清理后文本的字级强制对齐
  -> 原版完整性和音频质量校验
  -> 只构造 connect 内部相邻字符边界
  -> 保守低能量区压缩、零交叉切点和等功率淡化
  -> 生成 processed 候选
  -> processed 再次 fa-zh 对齐和质量校验
  -> original 与合格 processed 稳定排序
  -> 以选中波形的真实采样数驱动字幕时间轴
```

无 `connect` 的文本保持现有单候选 `model.generate()` 路径。`connect_processing=off` 保留三候选择优和强制对齐，但不生成 processed 候选。

## 4. 模块职责

- `omnivoice/utils/connect_markup.py`：解析标准及 `markup_version=26071300` 标记，输出合成文本、显示文本和精确 connect 字符范围。
- `omnivoice/utils/connect_candidate_selector.py`：规范 FunASR 时间戳、映射合成文本字符、构造标签内部边界，并计算候选质量指标。
- `omnivoice/utils/connect_waveform_processor.py`：检测低能量区、保护切点、寻找零交叉点、执行等功率淡化并验证采样数不变量。
- `omnivoice/utils/connect_candidate_pipeline.py`：生成三候选、调用对齐器、执行 original/processed 回退与排序、写入调试证据。
- `omnivoice/cli/cli_infer.py`：暴露 CLI 选项；普通合成与 `regenerate_segments` 均通过候选管线生成含 connect 的文本片段。
- 既有 `segment_timeline.py` 和 `audio_segment_regenerator.py`：不负责 connect 声学逻辑；继续根据最终 WAV 的真实时长重建 SRT/JSON 时间轴和安全发布文件。

## 5. 对齐器边界

首期采用 FunASR `fa-zh`。对齐适配器接受单声道或原始候选 WAV 路径与完整清理后中文文本，返回每个可发音中文字符的 `start_ms` 与 `end_ms`。

对齐结果必须满足：

- 文本字符完整且顺序一致。
- 每个 connect 字符都存在有限且递增的时间戳。
- 时间戳位于波形实际时长内。

无法满足任何条件时，original 候选淘汰；不会基于不完整对齐执行波形编辑。

## 6. 保守波形处理

仅处理同一 connect 范围内的相邻字符。例如 `[connect:课程门户首页]副屏` 仅尝试：`课|程`、`程|门`、`门|户`、`户|首`、`首|页`，绝不触碰 `页|副`。

每个边界的搜索区间为：

```text
search_start = left.end_ms + boundary_guard_ms
search_end = right.start_ms - boundary_guard_ms
```

保护区重叠、局部语音基线不足或没有充分连续低能量区时，跳过该边界且保留原波形。

默认参数：

| 参数 | 值 | 说明 |
|---|---:|---|
| `frame_ms` | 5ms | RMS 分析窗口 |
| `hop_ms` | 2.5ms | RMS 分析步长 |
| `boundary_guard_ms` | 20ms | 字符两端保护区 |
| `minimum_low_energy_ms` | 30ms | 最短可编辑低能量区 |
| `residual_gap_ms` | 20ms | 编辑后至少保留的低能量时长 |
| `maximum_shorten_ms` | 120ms | 单边界最多缩短量 |
| `crossfade_ms` | 8ms | 等功率淡化长度 |
| `relative_drop_db` | 18dB | 相对局部语音基线的能量下降 |
| `absolute_ceiling_dbfs` | -40dBFS | 低能量绝对上限 |
| `zero_crossing_search_ms` | 3ms | 切点零交叉搜索范围 |

一个分析帧必须同时低于局部语音基线 18dB 且不高于 -40dBFS。每边界只编辑最长连续低能量区，至少残留 20ms；总缩短量包含交叉淡化重叠。多个边界按原坐标从右向左处理。

编辑后必须验证：波形有限、无异常削波、声道数不变，且 `input_frames - output_frames == total_removed_samples`。任一不变量失败时丢弃 processed，保留 corresponding original。

## 7. 候选、回退与排序

每个含 connect 的模型片段生成 `connect_candidates=3` 个 original 候选。

1. original 对齐或质量校验失败：淘汰该候选，不尝试处理。
2. original 合格但无安全编辑区：只保留 original。
3. processed 的编辑、不变量、二次对齐或质量校验失败：只保留 original，并记录回退原因。
4. processed 合格：original 与 processed 共同参与选择。
5. 只有三个 original 全部不合格，当前文本片段才失败；正式输出不得发布。

合格版本稳定排序，从小到大依次比较：最大连续低能量停顿、连续低能量停顿总和、最大对齐字间隔、对齐字间隔总和、删除时长、候选编号。完全同分时优先 original。

时长和 RMS 离群标准只由三个 original 计算，processed 不能改变自身的验收基线。

## 8. CLI 契约

```text
--connect_candidates 3
--connect_processing conservative
--connect_max_shorten_ms 120
--connect_aligner_device cpu
--connect_debug_dir <directory>
```

`--connect_processing` 仅允许 `conservative` 或 `off`。其余声学阈值保留在强类型配置数据类中，避免未标定的内部参数成为公共接口。

对 `regenerate_segments`，每个请求继续使用各自的文本、语速、声音条件和现有 Markup 管线；最终 replacement WAV 的实际采样数决定替换字幕终点和后续所有字幕的累计偏移。

## 9. 调试证据

每个 connect 模型片段生成独立目录：

```text
<connect_debug_dir>/
  replacement_001/
    action_001/
      segment_001/
        candidate_001_original.wav
        candidate_001_original_alignment.json
        candidate_001_processed.wav
        candidate_001_processed_alignment.json
        candidate_001_edits.json
        candidate_002_...
        candidate_003_...
        selection.json
```

`candidate_NNN_edits.json` 记录每个边界的对齐间隔、搜索区、低能量区、删除样本数、淡化长度、是否处理及跳过原因。`selection.json` 记录处理选项、所有版本质量分数、选中候选/版本、删除采样数和回退原因。

## 10. 测试与验收

### 本地单测

- Markup 解析可准确映射标准及新版本 connect 字符范围。
- 只删除标签内部两个保护区之间的低能量样本。
- 有声帧、短静音、标签外区域和多边界原坐标保持正确。
- 淡化没有 NaN、无穷值、异常削波或采样数不一致。
- original/processed 二次对齐失败与质量失败均正确回退。
- 无 connect 时不加载对齐器也不进入候选处理器。
- 最终选中波形的真实时长正确进入普通字幕和 `regenerate_segments` 累计偏移。

### 真实环境验收

- 在 CPU 上运行 FunASR `fa-zh` 对齐真实候选，确认字符顺序和时间戳完整。
- 在 GPU 上合成短、长 connect 文本，并逐一试听 original、processed 和最终选中版。
- processed 不得出现吞字、重复尾音、爆音、音量抬升或标签外节奏改变。
- `selectedVersion=processed` 时 `totalRemovedSamples > 0`。
- SRT/JSON 最后一条字幕结束时间不超过最终音频总时长；替换区间只产生其既有字幕拆分规则允许的条目。

## 11. 依赖与发布原则

FunASR 是新增的可选运行时能力：无 connect 时不得加载；只有含 connect 的任务才构建或调用 `fa-zh` 对齐器。实现前必须将依赖及模型下载/缓存位置写入项目文档并确认安装方案。所有新输出先写临时目录，经音频与字幕校验后再安全发布；失败时不覆盖正式输出。
