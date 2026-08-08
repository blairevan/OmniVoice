# OmniVoice `connect` 与 IndexTTS 完整对齐设计

## 目标

以 IndexTTS 的最终保守方案为行为基准，重建 OmniVoice 的中文 `connect` 管线。完成后普通合成与局部重生成共享同一 Markup、候选、对齐、波形、字幕和调试契约；不以当前零散实现为兼容目标。

## 单一数据流

```text
raw markup
  -> ForcedSegment[]
  -> original candidates
  -> original alignment + original-only quality baseline
  -> optional conservative processed versions
  -> post-processing alignment + quality validation
  -> stable selection
  -> selected waveform sample counts
  -> SRT/JSON timeline or source-audio replacement timeline
```

每个 `text` action 都必须先变为 ForcedSegment 列表：含 connect 的单元进入候选管线；不含 connect 的单元保持单候选快速路径，但仍以独立模型片段的真实样本数生成一条字幕。不得对已经生成的长音频按字符比例事后拆字幕。

## 运行时边界

`parse_text_actions()` 继续唯一负责产生 `text` 与 `pause` action；`ForcedSegment` 只描述一个 text action 内的可合成单元，绝不创建或吸收 pause。这样 pause 的样本数和字幕空档只由 action 层累计一次。

每个 ForcedSegment 必须以一次独立 `OmniVoice.generate()` 生成。connect 管线调用时显式关闭 OmniVoice 的内部自动长文本分块（`audio_chunk_duration=0`，并确保阈值不会触发），否则模型内部 chunk 不能映射为独立字幕和候选证据。

同一调用还必须传入 `postprocess_output=False`、`pad_duration=0` 与 `fade_duration=0`。OmniVoice 即使关闭 `postprocess_output` 仍会执行边缘淡化/填充；这些参数保证每个 ForcedSegment 的样本数只来自模型生成和 connect 选择，不会被片段级边缘静音改变。

单个 connect token 的显示长度超过 `max_char_len` 时允许该 ForcedSegment 超长，但不得切开 token；日志和调试 JSON 记录 `oversizedConnectSegment=true`。超过模型安全输入长度时任务失败并提示用户缩短 connect 内容，不进行隐式切断。

模型安全输入长度必须来自 OmniVoice tokenizer/config 暴露的明确最大 token 数；若当前模型版本未暴露该值，CLI 不得编造隐式限制，只验证用户显式配置的 `--max_forced_segment_tokens`（缺省为不限制）并在 debug JSON 记录实际 token 数。

候选必须具有可控多样性和可复现性：运行时生成一个任务 seed，候选使用稳定派生 seed；`selection.json` 记录任务 seed、候选 seed、采样温度和波形哈希。若候选波形哈希相同，保留首个版本、记录 `duplicateCandidate`，不把重复版本当作独立竞争者。

## 模块边界

- `connect_markup.py`：Token 化标准/26071300 Markup；输出 `ForcedSegment(synthesis_text, subtitle_text, alignment_text, alignment_source_offsets, connect_spans)`；按 token、标点和显示字符长度预切分。
- `connect_candidate_selector.py`：对齐字符映射、候选声学指标、original-only 中位基线、质量门禁、语速校正间隔和稳定排序。
- `connect_waveform_processor.py`：多声道短窗 RMS、局部基线、保护区、连续低能量区、零交叉、等功率淡化及采样数不变量。
- `connect_candidate_pipeline.py`：候选写盘、对齐、处理、二次对齐、回退、完整 JSON 证据和选择。
- `cli_infer.py`：统一 CLI、普通/局部重生成适配器、正常输出边界静音、严格时间轴。

所有旧的 `clean_text_for_synthesis()` 与 `clean_text_for_subtitles()` 必须改为从同一个 ForcedSegment 适配器取值，不得保留第三套独立正则规则。

## 强制契约

1. 仅中文 `connect`；connect 范围不可跨预切模型片段。
2. 1–5 个 original，默认 3；所有 original 都计入时长/RMS 基线。
3. original 失败不处理；processed 失败只回退对应 original；所有 original 都不合格才失败。
4. `conservative` 使用 5ms/2.5ms RMS、20ms 保护区、30ms 最短低能量、20ms残留、120ms 上限、8ms淡化、-18dB/-40dBFS、3ms零交叉搜索；`off` 不测量或编辑波形。
5. 排序顺序：最大低能量停顿、低能量停顿总和、语速校正的最大/总对齐间隔、删除时长、候选号、original 优先。
6. 每个预切模型片段只生成一条字幕，结束时间由最终选中波形的真实采样数决定；局部替换一条替换字幕并累计移动后续字幕。
7. 仅完整普通输出添加 `boundary_silence=0.3` 并平移字幕；局部替换传入 `0`。
8. 调试目录固定为 `replacement_NNN/action_NNN/segment_NNN`，保存 original/processed WAV、对齐、edits、selection，且 selection 包含所有拒绝理由、指标和配置。
9. 最终时间轴在速度变换、边界静音和编码完成后重新读取正式音频；最后一条 SRT/JSON 结束时间不得超过真实总时长。局部替换强制 `boundary_silence=0`。
10. 音频保存函数返回实际发布路径；MP3 编码回退为 WAV 时，SRT/JSON 的 `audio_file`、总时长和日志均使用该实际路径。
11. connect 调试根目录不得混入旧任务：显式目录已存在时失败；默认目录以输出 stem 加唯一任务后缀创建。目录创建在候选生成前完成且不会覆盖正式音频/字幕/源文件。
12. 速度变换前记录帧数；变换后重新读取实际帧数，以 `final_frames / pre_speed_frames` 缩放所有模型片段字幕，而非按请求 speed 推算。随后才添加完整输出边界静音。
13. FunASR 适配器只使用临时对齐 WAV：保留原多声道候选用于最终发布，另建单声道、明确采样率的分析视图；对齐失败 JSON 同时记录异常字符串、模型标识和返回结构摘要。

## 验收

本地：纯 Markup、映射、质量、波形、伪对齐器、伪 TTS、双声道、目录隔离、普通/局部字幕与安全发布测试。伪 TTS 必须断言每次 ForcedSegment 调用的 `postprocess_output=False`、`audio_chunk_duration=0`、`pad_duration=0`、`fade_duration=0`；覆盖重复候选、实际变速比例、MP3 回退、JSON `audio_file` 同步、最终 SRT/JSON 边界和 FunASR 懒加载。FunASR 保持 `connect` optional extra，锁文件随依赖声明一起更新。

服务器：真实 FunASR `fa-zh`、GPU 合成、逐候选 A/B 听感、无吞字/爆音/标签外节奏变化、最终 SRT/JSON 不越过音频时长。
