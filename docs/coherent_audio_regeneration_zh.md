# OmniVoice 连贯合成与局部返工

OmniVoice 的正常生成和字幕对齐返工与 IndexTTS 采用相同的核心边界：连续句组一次性合成，使用自然生成时长，局部返工按原始样本轴切割，并在输出样本轴累计时长 delta。

## 返工请求

正式请求格式为：

```json
[
  ["00:00:04.509", "00:00:06.655", ["新的第一句。"], 1.0],
  ["00:00:10.000", "00:00:14.000", ["连续句一。", "连续句二。"], 1.1]
]
```

四个字段分别是开始时间、结束时间、替换文本或句子数组、句组语速。数组文本必须与被覆盖的完整字幕项一一对应。旧的三字段请求仍可使用，此时使用全局 `--speed` 作为句组语速。

相邻且语速一致的完整字幕请求会在 TTS 前合并；语速不同的相邻请求会被拒绝，避免分别生成后再用淡化伪造连贯性。

## WhisperX 运行时

OmniVoice 不在默认环境安装 WhisperX，而是调用共享隔离运行时：

```text
/opt/app/aining/digital_human/whisperx
```

默认本地对齐模型为：

```text
/opt/app/aining/digital_human/whisperx/models/zh
```

调用过程设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，禁止运行期联网下载。`--whisperx_timeout_seconds` 默认 10800 秒，表示整个任务的单调总 deadline；多个句组共享这一 deadline。

WhisperX 只允许 CUDA。CUDA 不可用、对齐 OOM、超时、进程失败、模型缺失或 TTS GPU 释放失败时，保留已生成音频，改用发音单元权重分配字幕时间，不自动回退 CPU。

正常合成只使用 `--leading_silence`、`--trailing_silence`（单位为毫秒，默认均为 300）和 `--subtitle_offset`（`auto` 或毫秒数）。不再支持 `--boundary_silence`。

当 `--subtitle_offset auto` 时，`active_start_seconds` 从本次实际生成的 TTS 波形中动态测量，发生在首尾静音追加之前；测量值、检测阈值、解析后的 offset 和首尾静音会写入 timing 日志。

OmniVoice 高级 CLI 默认启用 HuggingFace/Transformers 本地离线缓存，不会在运行时访问 HuggingFace；只有明确传入 `--no-offline` 才允许联网下载。服务器必须提前缓存 `k2-fsa/OmniVoice` 及其 audio tokenizer。

## GPU 生命周期

同一个任务分两个阶段：

1. 在 OmniVoice GPU 模型仍可用时完成全部 TTS 句组并写入临时 WAV。
2. 完成最后一个 TTS 句组后，将 OmniVoice 模型、audio tokenizer 和可选 ASR 组件迁出 GPU，执行垃圾回收和 CUDA 缓存清理；只有确认进程内 `torch.cuda.memory_allocated()` 不超过 128 MiB 后，才启动 WhisperX CUDA。

如果没有多句组需要对齐，不启动 WhisperX，也不进行不必要的 GPU 释放。

## 时间轴与拼接

局部返工同时维护：

- 原始样本轴：固定源音频切点，不受前一个返工组时长变化影响。
- 输出样本轴：使用此前所有句组的累计 delta 平移后续未修改音频和字幕。

返工片段边界使用 15ms 等功率 overlap-add。实际左右 overlap 样本数参与：

```text
group_delta_samples
  = generated_samples
  - original_range_samples
  - left_overlap_samples
  - right_overlap_samples
```

SRT、JSON 和句组 manifest 都从同一份整数样本时间轴生成。JSON 句组记录 `timing_method`、`alignment_coverage`、`fallback_reason`、生成样本数、左右 overlap 和 delta。

## 验收

纯逻辑测试、fake-TTS 和静态检查只能验证时间轴、文件安全和编排逻辑。目标服务器还必须验证 GPU 内存释放、共享 WhisperX 实际字符对齐、音频编解码、字幕单调性、无丢句/重复句，以及连续句组相对逐句合成的听感效果。
