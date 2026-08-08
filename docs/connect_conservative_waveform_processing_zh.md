# OmniVoice `connect` 连读处理

`[connect:课程门户首页]` 仅支持中文。默认对含该标记的文本生成三个完整候选，使用 FunASR `fa-zh` 对齐，并只在标签内部相邻字符之间的安全低能量区进行保守压缩；处理版对齐或质量失败时自动回退原始候选。

安装对齐能力：

```bash
uv sync --extra connect
```

默认保守模式：

```bash
omnivoice-infer-advanced --text '[connect:课程门户首页]副屏组件。' \
  --connect_processing conservative --connect_aligner_device cpu \
  --output output.wav
```

使用 `--connect_processing off` 可关闭波形编辑，但仍保留三候选与强制对齐。每个边界最多缩短 120ms，并保护两侧 20ms 发音区域。真实 GPU 合成后必须试听 original/processed，确认未吞字、爆音或改变标签外韵律。每个含 `connect` 的最终模型片段和局部替换区间只生成一条字幕，严格覆盖最终 WAV 的真实样本时长，不按字符数比例二次拆分。

可用的稳定性与诊断参数包括：`--connect_candidates 1-5`、`--connect_seed`、`--connect_max_gap_ms`、`--connect_debug_dir`、`--connect_aligner_model`，以及显式的 `--max_forced_segment_tokens`。调试目录必须不存在，程序会写入 `candidate_NNN` 的原始/处理波形、对齐结果、编辑记录、候选拒绝原因和最终 `selection.json`；候选失败不会中止其它候选。未显式配置 token 上限时不会伪造模型限制。

普通完整输出默认在最终编码前后边界各增加 `0.3s` 静音并同步字幕；源音频局部替换固定不增加边界静音。MP3 编码不可用时会发布同名 WAV，并在 JSON 元数据中使用实际发布路径。
