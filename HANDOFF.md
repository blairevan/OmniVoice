# OmniVoice Handoff

## 文档维护规则

- 后续每次更新本文件时，优先增量修改或追加相关章节，不得无理由整体覆盖已有内容。
- 可以修正过时结论，但必须在“修订记录”中追加修改时间和修改小结。
- 时间统一使用服务器/项目约定的北京时间（Asia/Shanghai）。
- 必须区分本地静态/单元测试、真实服务器运行、GPU、上传结果和人工试听，不能互相替代。

## 当前任务

让 OmniVoice 自己从项目根目录 YAML 配置中读取 WhisperX 独立运行目录和中文对齐模型目录，避免 `xc-server` Consumer 传递机器路径；随后通过真实服务器日志确认 WhisperX 已被启用，并检查完整音频生成链路。

当前服务器目录约定：

```text
/root/digital_human/OmniVoice
/root/digital_human/whisperx
/root/digital_human/whisperx/models/zh
```

当前配置：

```yaml
whisperx:
  runtime_dir: /root/digital_human/whisperx
  model_dir: /root/digital_human/whisperx/models/zh
```

## 已完成内容

### 1. WhisperX YAML 配置

- 新增项目根目录 `config.yaml`，使用服务器绝对路径。
- 新增 `omnivoice/utils/runtime_config.py`：
  - 从仓库根目录定位 `config.yaml`，不依赖进程当前工作目录。
  - 使用 `yaml.safe_load()`。
  - 校验 YAML 根节点、`whisperx` 节点和非空字符串路径。
  - 配置文件缺失时保留 generated-timing 回退；配置存在但格式错误时明确失败。
- `omnivoice/cli/cli_infer.py` 已接入配置加载，优先级为：

  ```text
  CLI 参数 > 环境变量 > config.yaml > generated timing fallback
  ```

- `xc-server` 无需增加 WhisperX 参数或环境变量。
- `pyproject.toml` 已直接声明 `pyyaml>=6.0,<7`，`uv.lock` 已同步。
- README 已更新为当前 YAML 配置契约。

### 2. 自动化验证

- 新增 `tests/test_runtime_config.py`，覆盖：
  - 正常绝对路径加载。
  - 文件缺失。
  - 缺少 `whisperx` 节点。
  - YAML 语法错误。
  - 根节点或 `whisperx` 节点类型错误。
  - 空白路径和非字符串路径。
- 更新 `tests/test_connect_cli_integration.py`，覆盖 YAML、环境变量和 CLI 的覆盖优先级。
- 最后一次本地验证结果：

  ```text
  uv run python -m unittest discover -s tests -q
  Ran 96 tests
  OK
  ```

- `py_compile` 和 `git diff --check` 均通过。
- 无环境变量、无 WhisperX CLI 参数时，解析器实际输出：

  ```text
  /root/digital_human/whisperx
  /root/digital_human/whisperx/models/zh
  ```

### 3. 真实服务器验收

2026-08-09 23:52 的服务器日志确认：

- OmniVoice 模型从本地 HuggingFace snapshot 加载，模型加载耗时 `1.195s`。
- `[connect:教师数量]` 生成 3 个候选，3 个全部成功。
- 本地 FunASR `fa-zh` 从明确 snapshot 的 `model.pt` 加载，权重全部匹配。
- 最终选择 `candidate=1 version=original`，`max_gap_ms=0`，未删除采样。
- WhisperX 对齐阶段耗时 `7.242s`，日志为：

  ```text
  [timing] subtitle alignment/timeline: 7.242s subtitles=7 whisperx=True
  ```

- 两个多字幕分组的内部字幕边界相较 generated-timing 结果均发生变化，而分组采样数保持一致；这表明两个分组实际采用了 WhisperX 字符对齐结果。
- `active_start_seconds=0` 是实际波形检测值；增加 300ms 头部静音后，第一条字幕从 `0.300s` 开始。
- 音频时长计算一致：`31.960 + 0.300 + 1.000 = 33.260s`。
- MP3、SRT 和 JSON 字幕均已成功写入输出目录。
- CLI 总耗时 `39.411s`，其中 TTS/group generation 为 `30.678s`。

## 当前卡住或未闭环的问题

### 1. 上传最终状态缺少日志证据

最新附件只到：

```text
Starting file upload via API
```

尚未看到：

```text
Audio generation completed successfully
```

因此当前只能确认本地 MP3/SRT/JSON 生成成功，不能确认文件上传、音频时长/字幕写库和任务状态更新全部完成。

### 2. 参考音频过长

参考音频为 `21.4s`，每次 TTS 调用都会触发 `>20s` 警告。该问题不阻塞生成，但可能增加耗时、显存占用并降低音色克隆质量。建议准备与参考文本严格对应的 3–10 秒音频。

### 3. 缺少人工试听验收

日志和字幕边界只能证明处理链路与时间线，不能证明以下听感：

- `connect` 拼接处是否自然。
- 音色克隆是否稳定。
- 字幕与实际发音是否主观同步。
- 头部 300ms、尾部 1000ms 静音是否符合产品体验。

### 4. WhisperX 失败分支可观测性不足

`resolve_generated_group_timings()` 当前捕获所有对齐异常后直接回退，没有记录具体分组和 stderr。此次可通过字幕边界变化确认成功，但未来失败时仅看 `whisperx=True` 不能证明所有分组均成功。若要增强诊断，应单独增加分组级成功/失败日志；此项尚未实施。

### 5. UV 重建是否重复发生待观察

部署新依赖后的首次服务器运行出现：

```text
Building omnivoice
Uninstalled 1 package
Installed 1 package
```

首次同步可以接受。如果后续每个生成任务都重复出现，需要检查 `uv run` 的项目安装缓存、文件时间戳或部署流程。

## 下一步计划

1. 获取 `Starting file upload via API` 后的完整 Consumer 日志，确认：
   - 上传方式及 HTTP 结果。
   - `Audio generation completed successfully`。
   - 数据库中的音频状态为 ready。
   - 音频文件 ID、时长和字幕写入成功。
2. 下载并试听 `/root/digital_human/OmniVoice/output/809193442.mp3`，重点检查 connect 位置、换行 0.1 秒停顿和显式 0.2 秒停顿。
3. 检查对应 SRT/JSON 是否与试听一致，特别关注以下边界：
   - `7.759s`。
   - `18.368s`、`20.454s`、`23.103s`。
   - 显式 pause 产生的 `28.820s -> 29.020s`。
4. 将 21.4 秒参考音频裁剪为 3–10 秒，并同步更新完全匹配的参考文本，再做一次音色与耗时对比。
5. 再运行一次生成任务，观察 `Building omnivoice / Uninstalled / Installed` 是否重复。
6. 如需提升长期可诊断性，为每个 WhisperX 分组增加成功、失败、耗时和回退原因日志，并补充自动化测试；实施前先确认需求。

## 踩过的坑

1. **把路径配置责任放在 xc-server。** 重构移除机器绝对路径默认值后，Consumer 没有传 CLI 参数或环境变量，导致 `WhisperX alignment is not configured`。最终改为 OmniVoice 根目录 YAML 自管理路径。
2. **只看 `whisperx=True` 就认定全部对齐成功。** 当前该字段本质上表示 aligner 对象已创建；分组异常会被静默回退。此次还必须结合 7.242 秒耗时、相同采样长度和字幕内部边界变化判断实际成功。
3. **误判 FunASR 固定日志为联网下载。** `download models from model hub: ms` 是固定文案；应结合明确本地 snapshot、`model.pt` 和 `All keys matched successfully` 判断。真正下载通常还会出现 `Downloading ... files`。
4. **把 `no_safe_low_energy_edit` 当成失败。** 它只表示没有安全的低能量剪切点，原始候选仍可被接受并选中。
5. **把 `active_start_seconds` 当成固定默认值。** 它是人工 padding 前从实际生成波形动态测量的结果，本次为 0；`subtitle_offset=auto` 使用该测量值。
6. **用脚本成功代替完整业务成功。** `Python script executed successfully` 只证明 TTS CLI 退出码为 0；仍需确认上传、写库、ready 状态和试听。
7. **参考音频警告不是失败原因，但不能忽略。** 21.4 秒参考音频不阻塞任务，却会重复影响 5 次 TTS 调用的性能和潜在音质。
8. **README 中曾保留已失效的机器默认路径说明。** 当前真相以 CLI 代码和 `config.yaml` 为准，文档已修正。

## 关键文件

- `config.yaml`
- `omnivoice/utils/runtime_config.py`
- `omnivoice/cli/cli_infer.py`
- `omnivoice/utils/whisperx_alignment.py`
- `omnivoice/utils/synthesis_orchestrator.py`
- `omnivoice/utils/connect_candidate_pipeline.py`
- `tests/test_runtime_config.py`
- `tests/test_connect_cli_integration.py`
- `pyproject.toml`
- `uv.lock`
- `README.md`

## 修订记录

| 修改时间（Asia/Shanghai） | 修改小结 |
| --- | --- |
| 2026-08-09 23:58:09 | 首次创建 HANDOFF；整理 WhisperX YAML 配置、自动化与服务器验收、未闭环上传/试听、下一步计划和已知坑。 |
