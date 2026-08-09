# OmniVoice 安装指南

本文档提供 OmniVoice 的详细安装说明,支持多种硬件平台和安装方式。

## 📋 目录

- [系统要求](#系统要求)
- [快速安装](#快速安装)
- [详细安装步骤](#详细安装步骤)
- [平台特定说明](#平台特定说明)
- [本地模型配置](#本地模型配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)
- [卸载](#卸载)

---

## 🖥️ 系统要求

### 基本要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.10+ | 3.11 或 3.12 |
| **PyTorch** | 2.4.0+ | 2.9.0+ |
| **内存** | 8 GB | 16 GB+ |
| **磁盘空间** | 10 GB | 20 GB+ (含模型) |

### 硬件支持

| 硬件平台 | 支持状态 | 推理速度 | 备注 |
|---------|---------|---------|------|
| **NVIDIA GPU (CUDA)** | ✅ 完整支持 | ⚡ 极快 (RTF 0.025) | 推荐 CUDA 12.8 |
| **Apple Silicon (MPS)** | ✅ 完整支持 | 🚀 快 | M1/M2/M3/M4/M5 全系列 |
| **Intel Arc GPU (XPU)** | ✅ 实验性支持 | 🚀 快 | Alchemist/Battlemage |
| **CPU** | ⚠️ 仅测试用 | 🐢 极慢 | 不推荐用于生产 |

---

## 🚀 快速安装

### 方式一:使用安装脚本 (推荐)

我们提供了自动化安装脚本,支持多种平台:

```bash
# 下载脚本
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice

# NVIDIA GPU (CUDA) - 推荐
bash install.sh --cuda

# Apple Silicon (Mac M系列)
bash install.sh --mps

# Intel Arc GPU
bash install.sh --xpu

# 中国大陆用户 - 使用镜像加速
bash install.sh --cuda --mirror
```

### 方式二:使用 pip

```bash
# 1. 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装 PyTorch (根据你的硬件选择)

# NVIDIA GPU (CUDA 12.8)
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Apple Silicon
pip install torch==2.9.0 torchaudio==2.9.0

# Intel Arc GPU
pip install torch torchaudio \
    --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 3. 安装 OmniVoice
pip install omnivoice
```

### 方式三:使用 uv (更快的包管理器)

```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync

# 使用镜像 (中国大陆)
uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"
```

### 方式四:从源码安装 (开发者)

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch (同上)
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 开发模式安装
pip install -e .
```

---

## 📝 详细安装步骤

### 步骤 1:安装 Python 3.10+

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

#### macOS

```bash
# 使用 Homebrew
brew install python@3.11
```

#### Windows

从 [Python 官网](https://www.python.org/downloads/) 下载安装包,安装时勾选 "Add Python to PATH"。

### 步骤 2:创建虚拟环境

**强烈建议使用虚拟环境以避免依赖冲突。**

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 验证
which python  # 应该指向 venv/bin/python
```

### 步骤 3:安装 PyTorch

根据你的硬件平台选择对应的安装命令:

#### NVIDIA GPU (CUDA)

```bash
# CUDA 12.8 (推荐)
pip install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch==2.9.0+cu121 torchaudio==2.9.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

**验证 CUDA 是否可用:**

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Apple Silicon (M1/M2/M3/M4/M5)

```bash
pip install torch==2.9.0 torchaudio==2.9.0
```

**验证 MPS 是否可用:**

```bash
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

#### Intel Arc GPU

1. 首先安装 [Intel GPU 驱动](https://dgpu-docs.intel.com/driver/installation.html)

2. 安装 PyTorch XPU 版本:

```bash
pip install torch torchaudio \
    --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

3. 验证 XPU:

```bash
python3 -c "import torch; print(f'XPU: {torch.xpu.is_available()}')"
```

### 步骤 4:安装 OmniVoice

```bash
# 从 PyPI 安装 (稳定版)
pip install omnivoice

# 或从 GitHub 安装 (最新版)
pip install git+https://github.com/k2-fsa/OmniVoice.git

# 或开发模式 (需要先 clone)
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### 步骤 5:验证安装

```python
python3 << EOF
import torch
import torchaudio
from omnivoice import OmniVoice

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Torchaudio: {torchaudio.__version__}")
print(f"✓ OmniVoice: 已安装")

# 检查硬件加速
if torch.cuda.is_available():
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("✓ MPS: Apple Silicon")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f"✓ XPU: Intel Arc GPU")
else:
    print("⚠ 使用 CPU 模式 (速度较慢)")

print("\n安装成功! 🎉")
EOF
```

---

## 🖥️ 平台特定说明

### NVIDIA GPU

**推荐的 CUDA 版本:**
- CUDA 12.8 (最新,推荐)
- CUDA 12.1 (稳定)

**检查 CUDA 版本:**

```bash
nvidia-smi
```

**显存要求:**
- 最小: 4 GB (推理)
- 推荐: 8 GB+ (推理和微调)

**多 GPU 支持:**
OmniVoice 支持多 GPU 并行推理,使用 `omnivoice-infer-batch` 命令。

### Apple Silicon

**支持的芯片:**
- M1, M1 Pro, M1 Max, M1 Ultra
- M2, M2 Pro, M2 Max, M2 Ultra
- M3, M3 Pro, M3 Max, M3 Ultra
- M4, M4 Pro, M4 Max, M4 Ultra
- M5, M5 Pro, M5 Max, M5 Ultra（2025+）

**性能优化:**
- 使用 `float16` 或 `bfloat16` 以获得最佳性能
- 建议至少 16 GB 统一内存

**示例:**

```python
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="mps",
    dtype=torch.float16
)
```

### Intel Arc GPU

**支持的架构:**
- Alchemist (A310, A380, A750, A770)
- Battlemage (B570, B580)

**注意事项:**
- `flash_attn` 在 XPU 上不可用,会自动回退到 SDPA
- 训练功能有限,建议使用单 GPU SDPA 训练

**测试过的设备:**
- Arc A310 (4 GB)
- Arc Pro B50 (16 GB)

---

## ✅ 验证安装

### 1. 基本验证

```bash
# 检查导入
python3 -c "from omnivoice import OmniVoice; print('✓ 导入成功')"

# 检查 CLI 工具
omnivoice-demo --help
omnivoice-infer --help
omnivoice-infer-batch --help
```

### 2. 功能测试

```python
from omnivoice import OmniVoice
import torch

# 加载模型 (首次会自动下载)
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",  # 或 "mps", "xpu"
    dtype=torch.float16
)

# 生成测试音频
audio = model.generate(
    text="Hello, OmniVoice is working!",
    instruct="male, moderate pitch"
)

print(f"✓ 生成音频成功,采样率: 24000 Hz, 时长: {len(audio[0])/24000:.2f} 秒")
```

### 3. Web UI 测试

```bash
# 启动 Web UI
omnivoice-demo --ip 0.0.0.0 --port 8001

# 访问: http://localhost:8001
```

---

## 📦 本地模型配置

如果你已经有模型文件，可以配置为离线模式，避免每次下载。

### 方式一：直接使用本地路径

```python
model = OmniVoice.from_pretrained(
    "/path/to/local/OmniVoice",  # 本地模型路径
    device_map="mps",
    dtype=torch.float16
)
```

CLI 命令：

```bash
omnivoice-infer \
    --model /path/to/local/OmniVoice \
    --text "测试" \
    --output test.wav
```

### 方式二：配置 HF 缓存（推荐）

将模型放入 HuggingFace 标准缓存，之后可以直接用 `"k2-fsa/OmniVoice"` 名称：

```bash
# 1. 创建缓存目录结构
HF_CACHE="$HOME/.cache/huggingface/hub/models--k2-fsa--OmniVoice"
mkdir -p "$HF_CACHE/snapshots/local"
mkdir -p "$HF_CACHE/refs"
mkdir -p "$HF_CACHE/blobs"

# 2. 复制模型文件
cp -r /path/to/OmniVoice/* "$HF_CACHE/snapshots/local/"

# 3. 创建 refs 指向
echo "local" > "$HF_CACHE/refs/main"
```

完成后可以直接用：

```python
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",  # 自动从本地缓存加载
    device_map="mps",
    dtype=torch.float16
)
```

### 方式三：设置环境变量

```bash
# 设置 HF 离线模式
export HF_HUB_OFFLINE=1

# 或设置镜像加速下载
export HF_ENDPOINT="https://hf-mirror.com"
```

### 方式四：统一初始化离线缓存（推荐）

仓库提供统一脚本，覆盖 OmniVoice 主模型、audio tokenizer、FunASR `fa-zh`
和 WhisperX 中文对齐模型。联网机器执行下载和校验：

```bash
uv run python scripts/prepare_offline_cache.py \
    --output-manifest /data/omnivoice-cache-manifest.json
```

将缓存复制到 GPU 服务器后，只做本地校验，不会访问网络：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/prepare_offline_cache.py \
    --check-only \
    --output-manifest /root/digital_human/omnivoice-cache-manifest.json
```

服务器默认检查 `/root/.cache/modelscope/models/iic--speech_timestamp_prediction-v1-16k-offline/snapshots/v2.0.4` 中的 FunASR 模型，以及 `/opt/app/aining/digital_human/whisperx/models/zh` 中的 WhisperX 模型。OmniVoice 运行时只接受完整的本地 FunASR snapshot；目录或 `model.pt`、配置文件缺失时会立即报错并提示手动下载，不会自动访问 ModelScope。缓存位置不同可以设置 `MODELSCOPE_CACHE`，或通过 `--connect_aligner_model` 传入完整本地模型目录。

---

## ❓ 常见问题

### Q1: 下载模型时网络超时

**问题:** 从 HuggingFace 下载模型时速度慢或超时

**解决方案:**

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 然后运行你的代码
python3 your_script.py
```

或在代码中设置:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### Q2: CUDA out of memory

**问题:** GPU 显存不足

**解决方案:**

```python
# 使用 float16 减少显存占用
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16  # 使用 float16
)

# 使用较短的参考音频 (3-10 秒)
# 避免过长的文本
```

### Q3: ImportError: libcuda.so.1

**问题:** CUDA 库未找到

**解决方案:**

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# 或重新安装 NVIDIA 驱动
```

### Q4: MPS 不可用 (Mac)

**问题:** `torch.backends.mps.is_available()` 返回 False

**解决方案:**

1. 确保使用 Apple Silicon Mac (M1/M2/M3/M4/M5)
2. 更新 macOS 到最新版本
3. 更新 PyTorch:

```bash
pip install --upgrade torch torchaudio
```

### Q5: 安装时编译错误

**问题:** 某些依赖包编译失败

**解决方案:**

```bash
# Ubuntu/Debian - 安装编译工具
sudo apt install build-essential python3-dev

# macOS - 安装 Xcode Command Line Tools
xcode-select --install
```

### Q6: 如何使用国内镜像?

**pip 镜像:**

```bash
# 阿里云
pip install omnivoice -i https://mirrors.aliyun.com/pypi/simple/

# 清华
pip install omnivoice -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**uv 镜像:**

```bash
uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"
```

---

## 🗑️ 卸载

### 卸载 OmniVoice

```bash
pip uninstall omnivoice
```

### 卸载 PyTorch

```bash
pip uninstall torch torchaudio
```

### 删除虚拟环境

```bash
# 退出虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf venv/
```

### 清理缓存

```bash
# 清理 pip 缓存
pip cache purge

# 清理 HuggingFace 缓存 (可选,会删除下载的模型)
rm -rf ~/.cache/huggingface/
```

---

## 📚 相关资源

- [GitHub 仓库](https://github.com/k2-fsa/OmniVoice)
- [HuggingFace 模型](https://huggingface.co/k2-fsa/OmniVoice)
- [在线演示](https://huggingface.co/spaces/k2-fsa/OmniVoice)
- [Google Colab](https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb)
- [API 文档](https://github.com/k2-fsa/OmniVoice#python-api)
- [论文](https://arxiv.org/abs/2604.00688)

---

## 🆘 获取帮助

如果遇到问题:

1. 查看 [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues)
2. 提交新的 Issue,附上:
   - 操作系统和版本
   - Python 版本
   - PyTorch 版本
   - 完整的错误信息
   - 重现步骤

---

**祝你使用愉快! 🎉**
