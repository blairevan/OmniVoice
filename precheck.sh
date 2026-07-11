#!/bin/bash
# OmniVoice 环境预检脚本
# 用法: bash precheck.sh

echo "========== OmniVoice 环境预检 =========="
echo "时间: $(date)"
echo ""

echo "--- 1. 系统信息 ---"
cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="
uname -m
echo ""

echo "--- 2. 内存 ---"
free -h | head -2
echo ""

echo "--- 3. 磁盘空间 ---"
df -h / | tail -1
echo ""

echo "--- 4. Python ---"
python3 --version 2>&1 || echo "未安装 python3"
which python3 2>/dev/null
echo ""

echo "--- 5. pip ---"
pip3 --version 2>&1 || echo "未安装 pip"
echo ""

echo "--- 6. NVIDIA 驱动 ---"
nvidia-smi 2>&1 || echo "nvidia-smi 执行失败"
echo ""

echo "--- 7. CUDA Toolkit ---"
nvcc --version 2>&1 || echo "未安装 nvcc (CUDA Toolkit)"
echo ""

echo "--- 8. PyTorch ---"
python3 -c "
import torch, torchaudio
print(f'torch=={torch.__version__}')
print(f'torchaudio=={torchaudio.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" 2>&1 || echo "PyTorch 未安装或导入失败"
echo ""

echo "--- 9. uv ---"
uv --version 2>&1 || echo "未安装 uv"
echo ""

echo "--- 10. 项目目录 ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ls -la "$SCRIPT_DIR/venv/bin/activate" 2>/dev/null && echo "venv 已存在" || echo "venv 不存在"
echo ""

echo "========== 预检完成 =========="
