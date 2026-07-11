#!/bin/bash

# OmniVoice 安装脚本
# 支持平台: NVIDIA GPU (CUDA), Apple Silicon (MPS), Intel Arc GPU (XPU)
# 使用方法: bash install.sh [选项]
#
# 选项:
#   --cuda       安装 NVIDIA CUDA 版本 (默认)
#   --mps        安装 Apple Silicon MPS 版本
#   --xpu        安装 Intel Arc GPU XPU 版本
#   --cpu        安装纯 CPU 版本
#   --dev        安装开发模式 (editable install)
#   --mirror     使用阿里云镜像加速
#   --help       显示帮助信息

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
OmniVoice 安装脚本

使用方法: bash install.sh [选项]

选项:
  --cuda       安装 NVIDIA CUDA 版本 (默认,需要 CUDA 12.8)
  --mps        安装 Apple Silicon MPS 版本 (Mac M1/M2/M3/M4/M5)
  --xpu        安装 Intel Arc GPU XPU 版本
  --cpu        安装纯 CPU 版本 (不推荐用于推理)
  --dev        安装开发模式 (editable install)
  --mirror     使用阿里云镜像加速 (中国大陆用户推荐)
  --help       显示此帮助信息

示例:
  bash install.sh --cuda              # 安装 CUDA 版本
  bash install.sh --mps --mirror      # 安装 MPS 版本,使用镜像
  bash install.sh --dev --cuda        # 安装开发模式

EOF
    exit 0
}

# 检查 Python 版本
check_python() {
    log_info "检查 Python 版本..."
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 Python 3,请先安装 Python 3.10 或更高版本"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
        log_error "Python 版本需要 >= 3.10,当前版本: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python 版本: $PYTHON_VERSION"
}

# 检查 pip
check_pip() {
    log_info "检查 pip..."
    if ! command -v pip3 &> /dev/null; then
        log_error "未找到 pip,请先安装 pip"
        exit 1
    fi
    log_success "pip 已安装"
}

# 创建虚拟环境 (可选)
create_venv() {
    log_info "是否创建虚拟环境? (推荐) [Y/n]"
    read -r response
    if [[ ! "$response" =~ ^[Nn]$ ]]; then
        if [ ! -d "venv" ]; then
            log_info "创建虚拟环境..."
            python3 -m venv venv
            log_success "虚拟环境已创建: venv/"
        fi

        log_info "激活虚拟环境..."
        # shellcheck disable=SC1091
        source venv/bin/activate
        log_success "虚拟环境已激活"
    fi
}

# 安装 PyTorch - NVIDIA CUDA
install_pytorch_cuda() {
    log_info "安装 PyTorch (NVIDIA CUDA 12.8)..."
    if [ "$USE_MIRROR" = true ]; then
        pip3 install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            -i https://mirrors.aliyun.com/pypi/simple/
    else
        pip3 install torch==2.9.0+cu128 torchaudio==2.9.0+cu128 \
            --extra-index-url https://download.pytorch.org/whl/cu128
    fi
    log_success "PyTorch (CUDA) 安装完成"
}

# 安装 PyTorch - Apple Silicon MPS
install_pytorch_mps() {
    log_info "安装 PyTorch (Apple Silicon MPS)..."
    if [ "$USE_MIRROR" = true ]; then
        pip3 install torch==2.9.0 torchaudio==2.9.0 \
            -i https://mirrors.aliyun.com/pypi/simple/
    else
        pip3 install torch==2.9.0 torchaudio==2.9.0
    fi
    log_success "PyTorch (MPS) 安装完成"
}

# 安装 PyTorch - Intel Arc GPU XPU
install_pytorch_xpu() {
    log_info "安装 PyTorch (Intel Arc GPU XPU)..."
    pip3 install torch torchaudio \
        --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    log_success "PyTorch (XPU) 安装完成"
}

# 安装 PyTorch - CPU only
install_pytorch_cpu() {
    log_info "安装 PyTorch (CPU only)..."
    if [ "$USE_MIRROR" = true ]; then
        pip3 install torch==2.9.0+cpu torchaudio==2.9.0+cpu \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -i https://mirrors.aliyun.com/pypi/simple/
    else
        pip3 install torch==2.9.0+cpu torchaudio==2.9.0+cpu \
            --extra-index-url https://download.pytorch.org/whl/cpu
    fi
    log_warning "CPU 版本仅用于测试,推理速度会非常慢"
    log_success "PyTorch (CPU) 安装完成"
}

# 验证 PyTorch 安装
verify_pytorch() {
    log_info "验证 PyTorch 安装..."
    python3 -c "
import torch
import torchaudio
print(f'PyTorch 版本: {torch.__version__}')
print(f'Torchaudio 版本: {torchaudio.__version__}')
if torch.cuda.is_available():
    print(f'CUDA 可用: {torch.cuda.is_available()}')
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 设备: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'MPS 可用: {torch.backends.mps.is_available()}')
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'XPU 可用: {torch.xpu.is_available()}')
    print(f'XPU 设备数: {torch.xpu.device_count()}')
else:
    print('使用 CPU 模式')
"
    log_success "PyTorch 验证通过"
}

# 安装 OmniVoice
install_omnivoice() {
    log_info "安装 OmniVoice..."

    if [ "$DEV_MODE" = true ]; then
        log_info "使用开发模式安装..."
        if [ "$USE_MIRROR" = true ]; then
            pip3 install -e . -i https://mirrors.aliyun.com/pypi/simple/
        else
            pip3 install -e .
        fi
    else
        log_info "从 PyPI 安装稳定版本..."
        if [ "$USE_MIRROR" = true ]; then
            pip3 install omnivoice -i https://mirrors.aliyun.com/pypi/simple/
        else
            pip3 install omnivoice
        fi
    fi

    log_success "OmniVoice 安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证 OmniVoice 安装..."
    python3 -c "
from omnivoice import OmniVoice
print('OmniVoice 导入成功')
print('安装验证通过!')
"
    log_success "OmniVoice 验证通过"
}

# 显示安装后提示
post_install_tips() {
    echo ""
    log_success "=========================================="
    log_success "OmniVoice 安装完成!"
    log_success "=========================================="
    echo ""
    log_info "快速开始:"
    echo "  1. 启动 Web UI: omnivoice-demo --ip 0.0.0.0 --port 8001"
    echo "  2. 单次推理: omnivoice-infer --help"
    echo "  3. Python API: from omnivoice import OmniVoice"
    echo ""
    log_info "如果下载模型时遇到网络问题,请设置 HuggingFace 镜像:"
    echo "  export HF_ENDPOINT=\"https://hf-mirror.com\""
    echo ""
    log_info "完整文档请访问: https://github.com/k2-fsa/OmniVoice"
    echo ""
}

# 主函数
main() {
    # 默认参数
    PLATFORM="cuda"
    DEV_MODE=false
    USE_MIRROR=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda)
                PLATFORM="cuda"
                shift
                ;;
            --mps)
                PLATFORM="mps"
                shift
                ;;
            --xpu)
                PLATFORM="xpu"
                shift
                ;;
            --cpu)
                PLATFORM="cpu"
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --mirror)
                USE_MIRROR=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                ;;
        esac
    done

    echo ""
    log_info "=========================================="
    log_info "OmniVoice 安装脚本"
    log_info "=========================================="
    echo ""
    log_info "安装平台: $PLATFORM"
    log_info "开发模式: $DEV_MODE"
    log_info "使用镜像: $USE_MIRROR"
    echo ""

    # 执行安装步骤
    check_python
    check_pip

    # 如果是开发模式,不创建虚拟环境(假设用户已经在虚拟环境中)
    if [ "$DEV_MODE" = false ]; then
        create_venv
    fi

    # 安装 PyTorch
    case $PLATFORM in
        cuda)
            install_pytorch_cuda
            ;;
        mps)
            install_pytorch_mps
            ;;
        xpu)
            install_pytorch_xpu
            ;;
        cpu)
            install_pytorch_cpu
            ;;
    esac

    verify_pytorch
    install_omnivoice
    verify_installation
    post_install_tips
}

# 执行主函数
main "$@"
