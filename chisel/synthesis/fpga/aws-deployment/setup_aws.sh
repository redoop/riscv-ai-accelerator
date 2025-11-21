#!/bin/bash
# AWS FPGA 环境配置脚本

set -e

echo "=========================================="
echo "AWS FPGA 环境配置"
echo "=========================================="

# 检查是否在 AWS 实例上
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
if [[ ! "$INSTANCE_TYPE" =~ ^f2\. ]]; then
    echo "警告：当前不在 F2 实例上，某些功能可能不可用"
fi

# 克隆 AWS FPGA 仓库（如果不存在）
if [ ! -d "$HOME/aws-fpga" ]; then
    echo "克隆 AWS FPGA 仓库..."
    cd $HOME
    git clone https://github.com/aws/aws-fpga.git
fi

# 设置环境变量
echo "设置 AWS FPGA 环境..."
cd $HOME/aws-fpga
source sdk_setup.sh
source hdk_setup.sh

# 检查 Vivado
if ! command -v vivado &> /dev/null; then
    echo "错误：未找到 Vivado，请安装 Xilinx Vivado 2021.2 或更高版本"
    exit 1
fi

echo "Vivado 版本："
vivado -version | head -1

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo "安装 AWS CLI..."
    pip install awscli
fi

echo "AWS CLI 版本："
aws --version

# 创建工作目录
WORK_DIR="$HOME/fpga_work"
mkdir -p $WORK_DIR
echo "工作目录：$WORK_DIR"

# 创建 S3 bucket（用于存储 AFI）
BUCKET_NAME="fpga-afi-$(date +%s)"
echo "创建 S3 bucket: $BUCKET_NAME"
aws s3 mb s3://$BUCKET_NAME || echo "Bucket 可能已存在"

# 保存配置
cat > $HOME/.fpga_config << EOF
export AWS_FPGA_REPO=$HOME/aws-fpga
export FPGA_WORK_DIR=$WORK_DIR
export AFI_BUCKET=$BUCKET_NAME
EOF

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo "请运行以下命令加载环境："
echo "  source $HOME/.fpga_config"
echo "  source $HOME/aws-fpga/sdk_setup.sh"
echo ""
