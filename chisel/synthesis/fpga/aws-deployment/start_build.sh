#!/bin/bash
# 在 F2 实例上启动 Vivado 构建

# 加载实例信息
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f2_instance_info"

if [ ! -f "$INFO_FILE" ]; then
    echo "❌ 错误: 未找到实例信息文件"
    echo "请先运行: ./launch_f2_vivado.sh"
    exit 1
fi

source "$INFO_FILE"

INSTANCE_IP="$PUBLIC_IP"
KEY_FILE="~/.ssh/${KEY_NAME}.pem"
USER="ubuntu"

echo "=== 启动 FPGA 构建 ==="
echo ""
echo "实例 IP: $INSTANCE_IP"
echo "预计时间: 2-4 小时"
echo "预计成本: $2.00-$4.00"
echo ""

# 启动构建（后台运行）
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} << 'ENDSSH'
cd fpga-project/scripts

# 设置 Vivado 环境
export PATH="/tools/Xilinx/2025.1/Vivado/bin:$PATH"

# 创建日志目录
mkdir -p ../build/logs

# 启动构建（使用 nohup 后台运行）
echo "启动 Vivado 构建..."
nohup vivado -mode batch -source build_fpga_f2.tcl > ../build/logs/vivado_build.log 2>&1 &

BUILD_PID=$!
echo "构建进程 PID: $BUILD_PID"
echo $BUILD_PID > ../build/logs/build.pid

echo ""
echo "构建已在后台启动！"
echo ""
echo "监控构建进度："
echo "  tail -f fpga-project/build/logs/vivado_build.log"
echo ""
echo "检查构建状态："
echo "  ps aux | grep vivado"
echo ""
ENDSSH

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║            FPGA 构建已启动！                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "监控构建："
echo "  ssh -i $KEY_FILE ${USER}@${INSTANCE_IP}"
echo "  tail -f fpga-project/build/logs/vivado_build.log"
echo ""
echo "预计完成时间：2-4 小时"
echo ""
