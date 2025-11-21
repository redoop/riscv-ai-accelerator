#!/bin/bash
# 快速修复：重新上传并构建（使用正确的设备）

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        快速修复：AFI 设备兼容性问题                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 步骤 1：检查 Verilog 是否已生成
echo "步骤 1/5：检查 Verilog 文件..."
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VERILOG_DIR="$REPO_ROOT/chisel/generated/simple_edgeaisoc"
echo "检查目录: $VERILOG_DIR"
if [ ! -d "$VERILOG_DIR" ] || [ -z "$(ls -A $VERILOG_DIR 2>/dev/null)" ]; then
    echo "❌ 未找到生成的 Verilog 文件"
    echo ""
    echo "请先生成 Verilog："
    echo "  cd $REPO_ROOT/chisel"
    echo "  sbt 'runMain riscv.ai.SimpleEdgeAiSoCMain'"
    exit 1
fi
echo "✓ Verilog 文件已生成: $(ls $VERILOG_DIR/*.sv | wc -l) 个文件"
echo ""

# 步骤 2：重新上传项目
echo "步骤 2/5：上传项目到 F2 实例..."
cd "$SCRIPT_DIR"
bash upload_project.sh
echo ""

# 步骤 3：停止旧构建
echo "步骤 3/5：停止旧的构建进程..."
source .f2_instance_info
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} << 'ENDSSH'
echo "停止 Vivado 进程..."
pkill -f vivado || true
echo "✓ 已停止"
ENDSSH
echo ""

# 步骤 4：清理并重新构建
echo "步骤 4/5：启动新构建（使用 xcvu9p 设备）..."
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} << 'ENDSSH'
cd fpga-project/scripts

# 清理旧构建
rm -rf ../build/checkpoints/*
rm -rf ../build/logs/*

# 设置 Vivado 环境
export PATH="/tools/Xilinx/2025.1/Vivado/bin:$PATH"

# 创建日志目录
mkdir -p ../build/logs

# 启动构建（使用 xcvu9p 设备）
echo "启动 Vivado 构建（xcvu9p 设备）..."
nohup vivado -mode batch -source build_fpga.tcl > ../build/logs/vivado_build.log 2>&1 &

BUILD_PID=$!
echo "构建进程 PID: $BUILD_PID"
echo $BUILD_PID > ../build/logs/build.pid

echo ""
echo "✓ 构建已启动"
ENDSSH
echo ""

# 步骤 5：显示监控命令
echo "步骤 5/5：监控构建进度"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              构建已启动！                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "监控构建："
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "  tail -f fpga-project/build/logs/vivado_build.log"
echo ""
echo "或使用自动监控："
echo "  cd $SCRIPT_DIR"
echo "  bash continuous_monitor.sh"
echo ""
echo "预计时间：2-4 小时"
echo "预计成本：\$2-4"
echo ""
echo "关键变化："
echo "  ✓ 使用 xcvu9p 设备（兼容 AFI）"
echo "  ✓ 修复了 Verilog 文件路径"
echo "  ✓ 修复了约束文件路径"
echo ""
