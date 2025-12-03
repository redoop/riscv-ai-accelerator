#!/bin/bash
# OpenLane 一键运行脚本

echo "=========================================="
echo "OpenLane 快速开始"
echo "=========================================="

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker 已安装"

# 设置路径
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_DIR="$(pwd)"

# 1. 安装 OpenLane
if [ ! -d "$OPENLANE_ROOT" ]; then
    echo ""
    echo "步骤 1/3: 安装 OpenLane (首次运行，需要 10-30 分钟)..."
    cd ~
    git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenLane.git
    cd OpenLane
    make
    echo "✅ OpenLane 安装完成"
else
    echo "✅ OpenLane 已安装: $OPENLANE_ROOT"
fi

# 2. 准备设计
echo ""
echo "步骤 2/3: 准备设计文件..."
DESIGN_PATH="$OPENLANE_ROOT/designs/asic_top"
mkdir -p "$DESIGN_PATH/src"

cp "$DESIGN_DIR/config.json" "$DESIGN_PATH/"
cp "$DESIGN_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"

echo "✅ 设计文件已准备"

# 3. 运行 OpenLane
echo ""
echo "步骤 3/3: 运行 OpenLane 流程..."
echo "预计时间: 1-2 小时"
echo "可以使用 Ctrl+C 中断"
echo ""

cd "$OPENLANE_ROOT"

# 启动 Docker 容器
make mount &
sleep 5

# 运行流程
docker exec -it openlane bash -c "cd /openlane && ./flow.tcl -design asic_top -tag run_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=========================================="
echo "✅ OpenLane 流程完成"
echo "=========================================="
echo ""
echo "结果位置: $DESIGN_PATH/runs/"
echo ""
echo "查看结果:"
echo "  cd $DESIGN_PATH/runs/"
echo "  ls -lh */results/final/gds/"
echo ""
