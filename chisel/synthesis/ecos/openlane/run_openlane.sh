#!/bin/bash
# OpenLane 完整 ASIC 流程

set -e

DESIGN_DIR="$(pwd)"
OPENLANE_ROOT="${OPENLANE_ROOT:-$HOME/OpenLane}"

echo "=========================================="
echo "OpenLane ASIC 流程"
echo "=========================================="

# 1. 检查/安装 OpenLane
if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "安装 OpenLane..."
    cd ~
    git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenLane.git
    cd OpenLane
    make
    cd "$DESIGN_DIR"
fi

echo "✅ OpenLane: $OPENLANE_ROOT"

# 2. 创建设计目录
DESIGN_NAME="asic_top"
DESIGN_PATH="$OPENLANE_ROOT/designs/$DESIGN_NAME"

echo "创建设计目录: $DESIGN_PATH"
mkdir -p "$DESIGN_PATH/src"

# 3. 复制文件
cp config.json "$DESIGN_PATH/"
cp ../project/netlist/asic_top_ics55.v "$DESIGN_PATH/src/"

echo "✅ 设计文件已复制"

# 4. 运行 OpenLane
echo ""
echo "运行 OpenLane 流程..."
echo "这将需要 1-2 小时..."
echo ""

cd "$OPENLANE_ROOT"
make mount
docker exec -it openlane bash -c "cd /openlane && ./flow.tcl -design $DESIGN_NAME -tag run_1"

echo ""
echo "✅ OpenLane 流程完成"
echo "结果: $DESIGN_PATH/runs/run_1/"
