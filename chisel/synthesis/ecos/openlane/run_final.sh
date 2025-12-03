#!/bin/bash
# OpenLane 最终运行脚本

set -e

echo "=========================================="
echo "OpenLane ASIC 流程"
echo "=========================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_NAME="asic_top"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 检查 OpenLane
if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "❌ OpenLane 未安装"
    exit 1
fi

echo "✅ OpenLane: $OPENLANE_ROOT"

# 准备设计
echo ""
echo "准备设计文件..."
DESIGN_PATH="$OPENLANE_ROOT/designs/$DESIGN_NAME"
mkdir -p "$DESIGN_PATH/src"

cp "$SCRIPT_DIR/config.json" "$DESIGN_PATH/"
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"

echo "✅ 设计已准备"

# 运行 OpenLane
echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo ""

cd "$OPENLANE_ROOT"

# 使用 Docker 运行 (使用容器内置 PDK)
sudo docker run --rm \
    -v "$OPENLANE_ROOT:/openlane" \
    -v "$OPENLANE_ROOT/designs:/openlane/install" \
    -e PDK_ROOT=/build/pdk \
    -e PDK=sky130A \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果: $DESIGN_PATH/runs/$RUN_TAG/"
echo ""
