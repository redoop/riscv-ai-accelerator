#!/bin/bash
# OpenLane 批处理运行脚本 (非交互式)

set -e

echo "=========================================="
echo "OpenLane ASIC 流程 (批处理模式)"
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

echo "✅ 设计已准备: $DESIGN_PATH"

# 运行 OpenLane (非交互式)
echo ""
echo "运行 OpenLane (批处理模式)..."
echo "预计时间: 1-2 小时"
echo "日志: $DESIGN_PATH/runs/$RUN_TAG/openlane.log"
echo ""

cd "$OPENLANE_ROOT"

# 使用非交互式 Docker 运行
sudo docker run --rm \
    -v "$OPENLANE_ROOT:/openlane" \
    -v "$OPENLANE_ROOT/designs:/openlane/install" \
    -v "$HOME:/home/$USER" \
    -e PDK_ROOT=/home/$USER/.ciel \
    -e PDK=sky130A \
    --user $(id -u):$(id -g) \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG" \
    2>&1 | tee "$DESIGN_PATH/runs/$RUN_TAG/openlane.log"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果: $DESIGN_PATH/runs/$RUN_TAG/"
echo ""
echo "GDSII: $DESIGN_PATH/runs/$RUN_TAG/results/final/gds/"
echo "报告: $DESIGN_PATH/runs/$RUN_TAG/reports/final/"
echo ""
