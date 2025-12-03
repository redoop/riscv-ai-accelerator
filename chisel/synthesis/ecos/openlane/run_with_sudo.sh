#!/bin/bash
# OpenLane 运行脚本 (使用 sudo)

echo "=========================================="
echo "OpenLane ASIC 流程"
echo "=========================================="

# 设置路径
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_DIR="$(pwd)"

# 检查 OpenLane 是否已安装
if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "❌ OpenLane 未安装"
    echo "请先运行: ./quick_start.sh"
    exit 1
fi

echo "✅ OpenLane 已安装: $OPENLANE_ROOT"

# 准备设计文件
echo ""
echo "准备设计文件..."
DESIGN_PATH="$OPENLANE_ROOT/designs/asic_top"
mkdir -p "$DESIGN_PATH/src"

cp "$DESIGN_DIR/config.json" "$DESIGN_PATH/"
cp "$DESIGN_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"

echo "✅ 设计文件已准备"

# 运行 OpenLane
echo ""
echo "运行 OpenLane 流程..."
echo "预计时间: 1-2 小时"
echo ""

cd "$OPENLANE_ROOT"

# 使用 sudo 运行 Docker
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

sudo docker run --rm \
    -v "$OPENLANE_ROOT:/openlane" \
    -v "$OPENLANE_ROOT/designs:/openlane/install" \
    -e PDK_ROOT=/openlane/pdks \
    -e PDK=sky130A \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "cd /openlane && ./flow.tcl -design asic_top -tag $RUN_TAG"

echo ""
echo "=========================================="
echo "✅ OpenLane 流程完成"
echo "=========================================="
echo ""
echo "结果位置: $DESIGN_PATH/runs/$RUN_TAG/"
echo ""
echo "查看结果:"
echo "  cd $DESIGN_PATH/runs/$RUN_TAG/results/final"
echo "  ls -lh gds/"
echo ""
