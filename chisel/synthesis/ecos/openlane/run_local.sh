#!/bin/bash
# 使用本地 OpenLane 安装运行

set -e

echo "=========================================="
echo "OpenLane ASIC 流程 (本地安装)"
echo "=========================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_NAME="asic_top"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 检查 OpenLane 是否已安装
if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "❌ OpenLane 未安装在 $OPENLANE_ROOT"
    echo "请先安装 OpenLane"
    exit 1
fi

echo "✅ OpenLane 已安装: $OPENLANE_ROOT"

# 准备设计文件
echo ""
echo "准备设计文件..."
DESIGN_PATH="$OPENLANE_ROOT/designs/$DESIGN_NAME"
mkdir -p "$DESIGN_PATH/src"

cp "$SCRIPT_DIR/config.json" "$DESIGN_PATH/"
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"

echo "✅ 设计文件已准备"
echo "  设计路径: $DESIGN_PATH"

# 运行 OpenLane
echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo ""

cd "$OPENLANE_ROOT"

# 使用 Docker 运行
sudo make mount &
sleep 5

# 在容器中运行流程
sudo docker exec -it openlane bash -c "
    cd /openlane && \
    ./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG
"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果位置: $DESIGN_PATH/runs/$RUN_TAG/"
echo ""
echo "查看 GDSII:"
echo "  ls -lh $DESIGN_PATH/runs/$RUN_TAG/results/final/gds/"
echo ""
echo "查看报告:"
echo "  cat $DESIGN_PATH/runs/$RUN_TAG/reports/final/summary.rpt"
echo ""
