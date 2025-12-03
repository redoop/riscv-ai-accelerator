#!/bin/bash
# OpenLane 运行脚本 v2 (使用内置 PDK)

set -e

echo "=========================================="
echo "OpenLane ASIC 流程 v2"
echo "=========================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_NAME="asic_top"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 创建工作目录
WORK_DIR="$SCRIPT_DIR/work"
mkdir -p "$WORK_DIR/$DESIGN_NAME/src"

# 复制设计文件
echo "准备设计文件..."
cp "$SCRIPT_DIR/config.json" "$WORK_DIR/$DESIGN_NAME/"
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$WORK_DIR/$DESIGN_NAME/src/"
echo "✅ 设计文件已准备"

echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo "日志: $WORK_DIR/openlane.log"
echo ""

# 运行 OpenLane (使用容器内置的 PDK)
sudo docker run --rm \
    -v "$WORK_DIR:/work" \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "
        # 使用容器内置的 PDK
        export PDK_ROOT=/build/pdk
        export PDK=sky130A
        
        # 运行流程
        ./flow.tcl -design /work/$DESIGN_NAME -tag $RUN_TAG
    " 2>&1 | tee "$WORK_DIR/openlane.log"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果位置: $WORK_DIR/$DESIGN_NAME/runs/$RUN_TAG/"
echo ""
echo "查看 GDSII:"
echo "  ls -lh $WORK_DIR/$DESIGN_NAME/runs/$RUN_TAG/results/final/gds/"
echo ""
echo "查看报告:"
echo "  cat $WORK_DIR/$DESIGN_NAME/runs/$RUN_TAG/reports/final/summary.rpt"
echo ""
