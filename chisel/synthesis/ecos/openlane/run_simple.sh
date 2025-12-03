#!/bin/bash
# OpenLane 简单运行脚本

set -e

echo "=========================================="
echo "OpenLane ASIC 流程 - 简化版"
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
echo ""

# 运行 OpenLane Docker 容器
sudo docker run --rm \
    -v "$WORK_DIR:/designs" \
    -e PDK=sky130A \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "cd /openlane && ./flow.tcl -design /designs/$DESIGN_NAME -tag $RUN_TAG"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果位置: $WORK_DIR/$DESIGN_NAME/runs/$RUN_TAG/"
echo ""
