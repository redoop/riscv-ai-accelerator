#!/bin/bash
# OpenLane 完整运行脚本 (包含 PDK 下载)

set -e

echo "=========================================="
echo "OpenLane ASIC 流程"
echo "=========================================="

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_NAME="asic_top"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 创建工作目录
WORK_DIR="$SCRIPT_DIR/work"
PDK_DIR="$WORK_DIR/pdks"
mkdir -p "$WORK_DIR/$DESIGN_NAME/src"
mkdir -p "$PDK_DIR"

# 复制设计文件
echo "准备设计文件..."
cp "$SCRIPT_DIR/config.json" "$WORK_DIR/$DESIGN_NAME/"
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$WORK_DIR/$DESIGN_NAME/src/"
echo "✅ 设计文件已准备"

# 下载 PDK (如果不存在)
if [ ! -d "$PDK_DIR/sky130A" ]; then
    echo ""
    echo "下载 SkyWater 130nm PDK..."
    echo "这可能需要 10-30 分钟..."
    
    sudo docker run --rm \
        -v "$PDK_DIR:/pdk" \
        ghcr.io/the-openroad-project/openlane:latest \
        bash -c "cd /pdk && volare enable --pdk sky130 --pdk-root /pdk 41c0908b47130d5675ff8484255b43f66463a7d6"
    
    echo "✅ PDK 下载完成"
else
    echo "✅ PDK 已存在"
fi

echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo ""

# 运行 OpenLane
sudo docker run --rm \
    -v "$WORK_DIR:/work" \
    -v "$PDK_DIR:/pdk" \
    -e PDK_ROOT=/pdk \
    -e PDK=sky130A \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "./flow.tcl -design /work/$DESIGN_NAME -tag $RUN_TAG"

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
