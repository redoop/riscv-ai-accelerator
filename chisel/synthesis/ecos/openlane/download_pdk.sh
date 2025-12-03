#!/bin/bash
# 下载 SkyWater 130nm PDK

set -e

echo "=========================================="
echo "下载 SkyWater 130nm PDK"
echo "=========================================="

OPENLANE_ROOT="$HOME/OpenLane"
PDK_DIR="$OPENLANE_ROOT/pdks"

if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "❌ OpenLane 未安装"
    exit 1
fi

echo "✅ OpenLane: $OPENLANE_ROOT"
echo "PDK 目录: $PDK_DIR"

# 创建 PDK 目录
mkdir -p "$PDK_DIR"

echo ""
echo "开始下载 PDK..."
echo "预计时间: 10-30 分钟"
echo "大小: ~2-3 GB"
echo ""

# 使用 OpenLane 的 make pdk 命令
cd "$OPENLANE_ROOT"

sudo make pdk

echo ""
echo "=========================================="
echo "✅ PDK 下载完成"
echo "=========================================="
echo ""
echo "PDK 位置: $PDK_DIR/sky130A"
echo ""
echo "下一步: 运行 OpenLane"
echo "  cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane"
echo "  ./run_final.sh"
echo ""
