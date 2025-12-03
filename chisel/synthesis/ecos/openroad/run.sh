#!/bin/bash
# OpenROAD P&R 流程运行脚本

cd "$(dirname "$0")"

echo "=========================================="
echo "OpenROAD P&R 流程"
echo "=========================================="
echo ""

# 检查 OpenROAD
if ! command -v openroad &> /dev/null; then
    echo "❌ OpenROAD 未安装"
    exit 1
fi

echo "✅ OpenROAD 已安装: $(which openroad)"
echo ""

# 创建输出目录
mkdir -p results logs

# 选择运行模式
if [ "$1" == "all" ]; then
    echo "运行完整流程..."
    openroad -exit run_all.tcl 2>&1 | tee logs/run_all.log
elif [ "$1" == "floorplan" ]; then
    echo "运行 Floorplan..."
    openroad -exit scripts/1_floorplan.tcl 2>&1 | tee logs/1_floorplan.log
elif [ "$1" == "placement" ]; then
    echo "运行 Placement..."
    openroad -exit scripts/2_placement.tcl 2>&1 | tee logs/2_placement.log
elif [ "$1" == "cts" ]; then
    echo "运行 CTS..."
    openroad -exit scripts/3_cts.tcl 2>&1 | tee logs/3_cts.log
elif [ "$1" == "routing" ]; then
    echo "运行 Routing..."
    openroad -exit scripts/4_routing.tcl 2>&1 | tee logs/4_routing.log
else
    echo "用法:"
    echo "  ./run.sh all         - 运行完整流程"
    echo "  ./run.sh floorplan   - 只运行 Floorplan"
    echo "  ./run.sh placement   - 只运行 Placement"
    echo "  ./run.sh cts         - 只运行 CTS"
    echo "  ./run.sh routing     - 只运行 Routing"
    echo ""
    echo "推荐:"
    echo "  ./run.sh all"
    exit 1
fi

echo ""
echo "✅ 完成"
echo "日志: logs/"
echo "结果: results/"
