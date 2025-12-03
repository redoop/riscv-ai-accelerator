#!/bin/bash
# OpenLane with ICS55 PDK

set -e

echo "=========================================="
echo "OpenLane ASIC 流程 - ICS55 55nm PDK"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_NAME="asic_top_ics55"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# PDK 路径
ICS55_PDK="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
STD_CELL_PATH="$ICS55_PDK/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"

if [ ! -d "$OPENLANE_ROOT" ]; then
    echo "❌ OpenLane 未安装"
    exit 1
fi

if [ ! -d "$ICS55_PDK" ]; then
    echo "❌ ICS55 PDK 未找到: $ICS55_PDK"
    exit 1
fi

echo "✅ OpenLane: $OPENLANE_ROOT"
echo "✅ ICS55 PDK: $ICS55_PDK"

# 准备设计目录
echo ""
echo "准备设计文件..."
DESIGN_PATH="$OPENLANE_ROOT/designs/$DESIGN_NAME"
mkdir -p "$DESIGN_PATH/src"
mkdir -p "$DESIGN_PATH/pdk"

# 复制设计文件
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"

# 复制 PDK 文件
echo "复制 PDK 文件..."
cp "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib" "$DESIGN_PATH/pdk/"
cp "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib" "$DESIGN_PATH/pdk/"
cp "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib" "$DESIGN_PATH/pdk/"
cp "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef" "$DESIGN_PATH/pdk/"

# 创建配置文件
cat > "$DESIGN_PATH/config.json" << 'EOF'
{
  "DESIGN_NAME": "asic_top",
  "VERILOG_FILES": "dir::src/asic_top_ics55.v",
  "CLOCK_PORT": "sys_clk_i_pad",
  "CLOCK_PERIOD": 40.0,
  
  "FP_CORE_UTIL": 30,
  "FP_ASPECT_RATIO": 1,
  "FP_SIZING": "absolute",
  "DIE_AREA": "0 0 2000 2000",
  
  "PL_TARGET_DENSITY": 0.35,
  "ROUTING_CORES": 4,
  
  "LIB_SYNTH": "dir::pdk/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  "LIB_FASTEST": "dir::pdk/ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib",
  "LIB_SLOWEST": "dir::pdk/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib",
  "LIB_TYPICAL": "dir::pdk/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  
  "TECH_LEF": "dir::pdk/ics55_LLSC_H7CL.lef",
  "CELL_LEF": "dir::pdk/ics55_LLSC_H7CL.lef",
  
  "SYNTH_STRATEGY": "AREA 0",
  "SYNTH_MAX_FANOUT": 6,
  
  "FP_PDN_VPITCH": 25,
  "FP_PDN_HPITCH": 25,
  
  "GLB_RT_ADJUSTMENT": 0.1,
  "DIODE_INSERTION_STRATEGY": 3,
  
  "RUN_CVC": false,
  "QUIT_ON_TIMING_VIOLATIONS": false,
  "QUIT_ON_MAGIC_DRC": false,
  "QUIT_ON_LVS_ERROR": false
}
EOF

echo "✅ 设计已准备: $DESIGN_PATH"

# 运行 OpenLane
echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo ""

cd "$OPENLANE_ROOT"

# 使用 Docker 运行
sudo docker run --rm \
    -v "$OPENLANE_ROOT:/openlane" \
    -v "$OPENLANE_ROOT/designs:/openlane/install" \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG -overwrite" \
    2>&1 | tee "$DESIGN_PATH/openlane_$RUN_TAG.log"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果: $DESIGN_PATH/runs/$RUN_TAG/"
echo "日志: $DESIGN_PATH/openlane_$RUN_TAG.log"
echo ""
