#!/bin/bash
# OpenLane with ICS55 PDK v2

set -e

echo "=========================================="
echo "OpenLane - ICS55 55nm PDK"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_NAME="asic_top_ics55"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 创建工作目录
WORK_DIR="$SCRIPT_DIR/work_ics55"
mkdir -p "$WORK_DIR/$DESIGN_NAME/src"
mkdir -p "$WORK_DIR/pdk/ics55"

# PDK 路径
ICS55_PDK="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
STD_CELL_PATH="$ICS55_PDK/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"

echo "✅ ICS55 PDK: $ICS55_PDK"

# 复制设计文件
echo ""
echo "准备设计文件..."
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$WORK_DIR/$DESIGN_NAME/src/"

# 复制 PDK 文件到工作目录
echo "准备 PDK 文件..."
cp "$STD_CELL_PATH/liberty/"*.lib "$WORK_DIR/pdk/ics55/"
cp "$STD_CELL_PATH/lef/"*.lef "$WORK_DIR/pdk/ics55/"

# 创建配置文件
cat > "$WORK_DIR/$DESIGN_NAME/config.json" << 'EOF'
{
  "DESIGN_NAME": "asic_top",
  "VERILOG_FILES": "dir::src/asic_top_ics55.v",
  "CLOCK_PORT": "sys_clk_i_pad",
  "CLOCK_PERIOD": 40.0,
  
  "PDK": "ics55",
  "STD_CELL_LIBRARY": "ics55_LLSC_H7CL",
  
  "FP_CORE_UTIL": 30,
  "FP_ASPECT_RATIO": 1,
  "FP_SIZING": "absolute",
  "DIE_AREA": "0 0 2000 2000",
  
  "PL_TARGET_DENSITY": 0.30,
  "ROUTING_CORES": 4,
  
  "LIB_SYNTH": "/work/pdk/ics55/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  "LIB_FASTEST": "/work/pdk/ics55/ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib",
  "LIB_SLOWEST": "/work/pdk/ics55/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib",
  "LIB_TYPICAL": "/work/pdk/ics55/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  
  "TECH_LEF": "/work/pdk/ics55/ics55_LLSC_H7CL.lef",
  "CELL_LEF": "/work/pdk/ics55/ics55_LLSC_H7CL.lef",
  
  "SYNTH_STRATEGY": "AREA 0",
  "SYNTH_MAX_FANOUT": 6,
  
  "FP_PDN_VPITCH": 25,
  "FP_PDN_HPITCH": 25,
  
  "GLB_RT_ADJUSTMENT": 0.1,
  
  "RUN_CVC": false,
  "QUIT_ON_TIMING_VIOLATIONS": false,
  "QUIT_ON_MAGIC_DRC": false,
  "QUIT_ON_LVS_ERROR": false
}
EOF

echo "✅ 设计已准备"

# 运行 OpenLane
echo ""
echo "运行 OpenLane..."
echo "预计时间: 1-2 小时"
echo ""

sudo docker run --rm \
    -v "$WORK_DIR:/work" \
    -e PDK_ROOT=/work/pdk \
    -e PDK=ics55 \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "cd /work && ./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG" \
    2>&1 | tee "$WORK_DIR/openlane_$RUN_TAG.log"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果: $WORK_DIR/$DESIGN_NAME/runs/$RUN_TAG/"
echo "日志: $WORK_DIR/openlane_$RUN_TAG.log"
echo ""
