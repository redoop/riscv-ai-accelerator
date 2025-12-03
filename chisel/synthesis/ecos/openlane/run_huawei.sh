#!/bin/bash
# OpenLane - 华为云镜像 + ICS55 PDK

set -e

echo "=========================================="
echo "OpenLane - 华为云镜像"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENLANE_ROOT="$HOME/OpenLane"
DESIGN_NAME="asic_top_ics55"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

# 华为云镜像
HUAWEI_IMAGE="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/ghcr.io/the-openroad-project/openlane:ff5509f65b17bfa4068d5336495ab1718987ff69"

# PDK 路径
ICS55_PDK="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
STD_CELL="$ICS55_PDK/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"

echo "✅ 华为云镜像: $HUAWEI_IMAGE"
echo "✅ ICS55 PDK: $ICS55_PDK"

# 准备设计
echo ""
echo "准备设计..."
DESIGN_PATH="$OPENLANE_ROOT/designs/$DESIGN_NAME"
mkdir -p "$DESIGN_PATH/src" "$DESIGN_PATH/pdk"

cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" "$DESIGN_PATH/src/"
cp "$STD_CELL/liberty/"*.lib "$DESIGN_PATH/pdk/"
cp "$STD_CELL/lef/"*.lef "$DESIGN_PATH/pdk/"

cat > "$DESIGN_PATH/config.json" << 'EOF'
{
  "DESIGN_NAME": "asic_top",
  "VERILOG_FILES": "dir::src/asic_top_ics55.v",
  "CLOCK_PORT": "sys_clk_i_pad",
  "CLOCK_PERIOD": 40.0,
  "FP_CORE_UTIL": 30,
  "FP_SIZING": "absolute",
  "DIE_AREA": "0 0 2000 2000",
  "PL_TARGET_DENSITY": 0.30,
  "LIB_SYNTH": "dir::pdk/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  "LIB_TYPICAL": "dir::pdk/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib",
  "TECH_LEF": "dir::pdk/ics55_LLSC_H7CL.lef",
  "CELL_LEF": "dir::pdk/ics55_LLSC_H7CL.lef",
  "RUN_CVC": false,
  "QUIT_ON_TIMING_VIOLATIONS": false
}
EOF

echo "✅ 设计已准备"

# 运行 OpenLane
echo ""
echo "运行 OpenLane (华为云镜像)..."
echo ""

cd "$OPENLANE_ROOT"

sudo docker run --rm \
    -v "$OPENLANE_ROOT:/openlane" \
    -v "$OPENLANE_ROOT/designs:/openlane/install" \
    "$HUAWEI_IMAGE" \
    bash -c "./flow.tcl -design $DESIGN_NAME -tag $RUN_TAG" \
    2>&1 | tee "$DESIGN_PATH/openlane_$RUN_TAG.log"

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "结果: $DESIGN_PATH/runs/$RUN_TAG/"
echo ""
