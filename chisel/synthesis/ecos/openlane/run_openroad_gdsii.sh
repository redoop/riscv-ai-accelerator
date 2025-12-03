#!/bin/bash
# OpenROAD 完整流程 - 生成 GDSII

set -e

echo "=========================================="
echo "OpenROAD -> GDSII (ICS55 55nm)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/openroad_gdsii"
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"

ICS55_PDK="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
STD_CELL="$ICS55_PDK/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"

mkdir -p "$WORK_DIR/$RUN_TAG"
cd "$WORK_DIR/$RUN_TAG"

echo "✅ 工作目录: $WORK_DIR/$RUN_TAG"

# 复制网表
cp "$SCRIPT_DIR/../project/netlist/asic_top_ics55.v" ./design.v

# 创建 OpenROAD 流程脚本
cat > flow.tcl << EOF
# OpenROAD 完整流程 -> GDSII

puts "========== 读取库文件 =========="
read_liberty $STD_CELL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_lef $STD_CELL/lef/ics55_LLSC_H7CL.lef

puts "========== 读取网表 =========="
read_verilog design.v
link_design asic_top

puts "========== 时钟约束 =========="
create_clock -name sys_clk -period 40.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 2.0 [get_clocks sys_clk]

puts "========== 布图规划 =========="
initialize_floorplan -die_area "0 0 2000 2000" -core_area "100 100 1900 1900" -site CoreSite

puts "========== IO 放置 =========="
place_pins -hor_layers metal3 -ver_layers metal4

puts "========== 电源网络 =========="
add_global_connection -net VDD -pin_pattern {^VDD$} -power
add_global_connection -net VSS -pin_pattern {^VSS$} -ground
set_voltage_domain -power VDD -ground VSS
define_pdn_grid -name main_grid
add_pdn_stripe -grid main_grid -layer metal1 -width 0.48 -pitch 5.0 -offset 2.5
add_pdn_stripe -grid main_grid -layer metal4 -width 1.6 -pitch 50.0 -offset 25.0
add_pdn_connect -grid main_grid -layers {metal1 metal4}
pdngen

puts "========== 全局布局 =========="
global_placement -density 0.30

puts "========== 详细布局 =========="
detailed_placement

puts "========== 时钟树综合 =========="
clock_tree_synthesis -root_buf BUFX4 -buf_list {BUFX2 BUFX4 BUFX8} -wire_unit 20

puts "========== 全局布线 =========="
global_route -guide_file route.guide -layers metal1:metal6 -clock_layers metal3:metal5

puts "========== 详细布线 =========="
detailed_route -guide route.guide -output_drc drc.rpt -output_maze maze.log

puts "========== 填充单元 =========="
filler_placement FILL*

puts "========== 输出 DEF =========="
write_def final.def

puts "========== 输出网表 =========="
write_verilog final.v

puts "========== 时序报告 =========="
report_checks -path_delay max -format full_clock_expanded > timing_max.rpt
report_checks -path_delay min -format full_clock_expanded > timing_min.rpt
report_tns > tns.rpt
report_wns > wns.rpt

puts "========== 面积报告 =========="
report_design_area > area.rpt

puts "========== 完成 =========="
exit
EOF

echo ""
echo "运行 OpenROAD..."
echo ""

# 运行 OpenROAD
if command -v openroad &> /dev/null; then
    openroad flow.tcl 2>&1 | tee openroad.log
else
    sudo docker run --rm -v "$WORK_DIR/$RUN_TAG:/work" -w /work \
        openroad/openroad:latest openroad flow.tcl 2>&1 | tee openroad.log
fi

# 检查 DEF 是否生成
if [ ! -f final.def ]; then
    echo "❌ DEF 文件未生成"
    exit 1
fi

echo ""
echo "========== 生成 GDSII =========="

# 使用 KLayout 生成 GDSII
if command -v klayout &> /dev/null; then
    echo "使用 KLayout 生成 GDSII..."
    klayout -b -r <(cat << 'KLAYOUT'
layout = RBA::Layout.new
layout.read("final.def")
layout.write("final.gds")
KLAYOUT
) 2>&1 | tee klayout.log
    
    if [ -f final.gds ]; then
        echo "✅ GDSII 已生成: final.gds"
    fi
else
    echo "⚠️  KLayout 未安装，跳过 GDSII 生成"
    echo "安装: sudo apt-get install klayout"
fi

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  final.def       - DEF 版图"
echo "  final.v         - 最终网表"
echo "  final.gds       - GDSII 版图 (如果 KLayout 可用)"
echo "  timing_max.rpt  - 时序报告"
echo "  area.rpt        - 面积报告"
echo ""
echo "位置: $WORK_DIR/$RUN_TAG/"
echo ""
