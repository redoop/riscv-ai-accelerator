#!/bin/bash
# 解决方案 3：使用 OpenROAD 进行完整 P&R 流程

echo "=========================================="
echo "解决方案 3：OpenROAD 完整流程"
echo "=========================================="

# 检查 OpenROAD
if ! command -v openroad &> /dev/null; then
    echo "❌ OpenROAD 未安装"
    echo "安装方法："
    echo "  Ubuntu: sudo apt install openroad"
    echo "  或访问: https://github.com/The-OpenROAD-Project/OpenROAD"
    exit 1
fi

echo "✅ OpenROAD 已安装"
echo ""

# 创建简化的 OpenROAD 流程
cat > openroad_flow.tcl << 'EOF'
# OpenROAD 简化流程

# 读取库文件
read_lef /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL.lef
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

puts "OpenROAD 流程需要完整的 P&R 步骤："
puts "1. Floorplanning"
puts "2. Placement"
puts "3. CTS"
puts "4. Routing"
puts ""
puts "这需要更多的配置和时间（1-2周学习）"
puts "建议先使用方案 1 或方案 2"
EOF

echo "OpenROAD 流程说明："
echo ""
echo "完整的 P&R 流程包括："
echo "  1. 布图规划 (Floorplanning)"
echo "  2. 布局 (Placement)"
echo "  3. 时钟树综合 (CTS)"
echo "  4. 布线 (Routing)"
echo ""
echo "学习资源："
echo "  - 官方文档: https://openroad.readthedocs.io/"
echo "  - 示例: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts"
echo ""
echo "预计时间：1-2 周学习 + 1 周实施"
echo ""
echo "建议："
echo "  1. 先使用方案 1（25MHz）验证功能"
echo "  2. 学习 OpenROAD 基础"
echo "  3. 运行完整流程达到 100MHz"

rm -f openroad_flow.tcl
