# OpenROAD 时钟树综合脚本

# 读取 LEF 文件
read_lef /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL.lef
read_lef /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/IO/ICsprout_55LLULP1233_IO_251013/lef/ICSIOA_N55_3P3_1P6M1TM.lef

# 读取 Liberty 库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/IO/ICsprout_55LLULP1233_IO_251013/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 初始化布图
initialize_floorplan -die_area "0 0 1000 1000" -core_area "50 50 950 950" -site unithd

# 全局布局
global_placement

# 详细布局
detailed_placement

# 时钟树综合
clock_tree_synthesis -root_buf BUFX4H7L -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L" -wire_unit 20

# 输出
write_verilog project/netlist/asic_top_cts.v
write_def project/netlist/asic_top_cts.def

# 报告
report_clock_skew
report_checks -path_delay max

puts "\n========== CTS Complete =========="
