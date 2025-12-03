# 测试 LEF 文件

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set TECH_LEF "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"

puts "读取 LEF: $TECH_LEF"
read_lef $TECH_LEF

puts "\n可用的 sites:"
set db [ord::get_db]
set tech [$db getTech]
set sites [$tech getSites]
foreach site $sites {
    puts "  - [$site getName]"
}

puts "\n完成"
