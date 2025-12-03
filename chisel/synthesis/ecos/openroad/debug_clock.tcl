# 调试时钟网络

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"

puts "读取设计..."
read_lef "tech_complete.lef"
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

puts "\n=== 时钟端口 ==="
foreach port [get_ports *clk*] {
    puts "  [get_name $port]"
}

puts "\n=== 时钟网络 ==="
foreach net [get_nets sys_clk*] {
    set name [get_name $net]
    set pins [get_pins -of_objects $net]
    puts "  $name (pins: [llength $pins])"
}

puts "\n=== 查找 DFF 单元 ==="
set dff_cells [get_cells -hierarchical -filter "ref_name=~DFF*"]
puts "  DFF 单元数量: [llength $dff_cells]"
if {[llength $dff_cells] > 0} {
    set sample [lindex $dff_cells 0]
    puts "  示例单元: [get_name $sample]"
    set ck_pin [get_pins [get_name $sample]/CK]
    if {$ck_pin != ""} {
        set net [get_nets -of_objects $ck_pin]
        if {$net != ""} {
            puts "  时钟网络: [get_name $net]"
        }
    }
}

puts "\n完成"
