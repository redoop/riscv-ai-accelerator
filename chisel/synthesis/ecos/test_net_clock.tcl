# 测试在网络上创建时钟

read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

puts "\n========== Test 1: Create clock on net =========="
if {[catch {
    create_clock -name sys_clk -period 10.0 [get_nets sys_clk]
    puts "SUCCESS: Clock created on net"
} err]} {
    puts "FAILED: $err"
}

puts "\n========== Test 2: Check for registers =========="
set regs [get_cells -hier -filter "is_sequential==true"]
puts "Found registers: [llength [query_objects $regs]]"

puts "\n========== Done =========="
