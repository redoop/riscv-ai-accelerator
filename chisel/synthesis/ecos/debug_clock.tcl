# 调试时钟定义

# 读取库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 检查 sys_clk 网络
puts "\n========== Checking sys_clk net =========="
if {[catch {get_nets sys_clk} net]} {
    puts "ERROR: sys_clk net not found: $net"
} else {
    puts "Found net: sys_clk"
}

# 尝试直接在网络上创建时钟（虚拟时钟）
puts "\n========== Method 1: Virtual clock =========="
if {[catch {
    create_clock -name sys_clk -period 10.0
    puts "Virtual clock created"
} err]} {
    puts "ERROR: $err"
}

# 尝试在输入端口上创建时钟
puts "\n========== Method 2: Clock on input port =========="
if {[catch {
    create_clock -name sys_clk_port -period 10.0 [get_ports sys_clk_i_pad]
    puts "Clock on port created"
} err]} {
    puts "ERROR: $err"
}

# 尝试使用 set_ideal_network
puts "\n========== Method 3: Ideal network =========="
if {[catch {
    set_ideal_network [get_nets sys_clk]
    puts "Ideal network set"
} err]} {
    puts "ERROR: $err"
}

puts "\n========== Done =========="
