# 检查 sys_clk 网络的驱动源

# 读取库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

puts "\n========== Checking sys_clk network =========="
set net [get_nets sys_clk]
puts "Net: $net"

# 获取所有连接的引脚
if {[catch {
    set all_pins [get_pins -of_objects $net]
    puts "All pins: $all_pins"
    
    # 尝试获取输出引脚（驱动源）
    set driver_pins [get_pins -of_objects $net -filter "direction==out"]
    puts "Driver pins (direction==out): $driver_pins"
    
    # 尝试获取输入引脚（负载）
    set load_pins [get_pins -of_objects $net -filter "direction==in"]
    puts "Load pins (direction==in): $load_pins"
    
} err]} {
    puts "ERROR: $err"
}

# 尝试另一种方法：在端口上创建时钟，然后传播
puts "\n========== Method: Create clock on port and propagate =========="
if {[catch {
    create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]
    set_propagated_clock [get_clocks sys_clk]
    puts "Clock created and set to propagated"
    
    # 检查路径
    puts "\n========== Checking paths =========="
    report_checks -path_delay max -format summary
    
} err]} {
    puts "ERROR: $err"
}

puts "\n========== Done =========="
