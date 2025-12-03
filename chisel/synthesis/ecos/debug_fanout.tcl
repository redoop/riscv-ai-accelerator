# 调试 sys_clk 网络的扇出

# 读取库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 获取 sys_clk 网络的所有扇出引脚
puts "\n========== Finding sys_clk fanout pins =========="
set net [get_nets sys_clk]
puts "Net: $net"

# 尝试获取该网络的所有引脚
if {[catch {
    set pins [get_pins -of_objects $net]
    puts "Found pins connected to sys_clk"
    
    # 尝试过滤输入引脚（时钟引脚）
    set clock_pins [get_pins -of_objects $net -filter "direction==in"]
    puts "Clock pins (direction==in): $clock_pins"
    
    # 尝试创建时钟
    if {$clock_pins != ""} {
        create_clock -name sys_clk -period 10.0 $clock_pins
        puts "Clock created on pins"
        
        # 检查是否有时序路径
        puts "\n========== Checking for paths =========="
        report_checks -path_delay max
    } else {
        puts "No input pins found"
    }
} err]} {
    puts "ERROR: $err"
}

puts "\n========== Done =========="
