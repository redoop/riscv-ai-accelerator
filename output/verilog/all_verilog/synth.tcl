# Yosys 综合脚本
# 日期: 2025-12-03
# 目标: ysyxSoC AI 加速器

# 读取所有 Verilog 文件（跳过 SystemVerilog）
# read_verilog -sv SimpleEdgeAiSoC.sv
read_verilog ysyx_26000001_with_ai.v
read_verilog ysyxSoCFull.v
read_verilog flash_fixed.v

# 读取外设文件
read_verilog uart_*.v
read_verilog spi_*.v
read_verilog sdram*.v
read_verilog apb_delayer.v
read_verilog axi4_delayer.v
read_verilog bitrev.v
read_verilog gpio_top_apb.v
read_verilog ps2_top_apb.v
read_verilog psram*.v
read_verilog EF_PSRAM_CTRL*.v
read_verilog vga_top_apb.v
read_verilog raminfr.v

# 设置顶层模块
hierarchy -check -top ysyxSoCTop

# 综合
synth -top ysyxSoCTop

# 统计
stat

# 输出网表
write_verilog -noattr synth_output.v

# 输出 JSON
write_json synth_output.json
