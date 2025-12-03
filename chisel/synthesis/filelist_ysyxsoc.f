# ysyxSoC Wrapper with AI Accelerators - File List
# Top module: ysyx_26000001

# Main wrapper
ysyxSoc/all_verilog/ysyx_26000001_with_ai.v

# SimpleEdgeAiSoC generated RTL (contains all AI accelerators and controllers)
# Note: This file should be copied from: ../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
# It contains: picorv32, ip1_SimpleCompactAccel, ip1_SimpleBitNetAccel, ip1_SPIFlash, ip1_PSRAM
ysyxSoc/all_verilog/SimpleEdgeAiSoC.sv

# ============================================================================
# ysyxSoC Peripheral Modules
# ============================================================================

# AXI/APB Infrastructure
ysyxSoc/all_verilog/apb_delayer.v
ysyxSoc/all_verilog/axi4_delayer.v

# PSRAM Controller
ysyxSoc/all_verilog/EF_PSRAM_CTRL.v
ysyxSoc/all_verilog/EF_PSRAM_CTRL_wb.v
ysyxSoc/all_verilog/psram.v
ysyxSoc/all_verilog/psram_top_apb.v

# Flash Controller
ysyxSoc/all_verilog/flash_fixed.v

# SDRAM Controller
ysyxSoc/all_verilog/sdram.v
ysyxSoc/all_verilog/sdram_axi.v
ysyxSoc/all_verilog/sdram_axi_core.v
ysyxSoc/all_verilog/sdram_axi_pmem.v
ysyxSoc/all_verilog/sdram_top_apb.v
ysyxSoc/all_verilog/sdram_top_axi.v

# SPI Controller
ysyxSoc/all_verilog/spi_top.v
ysyxSoc/all_verilog/spi_top_apb.v
ysyxSoc/all_verilog/spi_clgen.v
ysyxSoc/all_verilog/spi_shift.v
ysyxSoc/all_verilog/spi_defines.v

# UART Controller
ysyxSoc/all_verilog/uart_top_apb.v
ysyxSoc/all_verilog/uart_regs.v
ysyxSoc/all_verilog/uart_receiver.v
ysyxSoc/all_verilog/uart_transmitter.v
ysyxSoc/all_verilog/uart_rfifo.v
ysyxSoc/all_verilog/uart_tfifo.v
ysyxSoc/all_verilog/uart_sync_flops.v
ysyxSoc/all_verilog/uart_defines.v

# GPIO Controller
ysyxSoc/all_verilog/gpio_top_apb.v

# VGA Controller
ysyxSoc/all_verilog/vga_top_apb.v

# PS/2 Controller
ysyxSoc/all_verilog/ps2_top_apb.v

# Utility Modules
ysyxSoc/all_verilog/bitrev.v
ysyxSoc/all_verilog/raminfr.v

# ============================================================================
# ysyxSoC Top Level (if needed for full SoC synthesis)
# ============================================================================
# Uncomment the following line if synthesizing the complete ysyxSoC
# ysyxSoc/all_verilog/ysyxSoCFull.v
