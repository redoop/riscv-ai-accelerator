// ysyxSoC Wrapper for SimpleEdgeAiSoC
// Minimal wrapper that instantiates the complete SoC

module ysyx_26000001 (
  input         clock,
  input         reset,
  
  // IFU (Instruction Fetch Unit) Interface - unused in standalone mode
  output        io_ifu_reqValid,
  output [31:0] io_ifu_addr,
  input         io_ifu_respValid,
  input  [31:0] io_ifu_rdata,
  
  // LSU (Load/Store Unit) Interface - unused in standalone mode
  output        io_lsu_reqValid,
  output [31:0] io_lsu_addr,
  output [1:0]  io_lsu_size,
  output        io_lsu_wen,
  output [31:0] io_lsu_wdata,
  output [3:0]  io_lsu_wmask,
  input         io_lsu_respValid,
  input  [31:0] io_lsu_rdata
);

  // Tie off unused bus interfaces (SimpleEdgeAiSoC is self-contained)
  assign io_ifu_reqValid = 1'b0;
  assign io_ifu_addr = 32'h0;
  assign io_lsu_reqValid = 1'b0;
  assign io_lsu_addr = 32'h0;
  assign io_lsu_size = 2'b0;
  assign io_lsu_wen = 1'b0;
  assign io_lsu_wdata = 32'h0;
  assign io_lsu_wmask = 4'h0;

  // Instantiate complete SimpleEdgeAiSoC
  ip1_SimpleEdgeAiSoC soc (
    .clock(clock),
    .reset(reset),
    .io_uart_tx(),
    .io_uart_rx(1'b1),
    .io_lcd_spi_clk(),
    .io_lcd_spi_mosi(),
    .io_lcd_spi_cs(),
    .io_lcd_spi_dc(),
    .io_lcd_spi_rst(),
    .io_lcd_backlight(),
    .io_gpio_out(),
    .io_gpio_in(16'h0),
    .io_trap(),
    .io_compact_irq(),
    .io_bitnet_irq(),
    .io_uart_tx_irq(),
    .io_uart_rx_irq(),
    .io_flash_spi_clk(),
    .io_flash_spi_mosi(),
    .io_flash_spi_miso(1'b0),
    .io_flash_spi_cs(),
    .io_psram_spi_clk(),
    .io_psram_spi_cs(),
    .io_psram_spi_mosi(),
    .io_psram_spi_miso(1'b0),
    .io_psram_spi_sio2_out(),
    .io_psram_spi_sio2_oe(),
    .io_psram_spi_sio2_in(1'b0),
    .io_psram_spi_sio3_out(),
    .io_psram_spi_sio3_oe(),
    .io_psram_spi_sio3_in(1'b0)
  );

endmodule
