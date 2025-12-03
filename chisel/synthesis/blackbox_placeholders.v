// Placeholder modules for black boxes - auto-generated

module uart_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  input wire uart_rx, output wire uart_tx
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign uart_tx = 1'b1;
endmodule

module spi_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  output wire spi_sck, output wire spi_ss, output wire spi_mosi,
  input wire spi_miso, output wire spi_irq_out
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign spi_sck = 1'b0;
  assign spi_ss = 1'b1;
  assign spi_mosi = 1'b0;
  assign spi_irq_out = 1'b0;
endmodule

module gpio_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  output wire [15:0] gpio_out, input wire [15:0] gpio_in,
  output wire [7:0] gpio_seg_0, output wire [7:0] gpio_seg_1,
  output wire [7:0] gpio_seg_2, output wire [7:0] gpio_seg_3,
  output wire [7:0] gpio_seg_4, output wire [7:0] gpio_seg_5,
  output wire [7:0] gpio_seg_6, output wire [7:0] gpio_seg_7
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign gpio_out = 16'h0;
  assign gpio_seg_0 = 8'h0;
  assign gpio_seg_1 = 8'h0;
  assign gpio_seg_2 = 8'h0;
  assign gpio_seg_3 = 8'h0;
  assign gpio_seg_4 = 8'h0;
  assign gpio_seg_5 = 8'h0;
  assign gpio_seg_6 = 8'h0;
  assign gpio_seg_7 = 8'h0;
endmodule

module ps2_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  input wire ps2_clk, input wire ps2_data
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
endmodule

module vga_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  output wire [7:0] vga_r, output wire [7:0] vga_g, output wire [7:0] vga_b,
  output wire vga_hsync, output wire vga_vsync, output wire vga_valid
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign vga_r = 8'h0;
  assign vga_g = 8'h0;
  assign vga_b = 8'h0;
  assign vga_hsync = 1'b0;
  assign vga_vsync = 1'b0;
  assign vga_valid = 1'b0;
endmodule

module psram_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  output wire qspi_sck, output wire qspi_ce_n, inout wire [3:0] qspi_dio
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign qspi_sck = 1'b0;
  assign qspi_ce_n = 1'b1;
  assign qspi_dio = 4'bz;
endmodule

module sdram_top_apb (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  output wire in_pready, output wire in_pslverr, output wire [31:0] in_prdata,
  output wire sdram_clk, output wire sdram_cke, output wire sdram_cs,
  output wire sdram_ras, output wire sdram_cas, output wire sdram_we,
  output wire [12:0] sdram_a, output wire [1:0] sdram_ba,
  output wire [1:0] sdram_dqm, inout wire [15:0] sdram_dq
);
  assign in_pready = 1'b1;
  assign in_pslverr = 1'b0;
  assign in_prdata = 32'h0;
  assign sdram_clk = 1'b0;
  assign sdram_cke = 1'b0;
  assign sdram_cs = 1'b1;
  assign sdram_ras = 1'b1;
  assign sdram_cas = 1'b1;
  assign sdram_we = 1'b1;
  assign sdram_a = 13'h0;
  assign sdram_ba = 2'h0;
  assign sdram_dqm = 2'h0;
  assign sdram_dq = 16'bz;
endmodule

module apb_delayer (
  input wire clock, input wire reset,
  input wire in_psel, input wire in_penable, input wire in_pwrite,
  input wire [31:0] in_paddr, input wire [2:0] in_pprot,
  input wire [31:0] in_pwdata, input wire [3:0] in_pstrb,
  input wire in_pready, input wire in_pslverr, input wire [31:0] in_prdata,
  output wire out_psel, output wire out_penable, output wire out_pwrite,
  output wire [31:0] out_paddr, output wire [2:0] out_pprot,
  output wire [31:0] out_pwdata, output wire [3:0] out_pstrb,
  output wire out_pready, output wire out_pslverr, output wire [31:0] out_prdata
);
  assign out_psel = in_psel;
  assign out_penable = in_penable;
  assign out_pwrite = in_pwrite;
  assign out_paddr = in_paddr;
  assign out_pprot = in_pprot;
  assign out_pwdata = in_pwdata;
  assign out_pstrb = in_pstrb;
  assign out_pready = in_pready;
  assign out_pslverr = in_pslverr;
  assign out_prdata = in_prdata;
endmodule


module flash (
  output wire sck, output wire ss, output wire mosi, input wire miso
);
  assign sck = 1'b0;
  assign ss = 1'b1;
  assign mosi = 1'b0;
endmodule

// bitrev is used as flash module instance name
module bitrev (
  output wire sck, output wire ss, output wire mosi, input wire miso
);
  assign sck = 1'b0;
  assign ss = 1'b1;
  assign mosi = 1'b0;
endmodule

module psram (
  output wire sck, output wire ce_n, inout wire [3:0] dio
);
  assign sck = 1'b0;
  assign ce_n = 1'b1;
  assign dio = 4'bz;
endmodule

module sdram (
  input wire clk, input wire cke, input wire cs,
  input wire ras, input wire cas, input wire we,
  input wire [12:0] a, input wire [1:0] ba,
  input wire [1:0] dqm, inout wire [15:0] dq
);
  assign dq = 16'bz;
endmodule
