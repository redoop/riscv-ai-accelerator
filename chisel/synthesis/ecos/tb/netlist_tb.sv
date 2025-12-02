`timescale 1ns / 1ps

// Post-synthesis netlist testbench for ip1_SimpleEdgeAiSoC
module soc_tb ();
  localparam real CLK_PERIOD = 10.0; // 100MHz clock

  // DUT signals
  logic reset;
  logic clock;
  logic io_uart_tx;
  logic io_uart_rx;
  logic io_lcd_spi_clk;
  logic io_lcd_spi_mosi;
  logic io_lcd_spi_cs;
  logic io_lcd_spi_dc;
  logic io_lcd_spi_rst;
  logic io_lcd_backlight;
  logic io_trap;
  logic io_compact_irq;
  logic io_bitnet_irq;
  logic io_uart_tx_irq;
  logic io_uart_rx_irq;
  logic [31:0] io_gpio_out;
  logic [31:0] io_gpio_in;

  // Clock generation
  initial begin
    clock = 1'b0;
    forever begin
      #(CLK_PERIOD / 2) clock = ~clock;
    end
  end

  // Reset task
  task sim_reset(int delay);
    reset = 1'b1;
    repeat (delay) @(posedge clock);
    #1 reset = 1'b0;
  endtask

  // DUT instantiation
`ifdef POST_SYNTHESIS
  // Post-synthesis netlist
  ip1_SimpleEdgeAiSoC u_dut (
    .reset              (reset),
    .clock              (clock),
    .io_uart_tx         (io_uart_tx),
    .io_uart_rx         (io_uart_rx),
    .io_lcd_spi_clk     (io_lcd_spi_clk),
    .io_lcd_spi_mosi    (io_lcd_spi_mosi),
    .io_lcd_spi_cs      (io_lcd_spi_cs),
    .io_lcd_spi_dc      (io_lcd_spi_dc),
    .io_lcd_spi_rst     (io_lcd_spi_rst),
    .io_lcd_backlight   (io_lcd_backlight),
    .io_trap            (io_trap),
    .io_compact_irq     (io_compact_irq),
    .io_bitnet_irq      (io_bitnet_irq),
    .io_uart_tx_irq     (io_uart_tx_irq),
    .io_uart_rx_irq     (io_uart_rx_irq),
    .io_gpio_out        (io_gpio_out),
    .io_gpio_in         (io_gpio_in)
  );
`else
  // RTL simulation
  ip1_SimpleEdgeAiSoC u_dut (
    .reset              (reset),
    .clock              (clock),
    .io_uart_tx         (io_uart_tx),
    .io_uart_rx         (io_uart_rx),
    .io_lcd_spi_clk     (io_lcd_spi_clk),
    .io_lcd_spi_mosi    (io_lcd_spi_mosi),
    .io_lcd_spi_cs      (io_lcd_spi_cs),
    .io_lcd_spi_dc      (io_lcd_spi_dc),
    .io_lcd_spi_rst     (io_lcd_spi_rst),
    .io_lcd_backlight   (io_lcd_backlight),
    .io_trap            (io_trap),
    .io_compact_irq     (io_compact_irq),
    .io_bitnet_irq      (io_bitnet_irq),
    .io_uart_tx_irq     (io_uart_tx_irq),
    .io_uart_rx_irq     (io_uart_rx_irq),
    .io_gpio_out        (io_gpio_out),
    .io_gpio_in         (io_gpio_in)
  );
`endif

  // UART loopback for simple testing
  assign io_uart_rx = io_uart_tx;

  // GPIO loopback
  assign io_gpio_in = io_gpio_out;

  // Test stimulus
  initial begin
    $display("========================================");
`ifdef POST_SYNTHESIS
    $display("Post-Synthesis Netlist Simulation");
    $display("Netlist: ip1_SimpleEdgeAiSoC");
`else
    $display("RTL Simulation");
    $display("Module: SimpleEdgeAiSoC");
`endif
    $display("Clock Period: %.2f ns (%.0f MHz)", CLK_PERIOD, 1000.0/CLK_PERIOD);
    $display("========================================");
    $display("");

    // Initialize inputs
    io_uart_rx = 1'b1;
    io_gpio_in = 32'h0;

    // Apply reset
    $display("[%0t] Applying reset...", $time);
    sim_reset(10);
    $display("[%0t] Reset released", $time);

    // Run simulation
    repeat (100) @(posedge clock);
    
    $display("");
    $display("[%0t] Simulation completed successfully!", $time);
    $display("========================================");
    $display("Signal Status:");
    $display("  io_uart_tx      = %b", io_uart_tx);
    $display("  io_lcd_spi_clk  = %b", io_lcd_spi_clk);
    $display("  io_lcd_spi_cs   = %b", io_lcd_spi_cs);
    $display("  io_trap         = %b", io_trap);
    $display("  io_compact_irq  = %b", io_compact_irq);
    $display("  io_bitnet_irq   = %b", io_bitnet_irq);
    $display("  io_gpio_out     = 0x%08h", io_gpio_out);
    $display("========================================");

    $finish;
  end

  // Waveform dump
  initial begin
`ifdef POST_SYNTHESIS
    $dumpfile("soc_tb_netlist.vcd");
`else
    $dumpfile("soc_tb_rtl.vcd");
`endif
    $dumpvars(0, soc_tb);
  end

  // Timeout watchdog
  initial begin
    #100000; // 100us timeout
    $display("");
    $display("ERROR: Simulation timeout!");
    $finish;
  end

endmodule
