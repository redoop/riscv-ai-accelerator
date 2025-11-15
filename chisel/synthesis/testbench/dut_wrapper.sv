`timescale 1ns/1ps

module SimpleEdgeAiSoC_dut (
  input  logic        clock,
  input  logic        reset,
  input  logic        io_uart_rx,
  output logic        io_uart_tx,
  input  logic [31:0] io_gpio_in,
  output logic [31:0] io_gpio_out,
  output logic [31:0] io_gpio_oe,
  output logic        io_compact_irq,
  output logic        io_bitnet_irq,
  output logic        io_trap
);
  assign io_gpio_oe = 32h0
