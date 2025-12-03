// ysyxSoC Wrapper for SimpleEdgeAiSoC
// Adapts SimpleEdgeAiSoC to ysyxSoC CPU interface specification
// Student ID: 26000001 (example)

module ysyx_26000001 (
  input         clock,
  input         reset,
  
  // IFU (Instruction Fetch Unit) Interface
  output        io_ifu_reqValid,
  output [31:0] io_ifu_addr,
  input         io_ifu_respValid,
  input  [31:0] io_ifu_rdata,
  
  // LSU (Load/Store Unit) Interface
  output        io_lsu_reqValid,
  output [31:0] io_lsu_addr,
  output [1:0]  io_lsu_size,
  output        io_lsu_wen,
  output [31:0] io_lsu_wdata,
  output [3:0]  io_lsu_wmask,
  input         io_lsu_respValid,
  input  [31:0] io_lsu_rdata
);

  // Internal wires for PicoRV32 memory interface
  wire        mem_valid;
  wire        mem_instr;
  wire        mem_ready;
  wire [31:0] mem_addr;
  wire [31:0] mem_wdata;
  wire [3:0]  mem_wstrb;
  wire [31:0] mem_rdata;
  
  // Instantiate SimpleEdgeAiSoC (simplified - only PicoRV32 core)
  // Note: Full SoC has peripherals, we only expose memory interface here
  picorv32 #(
    .ENABLE_COUNTERS(1),
    .ENABLE_REGS_16_31(1),
    .ENABLE_REGS_DUALPORT(1),
    .LATCHED_MEM_RDATA(0),
    .TWO_STAGE_SHIFT(1),
    .TWO_CYCLE_COMPARE(0),
    .TWO_CYCLE_ALU(0),
    .CATCH_MISALIGN(1),
    .CATCH_ILLINSN(1),
    .ENABLE_PCPI(0),
    .ENABLE_MUL(1),
    .ENABLE_DIV(1),
    .ENABLE_FAST_MUL(1),
    .ENABLE_IRQ(1),
    .ENABLE_IRQ_QREGS(1),
    .PROGADDR_RESET(32'h30000000),  // Reset to Flash address
    .PROGADDR_IRQ(32'h00000010),
    .STACKADDR(32'h0000FFF0)
  ) cpu (
    .clk(clock),
    .resetn(~reset),
    .trap(),
    
    // Memory interface
    .mem_valid(mem_valid),
    .mem_instr(mem_instr),
    .mem_ready(mem_ready),
    .mem_addr(mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_rdata(mem_rdata),
    
    // IRQ interface (unused)
    .irq(32'h0),
    .eoi(),
    
    // PCPI interface (unused)
    .pcpi_valid(),
    .pcpi_insn(),
    .pcpi_rs1(),
    .pcpi_rs2(),
    .pcpi_wr(1'b0),
    .pcpi_rd(32'h0),
    .pcpi_wait(1'b0),
    .pcpi_ready(1'b0)
  );
  
  // Adapt PicoRV32 interface to ysyxSoC SimpleBus interface
  
  // IFU (Instruction Fetch)
  assign io_ifu_reqValid = mem_valid & mem_instr;
  assign io_ifu_addr = mem_addr;
  
  // LSU (Load/Store)
  assign io_lsu_reqValid = mem_valid & ~mem_instr;
  assign io_lsu_addr = mem_addr;
  assign io_lsu_wen = |mem_wstrb;
  assign io_lsu_wdata = mem_wdata;
  assign io_lsu_wmask = mem_wstrb;
  
  // Calculate size from wmask
  // 4'b1111 -> 2'b10 (4 bytes)
  // 4'b0011, 4'b1100 -> 2'b01 (2 bytes)
  // 4'b0001, 4'b0010, 4'b0100, 4'b1000 -> 2'b00 (1 byte)
  assign io_lsu_size = (mem_wstrb == 4'b1111) ? 2'b10 :
                       ((mem_wstrb == 4'b0011) || (mem_wstrb == 4'b1100)) ? 2'b01 :
                       2'b00;
  
  // Memory response
  wire ifu_resp = io_ifu_respValid & mem_instr;
  wire lsu_resp = io_lsu_respValid & ~mem_instr;
  
  assign mem_ready = ifu_resp | lsu_resp;
  assign mem_rdata = mem_instr ? io_ifu_rdata : io_lsu_rdata;

endmodule
