// ysyxSoC Wrapper for SimpleEdgeAiSoC with AI Accelerators
// Integrates PicoRV32 + CompactAccel + BitNetAccel into ysyxSoC
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

  // ============================================================================
  // Memory Map (Hybrid: SimpleEdgeAiSoC + ysyxSoC)
  // ============================================================================
  // 0x0000_0000 - 0x0FFF_FFFF: PSRAM (ysyxSoC - larger, 256MB)
  // 0x0400_0000 - 0x047F_FFFF: PSRAM (SimpleEdgeAiSoC - 8MB, overlaps)
  // 0x1000_0000 - 0x1FFF_FFFF: SDRAM (ysyxSoC)
  // 0x2000_0000 - 0x2000_0FFF: CompactAccel (SimpleEdgeAiSoC)
  // 0x2000_1000 - 0x2000_1FFF: BitNetAccel (SimpleEdgeAiSoC)
  // 0x2000_2000 - 0x2000_2FFF: Flash Controller (SimpleEdgeAiSoC)
  // 0x2000_3000 - 0x2000_3FFF: PSRAM Controller (SimpleEdgeAiSoC)
  // 0x2000_4000 - 0x2FFF_FFFF: Other peripherals (ysyxSoC)
  // 0x3000_0000 - 0x3FFF_FFFF: Flash Memory (both, 16MB for SimpleEdgeAiSoC)

  localparam COMPACT_BASE = 32'h20000000;
  localparam COMPACT_SIZE = 32'h00001000;
  localparam BITNET_BASE  = 32'h20001000;
  localparam BITNET_SIZE  = 32'h00001000;
  localparam FLASH_CTRL_BASE = 32'h20002000;
  localparam FLASH_CTRL_SIZE = 32'h00001000;
  localparam PSRAM_CTRL_BASE = 32'h20003000;
  localparam PSRAM_CTRL_SIZE = 32'h00001000;
  localparam FLASH_MEM_BASE  = 32'h30000000;
  localparam FLASH_MEM_SIZE  = 32'h01000000;  // 16 MB
  localparam PSRAM_MEM_BASE  = 32'h04000000;
  localparam PSRAM_MEM_SIZE  = 32'h00800000;  // 8 MB

  // ============================================================================
  // PicoRV32 CPU Core
  // ============================================================================
  
  wire        mem_valid;
  wire        mem_instr;
  wire        mem_ready;
  wire [31:0] mem_addr;
  wire [31:0] mem_wdata;
  wire [3:0]  mem_wstrb;
  wire [31:0] mem_rdata;
  wire [31:0] cpu_irq;
  
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
    .PROGADDR_RESET(32'h30000000),  // Reset to Flash
    .PROGADDR_IRQ(32'h00000010),
    .STACKADDR(32'h0000FFF0)
  ) cpu (
    .clk(clock),
    .resetn(~reset),
    .trap(),
    .mem_valid(mem_valid),
    .mem_instr(mem_instr),
    .mem_ready(mem_ready),
    .mem_addr(mem_addr),
    .mem_wdata(mem_wdata),
    .mem_wstrb(mem_wstrb),
    .mem_rdata(mem_rdata),
    .irq(cpu_irq),
    .eoi(),
    .pcpi_valid(),
    .pcpi_insn(),
    .pcpi_rs1(),
    .pcpi_rs2(),
    .pcpi_wr(1'b0),
    .pcpi_rd(32'h0),
    .pcpi_wait(1'b0),
    .pcpi_ready(1'b0)
  );

  // ============================================================================
  // Address Decoder
  // ============================================================================
  
  wire compact_sel    = (mem_addr >= COMPACT_BASE) && 
                        (mem_addr < COMPACT_BASE + COMPACT_SIZE);
  wire bitnet_sel     = (mem_addr >= BITNET_BASE) && 
                        (mem_addr < BITNET_BASE + BITNET_SIZE);
  wire flash_ctrl_sel = (mem_addr >= FLASH_CTRL_BASE) && 
                        (mem_addr < FLASH_CTRL_BASE + FLASH_CTRL_SIZE);
  wire psram_ctrl_sel = (mem_addr >= PSRAM_CTRL_BASE) && 
                        (mem_addr < PSRAM_CTRL_BASE + PSRAM_CTRL_SIZE);
  wire flash_mem_sel  = (mem_addr >= FLASH_MEM_BASE) && 
                        (mem_addr < FLASH_MEM_BASE + FLASH_MEM_SIZE);
  wire psram_mem_sel  = (mem_addr >= PSRAM_MEM_BASE) && 
                        (mem_addr < PSRAM_MEM_BASE + PSRAM_MEM_SIZE);
  
  wire local_sel = compact_sel | bitnet_sel | flash_ctrl_sel | psram_ctrl_sel | 
                   flash_mem_sel | psram_mem_sel;
  wire bus_sel   = ~local_sel & ~mem_instr;  // Non-local data access goes to bus

  // ============================================================================
  // AI Accelerators + Storage Controllers
  // ============================================================================
  
  // CompactAccel signals
  wire [31:0] compact_addr;
  wire [31:0] compact_wdata;
  wire [31:0] compact_rdata;
  wire        compact_wen;
  wire        compact_ren;
  wire        compact_valid;
  wire        compact_irq;
  
  // BitNetAccel signals
  wire [31:0] bitnet_addr;
  wire [31:0] bitnet_wdata;
  wire [31:0] bitnet_rdata;
  wire        bitnet_wen;
  wire        bitnet_ren;
  wire        bitnet_valid;
  wire        bitnet_irq;
  
  // Flash Controller signals
  wire [31:0] flash_ctrl_addr;
  wire [31:0] flash_ctrl_wdata;
  wire [31:0] flash_ctrl_rdata;
  wire        flash_ctrl_wen;
  wire        flash_ctrl_ren;
  wire        flash_ctrl_valid;
  
  // PSRAM Controller signals
  wire [31:0] psram_ctrl_addr;
  wire [31:0] psram_ctrl_wdata;
  wire [31:0] psram_ctrl_rdata;
  wire        psram_ctrl_wen;
  wire        psram_ctrl_ren;
  
  // Flash Memory signals
  wire [31:0] flash_mem_addr;
  wire [31:0] flash_mem_wdata;
  wire [31:0] flash_mem_rdata;
  wire        flash_mem_wen;
  wire        flash_mem_ren;
  wire        flash_mem_valid;
  wire        flash_mem_ready;
  
  // PSRAM Memory signals
  wire [31:0] psram_mem_addr;
  wire [31:0] psram_mem_wdata;
  wire [31:0] psram_mem_rdata;
  wire        psram_mem_wen;
  wire        psram_mem_ren;
  wire        psram_mem_valid;
  wire        psram_mem_ready;
  
  // Connect AI accelerators
  assign compact_addr  = mem_addr - COMPACT_BASE;
  assign compact_wdata = mem_wdata;
  assign compact_wen   = compact_sel & mem_valid & (|mem_wstrb);
  assign compact_ren   = compact_sel & mem_valid & (~|mem_wstrb);
  assign compact_valid = compact_sel & mem_valid;
  
  assign bitnet_addr   = mem_addr - BITNET_BASE;
  assign bitnet_wdata  = mem_wdata;
  assign bitnet_wen    = bitnet_sel & mem_valid & (|mem_wstrb);
  assign bitnet_ren    = bitnet_sel & mem_valid & (~|mem_wstrb);
  assign bitnet_valid  = bitnet_sel & mem_valid;
  
  // Connect Flash controller
  assign flash_ctrl_addr  = mem_addr - FLASH_CTRL_BASE;
  assign flash_ctrl_wdata = mem_wdata;
  assign flash_ctrl_wen   = flash_ctrl_sel & mem_valid & (|mem_wstrb);
  assign flash_ctrl_ren   = flash_ctrl_sel & mem_valid & (~|mem_wstrb);
  assign flash_ctrl_valid = flash_ctrl_sel & mem_valid;
  
  // Connect PSRAM controller
  assign psram_ctrl_addr  = mem_addr - PSRAM_CTRL_BASE;
  assign psram_ctrl_wdata = mem_wdata;
  assign psram_ctrl_wen   = psram_ctrl_sel & mem_valid & (|mem_wstrb);
  assign psram_ctrl_ren   = psram_ctrl_sel & mem_valid & (~|mem_wstrb);
  assign psram_ctrl_valid = psram_ctrl_sel & mem_valid;
  
  // Connect Flash memory
  assign flash_mem_addr  = mem_addr;
  assign flash_mem_wdata = mem_wdata;
  assign flash_mem_wen   = flash_mem_sel & mem_valid & (|mem_wstrb);
  assign flash_mem_ren   = flash_mem_sel & mem_valid & (~|mem_wstrb);
  assign flash_mem_valid = flash_mem_sel & mem_valid;
  
  // Connect PSRAM memory
  assign psram_mem_addr  = mem_addr;
  assign psram_mem_wdata = mem_wdata;
  assign psram_mem_wen   = psram_mem_sel & mem_valid & (|mem_wstrb);
  assign psram_mem_ren   = psram_mem_sel & mem_valid & (~|mem_wstrb);
  assign psram_mem_valid = psram_mem_sel & mem_valid;
  
  // Instantiate modules (using generated Verilog)
  ip1_SimpleCompactAccel compact_accel (
    .clock(clock),
    .reset(reset),
    .io_reg_addr(compact_addr),
    .io_reg_wdata(compact_wdata),
    .io_reg_rdata(compact_rdata),
    .io_reg_wen(compact_wen),
    .io_reg_ren(compact_ren),
    .io_reg_valid(compact_valid),
    .io_irq(compact_irq)
  );
  
  ip1_SimpleBitNetAccel bitnet_accel (
    .clock(clock),
    .reset(reset),
    .io_reg_addr(bitnet_addr),
    .io_reg_wdata(bitnet_wdata),
    .io_reg_rdata(bitnet_rdata),
    .io_reg_wen(bitnet_wen),
    .io_reg_ren(bitnet_ren),
    .io_reg_valid(bitnet_valid),
    .io_irq(bitnet_irq)
  );
  
  ip1_SPIFlash flash_controller (
    .clock(clock),
    .reset(reset),
    .io_addr(flash_ctrl_addr),
    .io_wdata(flash_ctrl_wdata),
    .io_rdata(flash_ctrl_rdata),
    .io_wen(flash_ctrl_wen),
    .io_ren(flash_ctrl_ren),
    .io_valid(flash_ctrl_valid),
    .io_spi_clk(),
    .io_spi_mosi(),
    .io_spi_miso(1'b0),
    .io_spi_cs()
  );
  
  ip1_PSRAM psram_controller (
    .clock(clock),
    .reset(reset),
    .io_reg_addr(psram_ctrl_addr),
    .io_reg_wdata(psram_ctrl_wdata),
    .io_reg_rdata(psram_ctrl_rdata),
    .io_reg_wen(psram_ctrl_wen),
    .io_reg_ren(psram_ctrl_ren),
    .io_spi_clk(),
    .io_spi_cs(),
    .io_spi_mosi(),
    .io_spi_miso(1'b0),
    .io_spi_sio2_out(),
    .io_spi_sio2_oe(),
    .io_spi_sio2_in(1'b0),
    .io_spi_sio3_out(),
    .io_spi_sio3_oe(),
    .io_spi_sio3_in(1'b0)
  );
  
  // Flash and PSRAM memory access - always ready (combinational)
  assign flash_mem_rdata = 32'h00000013;  // NOP
  assign psram_mem_rdata = 32'h0;

  // ============================================================================
  // SimpleBus Interface Adaptation
  // ============================================================================
  
  // IFU (Instruction Fetch) - always goes to bus
  assign io_ifu_reqValid = mem_valid & mem_instr;
  assign io_ifu_addr = mem_addr;
  
  // LSU (Load/Store) - only non-accelerator accesses go to bus
  assign io_lsu_reqValid = mem_valid & bus_sel;
  assign io_lsu_addr = mem_addr;
  assign io_lsu_wen = |mem_wstrb;
  assign io_lsu_wdata = mem_wdata;
  assign io_lsu_wmask = mem_wstrb;
  
  // Calculate size from wmask
  assign io_lsu_size = (mem_wstrb == 4'b1111) ? 2'b10 :
                       ((mem_wstrb == 4'b0011) || (mem_wstrb == 4'b1100)) ? 2'b01 :
                       2'b00;
  
  // ============================================================================
  // Response Multiplexing
  // ============================================================================
  
  wire ifu_resp        = io_ifu_respValid & mem_instr;
  wire lsu_resp        = io_lsu_respValid & bus_sel;
  // Local modules respond immediately (combinational)
  wire compact_resp    = compact_sel & mem_valid;
  wire bitnet_resp     = bitnet_sel & mem_valid;
  wire flash_ctrl_resp = flash_ctrl_sel & mem_valid;
  wire psram_ctrl_resp = psram_ctrl_sel & mem_valid;
  wire flash_mem_resp  = flash_mem_sel & mem_valid;
  wire psram_mem_resp  = psram_mem_sel & mem_valid;
  
  assign mem_ready = ifu_resp | lsu_resp | compact_resp | bitnet_resp | 
                     flash_ctrl_resp | psram_ctrl_resp | flash_mem_resp | psram_mem_resp;
  
  // Data multiplexing
  reg [31:0] local_rdata;
  always @(*) begin
    if (compact_sel)
      local_rdata = compact_rdata;
    else if (bitnet_sel)
      local_rdata = bitnet_rdata;
    else if (flash_ctrl_sel)
      local_rdata = flash_ctrl_rdata;
    else if (psram_ctrl_sel)
      local_rdata = psram_ctrl_rdata;
    else if (flash_mem_sel)
      local_rdata = flash_mem_rdata;
    else if (psram_mem_sel)
      local_rdata = psram_mem_rdata;
    else
      local_rdata = 32'h0;
  end
  
  assign mem_rdata = mem_instr ? io_ifu_rdata : 
                     local_sel ? local_rdata : 
                     io_lsu_rdata;

  // ============================================================================
  // Interrupt Routing
  // ============================================================================
  
  assign cpu_irq = {
    12'h0,           // [31:20] Reserved
    2'h0,            // [19:18] Reserved
    bitnet_irq,      // [17] BitNet accelerator
    compact_irq,     // [16] Compact accelerator
    16'h0            // [15:0] Reserved
  };

endmodule
