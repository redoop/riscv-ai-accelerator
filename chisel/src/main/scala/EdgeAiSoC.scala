// EdgeAiSoC.scala - RISC-V + AI Accelerator Integration
// Complete SoC with RISC-V core, dual AI accelerators, and peripherals

package riscv.ai

import chisel3._
import chisel3.util._

// ============================================================================
// AXI4-Lite Interface Definitions
// ============================================================================

class AXI4LiteWriteAddress extends Bundle {
  val addr = UInt(32.W)
  val valid = Bool()
  val ready = Bool()
}

class AXI4LiteWriteData extends Bundle {
  val data = UInt(32.W)
  val strb = UInt(4.W)
  val valid = Bool()
  val ready = Bool()
}

class AXI4LiteWriteResponse extends Bundle {
  val resp = UInt(2.W)
  val valid = Bool()
  val ready = Bool()
}

class AXI4LiteReadAddress extends Bundle {
  val addr = UInt(32.W)
  val valid = Bool()
  val ready = Bool()
}

class AXI4LiteReadData extends Bundle {
  val data = UInt(32.W)
  val resp = UInt(2.W)
  val valid = Bool()
  val ready = Bool()
}

class AXI4LiteIO extends Bundle {
  val aw = new AXI4LiteWriteAddress()
  val w = new AXI4LiteWriteData()
  val b = new AXI4LiteWriteResponse()
  val ar = new AXI4LiteReadAddress()
  val r = new AXI4LiteReadData()
}

// ============================================================================
// Memory Map Configuration
// ============================================================================

object MemoryMap {
  // RAM: 0x0000_0000 - 0x0FFF_FFFF (256 MB)
  val RAM_BASE = 0x00000000L
  val RAM_SIZE = 0x10000000L
  
  // CompactScale Accelerator: 0x1000_0000 - 0x1000_0FFF (4 KB)
  val COMPACT_BASE = 0x10000000L
  val COMPACT_SIZE = 0x00001000L
  
  // BitNetScale Accelerator: 0x1000_1000 - 0x1000_1FFF (4 KB)
  val BITNET_BASE = 0x10001000L
  val BITNET_SIZE = 0x00001000L
  
  // DMA Controller: 0x1000_2000 - 0x1000_2FFF (4 KB)
  val DMA_BASE = 0x10002000L
  val DMA_SIZE = 0x00001000L
  
  // Interrupt Controller: 0x1000_3000 - 0x1000_3FFF (4 KB)
  val INTC_BASE = 0x10003000L
  val INTC_SIZE = 0x00001000L
  
  // UART: 0x2000_0000 - 0x2000_FFFF (64 KB)
  val UART_BASE = 0x20000000L
  val UART_SIZE = 0x00010000L
  
  // SPI: 0x2001_0000 - 0x2001_FFFF (64 KB)
  val SPI_BASE = 0x20010000L
  val SPI_SIZE = 0x00010000L
  
  // GPIO: 0x2002_0000 - 0x2002_FFFF (64 KB)
  val GPIO_BASE = 0x20020000L
  val GPIO_SIZE = 0x00010000L
  
  // Flash ROM: 0x8000_0000 - 0x8FFF_FFFF (256 MB)
  val FLASH_BASE = 0x80000000L
  val FLASH_SIZE = 0x10000000L
}

// ============================================================================
// Register Definitions for AI Accelerators
// ============================================================================

object AccelRegs {
  // Control and Status Registers
  val CTRL = 0x000
  val STATUS = 0x004
  val INT_EN = 0x008
  val INT_STATUS = 0x00C
  
  // DMA Registers
  val DMA_SRC = 0x010
  val DMA_DST = 0x014
  val DMA_LEN = 0x018
  
  // Configuration Registers
  val MATRIX_SIZE = 0x01C
  val CONFIG = 0x020
  val SPARSITY_EN = 0x024
  
  // Performance Counters
  val PERF_CYCLES = 0x028
  val PERF_OPS = 0x02C
  
  // Data Buffers
  val MATRIX_A_BASE = 0x100
  val MATRIX_B_BASE = 0x300
  val MATRIX_C_BASE = 0x500
  
  // Control Bits
  val CTRL_START = 0x1
  val CTRL_RESET = 0x2
  val CTRL_DMA_EN = 0x4
  
  // Status Bits
  val STATUS_BUSY = 0x1
  val STATUS_DONE = 0x2
  val STATUS_ERROR = 0x4
}

// ============================================================================
// DMA Controller
// ============================================================================

class EdgeDMAController extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val axi_master = new AXI4LiteIO()
    val irq = Output(Bool())
  })
  
  // DMA Registers
  val srcAddr = RegInit(0.U(32.W))
  val dstAddr = RegInit(0.U(32.W))
  val length = RegInit(0.U(32.W))
  val ctrl = RegInit(0.U(32.W))
  val status = RegInit(0.U(32.W))
  val transferCount = RegInit(0.U(32.W))
  
  // DMA State Machine
  val sIdle :: sReadReq :: sReadData :: sWriteReq :: sWriteData :: sWriteResp :: sDone :: Nil = Enum(7)
  val state = RegInit(sIdle)
  
  val dataBuffer = RegInit(0.U(32.W))
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  io.axi_master.aw.valid := false.B
  io.axi_master.aw.addr := 0.U
  io.axi_master.w.valid := false.B
  io.axi_master.w.data := 0.U
  io.axi_master.w.strb := 0xF.U
  io.axi_master.b.ready := false.B
  io.axi_master.ar.valid := false.B
  io.axi_master.ar.addr := 0.U
  io.axi_master.r.ready := false.B
  
  io.irq := false.B
  
  // DMA State Machine
  switch(state) {
    is(sIdle) {
      status := 0.U
      when(ctrl(0)) {  // Start bit
        state := sReadReq
        transferCount := 0.U
      }
    }
    
    is(sReadReq) {
      io.axi_master.ar.valid := true.B
      io.axi_master.ar.addr := srcAddr + (transferCount << 2)
      when(io.axi_master.ar.ready) {
        state := sReadData
      }
    }
    
    is(sReadData) {
      io.axi_master.r.ready := true.B
      when(io.axi_master.r.valid) {
        dataBuffer := io.axi_master.r.data
        state := sWriteReq
      }
    }
    
    is(sWriteReq) {
      io.axi_master.aw.valid := true.B
      io.axi_master.aw.addr := dstAddr + (transferCount << 2)
      when(io.axi_master.aw.ready) {
        state := sWriteData
      }
    }
    
    is(sWriteData) {
      io.axi_master.w.valid := true.B
      io.axi_master.w.data := dataBuffer
      when(io.axi_master.w.ready) {
        state := sWriteResp
      }
    }
    
    is(sWriteResp) {
      io.axi_master.b.ready := true.B
      when(io.axi_master.b.valid) {
        transferCount := transferCount + 1.U
        when(transferCount >= (length >> 2)) {
          state := sDone
        }.otherwise {
          state := sReadReq
        }
      }
    }
    
    is(sDone) {
      status := 2.U  // Done bit
      io.irq := true.B
      ctrl := 0.U
      state := sIdle
    }
  }
  
  // Register interface (simplified)
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    
    switch(io.axi_slave.aw.addr(7, 0)) {
      is(AccelRegs.DMA_SRC.U) { srcAddr := io.axi_slave.w.data }
      is(AccelRegs.DMA_DST.U) { dstAddr := io.axi_slave.w.data }
      is(AccelRegs.DMA_LEN.U) { length := io.axi_slave.w.data }
      is(AccelRegs.CTRL.U) { ctrl := io.axi_slave.w.data }
    }
  }
  
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    
    switch(io.axi_slave.ar.addr(7, 0)) {
      is(AccelRegs.STATUS.U) { io.axi_slave.r.data := status }
      is(AccelRegs.DMA_SRC.U) { io.axi_slave.r.data := srcAddr }
      is(AccelRegs.DMA_DST.U) { io.axi_slave.r.data := dstAddr }
      is(AccelRegs.DMA_LEN.U) { io.axi_slave.r.data := length }
    }
  }
}

// ============================================================================
// Interrupt Controller
// ============================================================================

class EdgeInterruptController extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val irq = Input(Vec(32, Bool()))
    val cpu_irq = Output(UInt(32.W))
  })
  
  val irqEnable = RegInit(0.U(32.W))
  val irqPending = RegInit(0.U(32.W))
  
  // Capture interrupt signals
  for (i <- 0 until 32) {
    when(io.irq(i)) {
      irqPending := irqPending | (1.U << i)
    }
  }
  
  io.cpu_irq := irqPending & irqEnable
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  // Register interface
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    
    switch(io.axi_slave.aw.addr(7, 0)) {
      is(0x00.U) { irqEnable := io.axi_slave.w.data }
      is(0x04.U) { irqPending := irqPending & ~io.axi_slave.w.data } // Clear on write
    }
  }
  
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    
    switch(io.axi_slave.ar.addr(7, 0)) {
      is(0x00.U) { io.axi_slave.r.data := irqEnable }
      is(0x04.U) { io.axi_slave.r.data := irqPending }
    }
  }
}

// ============================================================================
// Simple Peripherals
// ============================================================================

class UARTController extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val tx = Output(Bool())
    val rx = Input(Bool())
  })
  
  val txData = RegInit(0.U(8.W))
  val txValid = RegInit(false.B)
  val rxData = RegInit(0.U(8.W))
  val rxValid = RegInit(false.B)
  
  io.tx := true.B // Idle high
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  // Simplified register interface
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    txData := io.axi_slave.w.data(7, 0)
    txValid := true.B
  }
  
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    io.axi_slave.r.data := Cat(0.U(24.W), rxData)
  }
}

class GPIOController extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val gpio_out = Output(UInt(32.W))
    val gpio_in = Input(UInt(32.W))
  })
  
  val gpioOut = RegInit(0.U(32.W))
  io.gpio_out := gpioOut
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    gpioOut := io.axi_slave.w.data
  }
  
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    io.axi_slave.r.data := io.gpio_in
  }
}

// ============================================================================
// AI Accelerator Wrapper with AXI4-Lite Interface
// ============================================================================

class CompactScaleWrapper extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val irq = Output(Bool())
  })
  
  // Control and status registers
  val ctrl = RegInit(0.U(32.W))
  val status = RegInit(0.U(32.W))
  val intEnable = RegInit(0.U(32.W))
  val matrixSize = RegInit(0.U(32.W))
  val perfCycles = RegInit(0.U(32.W))
  val perfOps = RegInit(0.U(32.W))
  
  // Matrix buffers (simplified - 8x8 matrices)
  val matrixA = Mem(64, UInt(32.W))
  val matrixB = Mem(64, UInt(32.W))
  val matrixC = Mem(64, UInt(32.W))
  
  // State machine
  val sIdle :: sCompute :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val computeCounter = RegInit(0.U(8.W))
  
  io.irq := false.B
  status := 0.U
  
  // Computation state machine
  switch(state) {
    is(sIdle) {
      when(ctrl(0)) { // Start bit
        state := sCompute
        computeCounter := 0.U
        perfCycles := 0.U
      }
    }
    is(sCompute) {
      status := 1.U // Busy
      perfCycles := perfCycles + 1.U
      computeCounter := computeCounter + 1.U
      
      // Simplified computation (just increment counter)
      when(computeCounter >= 64.U) {
        state := sDone
        perfOps := 64.U * 64.U * 8.U // Approximate ops
      }
    }
    is(sDone) {
      status := 2.U // Done
      io.irq := intEnable(0)
      ctrl := 0.U
      state := sIdle
    }
  }
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  // AXI4-Lite write
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    
    val addr = io.axi_slave.aw.addr(11, 0)
    switch(addr) {
      is(AccelRegs.CTRL.U) { ctrl := io.axi_slave.w.data }
      is(AccelRegs.INT_EN.U) { intEnable := io.axi_slave.w.data }
      is(AccelRegs.MATRIX_SIZE.U) { matrixSize := io.axi_slave.w.data }
    }
    
    // Matrix A write (0x100-0x1FF)
    when(addr >= AccelRegs.MATRIX_A_BASE.U && addr < (AccelRegs.MATRIX_A_BASE + 256).U) {
      val idx = (addr - AccelRegs.MATRIX_A_BASE.U) >> 2
      matrixA(idx) := io.axi_slave.w.data
    }
    
    // Matrix B write (0x300-0x3FF)
    when(addr >= AccelRegs.MATRIX_B_BASE.U && addr < (AccelRegs.MATRIX_B_BASE + 256).U) {
      val idx = (addr - AccelRegs.MATRIX_B_BASE.U) >> 2
      matrixB(idx) := io.axi_slave.w.data
    }
  }
  
  // AXI4-Lite read
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    
    val addr = io.axi_slave.ar.addr(11, 0)
    switch(addr) {
      is(AccelRegs.CTRL.U) { io.axi_slave.r.data := ctrl }
      is(AccelRegs.STATUS.U) { io.axi_slave.r.data := status }
      is(AccelRegs.INT_EN.U) { io.axi_slave.r.data := intEnable }
      is(AccelRegs.MATRIX_SIZE.U) { io.axi_slave.r.data := matrixSize }
      is(AccelRegs.PERF_CYCLES.U) { io.axi_slave.r.data := perfCycles }
      is(AccelRegs.PERF_OPS.U) { io.axi_slave.r.data := perfOps }
    }
    
    // Matrix C read (0x500-0x5FF)
    when(addr >= AccelRegs.MATRIX_C_BASE.U && addr < (AccelRegs.MATRIX_C_BASE + 256).U) {
      val idx = (addr - AccelRegs.MATRIX_C_BASE.U) >> 2
      io.axi_slave.r.data := matrixC(idx)
    }
  }
}

class BitNetScaleWrapper extends Module {
  val io = IO(new Bundle {
    val axi_slave = Flipped(new AXI4LiteIO())
    val irq = Output(Bool())
  })
  
  // Control and status registers
  val ctrl = RegInit(0.U(32.W))
  val status = RegInit(0.U(32.W))
  val intEnable = RegInit(0.U(32.W))
  val config = RegInit(0.U(32.W))
  val sparsityEn = RegInit(false.B)
  val matrixSize = RegInit(0.U(32.W))
  val perfCycles = RegInit(0.U(32.W))
  val perfOps = RegInit(0.U(32.W))
  
  // Matrix buffers (16x16 for BitNet)
  val activation = Mem(256, UInt(32.W))
  val weight = Mem(256, UInt(32.W))
  val result = Mem(256, UInt(32.W))
  
  // State machine
  val sIdle :: sCompute :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val computeCounter = RegInit(0.U(16.W))
  
  io.irq := false.B
  status := 0.U
  
  // Computation state machine
  switch(state) {
    is(sIdle) {
      when(ctrl(0)) { // Start bit
        state := sCompute
        computeCounter := 0.U
        perfCycles := 0.U
      }
    }
    is(sCompute) {
      status := 1.U // Busy
      perfCycles := perfCycles + 1.U
      computeCounter := computeCounter + 1.U
      
      // Simplified BitNet computation
      when(computeCounter >= 256.U) {
        state := sDone
        perfOps := 256.U * 256.U * 16.U // Approximate ops
      }
    }
    is(sDone) {
      status := 2.U // Done
      io.irq := intEnable(0)
      ctrl := 0.U
      state := sIdle
    }
  }
  
  // Default outputs
  io.axi_slave.aw.ready := false.B
  io.axi_slave.w.ready := false.B
  io.axi_slave.b.valid := false.B
  io.axi_slave.b.resp := 0.U
  io.axi_slave.ar.ready := false.B
  io.axi_slave.r.valid := false.B
  io.axi_slave.r.data := 0.U
  io.axi_slave.r.resp := 0.U
  
  // AXI4-Lite write
  when(io.axi_slave.aw.valid && io.axi_slave.w.valid) {
    io.axi_slave.aw.ready := true.B
    io.axi_slave.w.ready := true.B
    io.axi_slave.b.valid := true.B
    
    val addr = io.axi_slave.aw.addr(11, 0)
    switch(addr) {
      is(AccelRegs.CTRL.U) { ctrl := io.axi_slave.w.data }
      is(AccelRegs.INT_EN.U) { intEnable := io.axi_slave.w.data }
      is(AccelRegs.CONFIG.U) { config := io.axi_slave.w.data }
      is(AccelRegs.SPARSITY_EN.U) { sparsityEn := io.axi_slave.w.data(0) }
      is(AccelRegs.MATRIX_SIZE.U) { matrixSize := io.axi_slave.w.data }
    }
    
    // Activation write (0x100-0x4FF)
    when(addr >= AccelRegs.MATRIX_A_BASE.U && addr < AccelRegs.MATRIX_B_BASE.U) {
      val idx = (addr - AccelRegs.MATRIX_A_BASE.U) >> 2
      activation(idx) := io.axi_slave.w.data
    }
    
    // Weight write (0x300-0x4FF)
    when(addr >= AccelRegs.MATRIX_B_BASE.U && addr < AccelRegs.MATRIX_C_BASE.U) {
      val idx = (addr - AccelRegs.MATRIX_B_BASE.U) >> 2
      weight(idx) := io.axi_slave.w.data
    }
  }
  
  // AXI4-Lite read
  when(io.axi_slave.ar.valid) {
    io.axi_slave.ar.ready := true.B
    io.axi_slave.r.valid := true.B
    
    val addr = io.axi_slave.ar.addr(11, 0)
    switch(addr) {
      is(AccelRegs.CTRL.U) { io.axi_slave.r.data := ctrl }
      is(AccelRegs.STATUS.U) { io.axi_slave.r.data := status }
      is(AccelRegs.INT_EN.U) { io.axi_slave.r.data := intEnable }
      is(AccelRegs.CONFIG.U) { io.axi_slave.r.data := config }
      is(AccelRegs.SPARSITY_EN.U) { io.axi_slave.r.data := sparsityEn.asUInt }
      is(AccelRegs.MATRIX_SIZE.U) { io.axi_slave.r.data := matrixSize }
      is(AccelRegs.PERF_CYCLES.U) { io.axi_slave.r.data := perfCycles }
      is(AccelRegs.PERF_OPS.U) { io.axi_slave.r.data := perfOps }
    }
    
    // Result read (0x500-0x8FF)
    when(addr >= AccelRegs.MATRIX_C_BASE.U && addr < (AccelRegs.MATRIX_C_BASE + 1024).U) {
      val idx = (addr - AccelRegs.MATRIX_C_BASE.U) >> 2
      io.axi_slave.r.data := result(idx)
    }
  }
}

// ============================================================================
// Simple Address Decoder - Note: This is a simplified placeholder
// In a real design, you would need proper AXI crossbar logic
// ============================================================================

// For now, we'll skip the address decoder and directly connect in EdgeAiSoC

// ============================================================================
// PicoRV32 Wrapper (BlackBox for Verilog integration)
// ============================================================================

class PicoRV32 extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val resetn = Input(Bool())
    val trap = Output(Bool())
    
    // Memory interface
    val mem_valid = Output(Bool())
    val mem_instr = Output(Bool())
    val mem_ready = Input(Bool())
    val mem_addr = Output(UInt(32.W))
    val mem_wdata = Output(UInt(32.W))
    val mem_wstrb = Output(UInt(4.W))
    val mem_rdata = Input(UInt(32.W))
    
    // IRQ interface
    val irq = Input(UInt(32.W))
    val eoi = Output(UInt(32.W))
  })
  
  addResource("/rtl/picorv32.v")
}

// ============================================================================
// Memory Interface Adapter (Native PicoRV32 to AXI4-Lite)
// ============================================================================

class MemoryInterfaceAdapter extends Module {
  val io = IO(new Bundle {
    // PicoRV32 native interface
    val mem_valid = Input(Bool())
    val mem_instr = Input(Bool())
    val mem_ready = Output(Bool())
    val mem_addr = Input(UInt(32.W))
    val mem_wdata = Input(UInt(32.W))
    val mem_wstrb = Input(UInt(4.W))
    val mem_rdata = Output(UInt(32.W))
    
    // AXI4-Lite master interface
    val axi = new AXI4LiteIO()
  })
  
  val sIdle :: sWrite :: sRead :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)
  
  val isWrite = io.mem_wstrb.orR
  
  io.mem_ready := false.B
  io.mem_rdata := 0.U
  
  io.axi.aw.valid := false.B
  io.axi.aw.addr := 0.U
  io.axi.w.valid := false.B
  io.axi.w.data := 0.U
  io.axi.w.strb := 0.U
  io.axi.b.ready := false.B
  io.axi.ar.valid := false.B
  io.axi.ar.addr := 0.U
  io.axi.r.ready := false.B
  
  switch(state) {
    is(sIdle) {
      when(io.mem_valid) {
        when(isWrite) {
          state := sWrite
          io.axi.aw.valid := true.B
          io.axi.aw.addr := io.mem_addr
          io.axi.w.valid := true.B
          io.axi.w.data := io.mem_wdata
          io.axi.w.strb := io.mem_wstrb
        }.otherwise {
          state := sRead
          io.axi.ar.valid := true.B
          io.axi.ar.addr := io.mem_addr
        }
      }
    }
    
    is(sWrite) {
      io.axi.aw.valid := true.B
      io.axi.aw.addr := io.mem_addr
      io.axi.w.valid := true.B
      io.axi.w.data := io.mem_wdata
      io.axi.w.strb := io.mem_wstrb
      io.axi.b.ready := true.B
      
      when(io.axi.b.valid) {
        io.mem_ready := true.B
        state := sDone
      }
    }
    
    is(sRead) {
      io.axi.ar.valid := true.B
      io.axi.ar.addr := io.mem_addr
      io.axi.r.ready := true.B
      
      when(io.axi.r.valid) {
        io.mem_rdata := io.axi.r.data
        io.mem_ready := true.B
        state := sDone
      }
    }
    
    is(sDone) {
      io.mem_ready := true.B
      when(!io.mem_valid) {
        state := sIdle
      }
    }
  }
}

// ============================================================================
// EdgeAiSoC - Top Level Module
// ============================================================================

class EdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // External interfaces
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    val gpio_out = Output(UInt(32.W))
    val gpio_in = Input(UInt(32.W))
    
    // Debug/Status
    val trap = Output(Bool())
    val compact_irq = Output(Bool())
    val bitnet_irq = Output(Bool())
  })
  
  // ========== Instantiate Components ==========
  
  // RISC-V Core (PicoRV32)
  val riscv = Module(new PicoRV32())
  riscv.io.clk := clock
  riscv.io.resetn := !reset.asBool
  
  // Memory Interface Adapter
  val memAdapter = Module(new MemoryInterfaceAdapter())
  memAdapter.io.mem_valid := riscv.io.mem_valid
  memAdapter.io.mem_instr := riscv.io.mem_instr
  riscv.io.mem_ready := memAdapter.io.mem_ready
  memAdapter.io.mem_addr := riscv.io.mem_addr
  memAdapter.io.mem_wdata := riscv.io.mem_wdata
  memAdapter.io.mem_wstrb := riscv.io.mem_wstrb
  riscv.io.mem_rdata := memAdapter.io.mem_rdata
  
  // AI Accelerators (simplified - direct connection for now)
  val compactScale = Module(new CompactScaleWrapper())
  compactScale.io.axi_slave <> memAdapter.io.axi
  
  val bitnetScale = Module(new BitNetScaleWrapper())
  // For now, tie off bitnetScale inputs
  bitnetScale.io.axi_slave.aw.valid := false.B
  bitnetScale.io.axi_slave.aw.addr := 0.U
  bitnetScale.io.axi_slave.w.valid := false.B
  bitnetScale.io.axi_slave.w.data := 0.U
  bitnetScale.io.axi_slave.w.strb := 0.U
  bitnetScale.io.axi_slave.b.ready := false.B
  bitnetScale.io.axi_slave.ar.valid := false.B
  bitnetScale.io.axi_slave.ar.addr := 0.U
  bitnetScale.io.axi_slave.r.ready := false.B
  
  // Peripherals (simplified)
  io.uart_tx := true.B // Idle high
  io.gpio_out := 0.U
  
  // ========== Interrupt Wiring ==========
  
  // Simplified interrupt handling
  riscv.io.irq := Cat(Fill(14, false.B), bitnetScale.io.irq, compactScale.io.irq, Fill(16, false.B))
  
  // ========== Debug Outputs ==========
  
  io.trap := riscv.io.trap
  io.compact_irq := compactScale.io.irq
  io.bitnet_irq := bitnetScale.io.irq
}
