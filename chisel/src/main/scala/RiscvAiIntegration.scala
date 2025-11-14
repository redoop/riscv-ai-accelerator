package riscv.ai

import chisel3._
import chisel3.util._

/**
 * PicoRV32 BlackBox Wrapper
 * 将 Verilog 实现的 PicoRV32 处理器封装为 Chisel BlackBox
 * 
 * 注意: 使用 override def desiredName 来指定实际的 Verilog 模块名
 */
class PicoRV32BlackBox extends BlackBox with HasBlackBoxResource {
  // 指定实际的 Verilog 模块名为 "picorv32"
  override def desiredName = "picorv32"
  
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val resetn = Input(Bool())
    val trap = Output(Bool())
    
    // 内存接口
    val mem_valid = Output(Bool())
    val mem_instr = Output(Bool())
    val mem_ready = Input(Bool())
    val mem_addr = Output(UInt(32.W))
    val mem_wdata = Output(UInt(32.W))
    val mem_wstrb = Output(UInt(4.W))
    val mem_rdata = Input(UInt(32.W))
    
    // Look-Ahead 接口
    val mem_la_read = Output(Bool())
    val mem_la_write = Output(Bool())
    val mem_la_addr = Output(UInt(32.W))
    val mem_la_wdata = Output(UInt(32.W))
    val mem_la_wstrb = Output(UInt(4.W))
    
    // PCPI 接口 (用于连接 AI 加速器)
    val pcpi_valid = Output(Bool())
    val pcpi_insn = Output(UInt(32.W))
    val pcpi_rs1 = Output(UInt(32.W))
    val pcpi_rs2 = Output(UInt(32.W))
    val pcpi_wr = Input(Bool())
    val pcpi_rd = Input(UInt(32.W))
    val pcpi_wait = Input(Bool())
    val pcpi_ready = Input(Bool())
    
    // IRQ 接口
    val irq = Input(UInt(32.W))
    val eoi = Output(UInt(32.W))
    
    // Trace 接口
    val trace_valid = Output(Bool())
    val trace_data = Output(UInt(36.W))
  })
  
  // 添加 Verilog 源文件
  addResource("/rtl/picorv32.v")
}

/**
 * RISC-V AI 加速器系统集成
 * 将 PicoRV32 处理器与 AI 加速器通过 PCPI 接口连接
 */
class RiscvAiSystem(
  dataWidth: Int = 32,
  matrixSize: Int = 8,
  numMacUnits: Int = 16,
  memoryDepth: Int = 512
) extends Module {
  
  val io = IO(new Bundle {
    // 外部内存接口 (连接到 PicoRV32)
    val mem_valid = Output(Bool())
    val mem_instr = Output(Bool())
    val mem_ready = Input(Bool())
    val mem_addr = Output(UInt(32.W))
    val mem_wdata = Output(UInt(32.W))
    val mem_wstrb = Output(UInt(4.W))
    val mem_rdata = Input(UInt(32.W))
    
    // IRQ 接口
    val irq = Input(UInt(32.W))
    val eoi = Output(UInt(32.W))
    
    // 系统状态
    val trap = Output(Bool())
    val ai_busy = Output(Bool())
    val ai_done = Output(Bool())
    
    // 性能计数器
    val perf_counters = Output(Vec(4, UInt(32.W)))
    
    // Trace 接口
    val trace_valid = Output(Bool())
    val trace_data = Output(UInt(36.W))
  })
  
  // 实例化 PicoRV32 处理器
  val cpu = Module(new PicoRV32BlackBox)
  
  // 实例化 AI 加速器
  val aiAccel = Module(new CompactScaleAiChip(
    dataWidth = dataWidth,
    matrixSize = matrixSize,
    numMacUnits = numMacUnits,
    memoryDepth = memoryDepth,
    addrWidth = 10  // AI 加速器内部地址宽度
  ))
  
  // 连接 CPU 时钟和复位
  cpu.io.clk := clock
  cpu.io.resetn := !reset.asBool
  
  // 连接 CPU 内存接口到外部
  io.mem_valid := cpu.io.mem_valid
  io.mem_instr := cpu.io.mem_instr
  cpu.io.mem_ready := io.mem_ready
  io.mem_addr := cpu.io.mem_addr
  io.mem_wdata := cpu.io.mem_wdata
  io.mem_wstrb := cpu.io.mem_wstrb
  cpu.io.mem_rdata := io.mem_rdata
  
  // 连接 IRQ
  cpu.io.irq := io.irq
  io.eoi := cpu.io.eoi
  
  // 连接 Trap 和 Trace
  io.trap := cpu.io.trap
  io.trace_valid := cpu.io.trace_valid
  io.trace_data := cpu.io.trace_data
  
  // ========================================
  // PCPI 接口: 连接 CPU 和 AI 加速器
  // ========================================
  
  // PCPI 地址解码
  // 当 CPU 访问特定地址范围时，激活 AI 加速器
  val AI_ACCEL_BASE = 0x80000000L.U(32.W)
  val AI_ACCEL_SIZE = 0x00010000L.U(32.W)
  
  val pcpi_addr = cpu.io.pcpi_rs1
  val is_ai_access = (pcpi_addr >= AI_ACCEL_BASE) && 
                     (pcpi_addr < (AI_ACCEL_BASE + AI_ACCEL_SIZE))
  
  // AI 加速器 AXI 接口适配到 PCPI
  val pcpi_state_idle :: pcpi_state_read :: pcpi_state_write :: pcpi_state_done :: Nil = Enum(4)
  val pcpi_state = RegInit(pcpi_state_idle)
  val pcpi_result = RegInit(0.U(32.W))
  
  // 默认 PCPI 信号
  cpu.io.pcpi_wr := false.B
  cpu.io.pcpi_rd := pcpi_result
  cpu.io.pcpi_wait := false.B
  cpu.io.pcpi_ready := false.B
  
  // AI 加速器 AXI 接口默认值
  aiAccel.io.axi.awaddr := 0.U
  aiAccel.io.axi.awvalid := false.B
  aiAccel.io.axi.wdata := 0.U
  aiAccel.io.axi.wvalid := false.B
  aiAccel.io.axi.bready := false.B
  aiAccel.io.axi.araddr := 0.U
  aiAccel.io.axi.arvalid := false.B
  aiAccel.io.axi.rready := false.B
  
  // PCPI 状态机
  switch(pcpi_state) {
    is(pcpi_state_idle) {
      when(cpu.io.pcpi_valid && is_ai_access) {
        val local_addr = (pcpi_addr - AI_ACCEL_BASE)(9, 0)
        
        // 判断是读还是写操作 (通过 pcpi_insn 的 opcode)
        val is_store = cpu.io.pcpi_insn(6, 0) === "b0100011".U
        
        when(is_store) {
          // 写操作
          aiAccel.io.axi.awaddr := local_addr
          aiAccel.io.axi.awvalid := true.B
          aiAccel.io.axi.wdata := cpu.io.pcpi_rs2
          aiAccel.io.axi.wvalid := true.B
          pcpi_state := pcpi_state_write
          cpu.io.pcpi_wait := true.B
        }.otherwise {
          // 读操作
          aiAccel.io.axi.araddr := local_addr
          aiAccel.io.axi.arvalid := true.B
          pcpi_state := pcpi_state_read
          cpu.io.pcpi_wait := true.B
        }
      }
    }
    
    is(pcpi_state_read) {
      aiAccel.io.axi.rready := true.B
      when(aiAccel.io.axi.rvalid) {
        pcpi_result := aiAccel.io.axi.rdata
        pcpi_state := pcpi_state_done
      }.otherwise {
        cpu.io.pcpi_wait := true.B
      }
    }
    
    is(pcpi_state_write) {
      aiAccel.io.axi.bready := true.B
      when(aiAccel.io.axi.bvalid) {
        pcpi_state := pcpi_state_done
      }.otherwise {
        cpu.io.pcpi_wait := true.B
      }
    }
    
    is(pcpi_state_done) {
      cpu.io.pcpi_ready := true.B
      cpu.io.pcpi_wr := true.B
      cpu.io.pcpi_rd := pcpi_result
      pcpi_state := pcpi_state_idle
    }
  }
  
  // 连接 AI 加速器状态到输出
  io.ai_busy := aiAccel.io.status.busy
  io.ai_done := aiAccel.io.status.done
  io.perf_counters := aiAccel.io.perf_counters
}

/**
 * 顶层模块: RISC-V AI 加速器芯片
 * 包含完整的处理器、AI 加速器和内存接口
 */
class RiscvAiChip extends Module {
  val io = IO(new Bundle {
    // 简化的外部接口
    val mem_valid = Output(Bool())
    val mem_ready = Input(Bool())
    val mem_addr = Output(UInt(32.W))
    val mem_wdata = Output(UInt(32.W))
    val mem_wstrb = Output(UInt(4.W))
    val mem_rdata = Input(UInt(32.W))
    
    // 中断
    val irq = Input(UInt(32.W))
    
    // 状态
    val trap = Output(Bool())
    val busy = Output(Bool())
    
    // 性能监控
    val perf_counters = Output(Vec(4, UInt(32.W)))
  })
  
  // 实例化集成系统
  val system = Module(new RiscvAiSystem(
    dataWidth = 32,
    matrixSize = 8,
    numMacUnits = 16,
    memoryDepth = 512
  ))
  
  // 连接外部接口
  io.mem_valid := system.io.mem_valid
  system.io.mem_ready := io.mem_ready
  io.mem_addr := system.io.mem_addr
  io.mem_wdata := system.io.mem_wdata
  io.mem_wstrb := system.io.mem_wstrb
  system.io.mem_rdata := io.mem_rdata
  
  system.io.irq := io.irq
  
  io.trap := system.io.trap
  io.busy := system.io.ai_busy
  io.perf_counters := system.io.perf_counters
}
