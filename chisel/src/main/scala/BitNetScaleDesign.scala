package riscv.ai

import chisel3._
import chisel3.util._

/**
 * BitNet 计算单元
 * 专门处理 {-1, 0, +1} 权重的矩阵乘法
 * 无需乘法器，只需加减法
 */
class BitNetComputeUnit(dataWidth: Int = 16) extends Module {
  val io = IO(new Bundle {
    val activation = Input(SInt(dataWidth.W))  // 激活值 (8-bit 或 16-bit)
    val weight = Input(UInt(2.W))              // 权重编码: 00=0, 01=+1, 10=-1
    val accumulator = Input(SInt(32.W))        // 累加器输入
    val result = Output(SInt(32.W))            // 结果输出
    val valid = Output(Bool())
  })

  // BitNet 计算逻辑：根据权重值进行加/减/跳过
  io.result := MuxCase(io.accumulator, Seq(
    (io.weight === 1.U) -> (io.accumulator + io.activation),  // 权重 = +1: 加法
    (io.weight === 2.U) -> (io.accumulator - io.activation)   // 权重 = -1: 减法
    // 权重 = 0 或其他: 跳过计算，返回累加器值
  ))
  
  io.valid := true.B
}

/**
 * BitNet 矩阵乘法器
 * 专门优化的 16x16 矩阵乘法器，支持 BitNet 权重
 */
class BitNetMatrixMultiplier(
  matrixSize: Int = 16,
  dataWidth: Int = 16
) extends Module {
  val addrWidth = log2Ceil(matrixSize * matrixSize)
  
  val io = IO(new Bundle {
    // 控制接口
    val start = Input(Bool())
    val done = Output(Bool())
    val busy = Output(Bool())
    
    // 激活值接口 (8-bit 或 16-bit)
    val activation = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(SInt(dataWidth.W))
      val readData = Output(SInt(dataWidth.W))
    }
    
    // 权重接口 (2-bit 编码)
    val weight = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(UInt(2.W))
      val readData = Output(UInt(2.W))
    }
    
    // 结果矩阵接口
    val result = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(SInt(32.W))
      val readData = Output(SInt(32.W))
      val valid = Output(Bool())
    }
  })

  // 内部存储器
  val activationMem = Mem(matrixSize * matrixSize, SInt(dataWidth.W))
  val weightMem = Mem(matrixSize * matrixSize, UInt(2.W))
  val resultMem = Mem(matrixSize * matrixSize, SInt(32.W))

  // BitNet 计算单元
  val computeUnit = Module(new BitNetComputeUnit(dataWidth))

  // FSM 状态
  val sIdle :: sCompute :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // 计算计数器
  val totalCycles = matrixSize * matrixSize * matrixSize
  val cycleCounter = RegInit(0.U(log2Ceil(totalCycles + 1).W))

  // 索引计算
  val k = cycleCounter % matrixSize.U
  val temp = cycleCounter / matrixSize.U
  val j = temp % matrixSize.U
  val i = temp / matrixSize.U

  // 激活值读写
  when(io.activation.writeEn) {
    activationMem(io.activation.addr) := io.activation.writeData
  }
  io.activation.readData := Mux(io.activation.readEn, activationMem(io.activation.addr), 0.S)

  // 权重读写
  when(io.weight.writeEn) {
    weightMem(io.weight.addr) := io.weight.writeData
  }
  io.weight.readData := Mux(io.weight.readEn, weightMem(io.weight.addr), 0.U)

  // 结果矩阵读写
  when(io.result.writeEn) {
    resultMem(io.result.addr) := io.result.writeData
  }
  io.result.readData := Mux(io.result.readEn, resultMem(io.result.addr), 0.S)
  io.result.valid := io.result.readEn

  // FSM 状态转换
  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sCompute
        cycleCounter := 0.U
        
        // 初始化结果矩阵
        for (idx <- 0 until matrixSize * matrixSize) {
          resultMem(idx) := 0.S
        }
      }
    }
    
    is(sCompute) {
      cycleCounter := cycleCounter + 1.U
      when(cycleCounter === (totalCycles - 1).U) {
        state := sDone
      }
    }
    
    is(sDone) {
      when(!io.start) {
        state := sIdle
      }
    }
  }

  // 计算逻辑
  val actAddr = i * matrixSize.U + k
  val weightAddr = k * matrixSize.U + j
  val resultAddr = i * matrixSize.U + j

  // 连接计算单元
  computeUnit.io.activation := activationMem(actAddr)
  computeUnit.io.weight := weightMem(weightAddr)
  
  // 累加逻辑
  val lastResult = RegNext(computeUnit.io.result, 0.S)
  val prevResultAddr = RegNext(resultAddr, 0.U)
  val prevK = RegNext(k, 0.U)
  val prevState = RegNext(state, sIdle)
  
  when(state === sCompute) {
    when(k === 0.U) {
      computeUnit.io.accumulator := 0.S
    }.otherwise {
      computeUnit.io.accumulator := lastResult
    }
  }.otherwise {
    computeUnit.io.accumulator := 0.S
  }
  
  // 更新结果
  when(prevState === sCompute && prevK === (matrixSize - 1).U) {
    resultMem(prevResultAddr) := lastResult
  }

  // 输出信号
  io.busy := state =/= sIdle
  io.done := state === sDone
}

/**
 * BitNetScaleAiChip - BitNet 专用 AI 加速芯片
 * 目标：控制在 50,000 instances 以内
 * 配置：
 * - 16个 BitNet 计算单元（无乘法器）
 * - 2个 16x16 BitNet 矩阵乘法器
 * - 压缩权重存储（2-bit/权重）
 * - 8-bit 激活值
 */
class BitNetScaleAiChip(
  dataWidth: Int = 16,           // 激活值位宽（8或16）
  matrixSize: Int = 16,          // 16x16 矩阵
  numComputeUnits: Int = 16,     // 16个计算单元
  numMatrixUnits: Int = 2,       // 2个矩阵乘法器
  memoryDepth: Int = 1024,       // 1K 深度存储器
  addrWidth: Int = 10            // 10位地址
) extends Module {
  
  val io = IO(new Bundle {
    // 简化的 AXI-Lite 接口
    val axi = new Bundle {
      val awaddr = Input(UInt(addrWidth.W))
      val awvalid = Input(Bool())
      val awready = Output(Bool())
      val wdata = Input(UInt(32.W))
      val wvalid = Input(Bool())
      val wready = Output(Bool())
      val bresp = Output(UInt(2.W))
      val bvalid = Output(Bool())
      val bready = Input(Bool())
      val araddr = Input(UInt(addrWidth.W))
      val arvalid = Input(Bool())
      val arready = Output(Bool())
      val rdata = Output(UInt(32.W))
      val rvalid = Output(Bool())
      val rready = Input(Bool())
    }
    
    // 状态接口
    val status = new Bundle {
      val busy = Output(Bool())
      val done = Output(Bool())
    }
    
    // 性能监控
    val perf_counters = Output(Vec(4, UInt(32.W)))
    
    // BitNet 特定配置
    val config = new Bundle {
      val sparsity_enable = Input(Bool())  // 稀疏性优化使能
      val activation_bits = Input(UInt(4.W)) // 激活值位宽配置
    }
  })

  // 16个 BitNet 计算单元
  val computeUnits = Seq.fill(numComputeUnits)(Module(new BitNetComputeUnit(dataWidth)))
  
  // 2个 BitNet 矩阵乘法器（16x16）
  val matrixUnits = Seq.fill(numMatrixUnits)(Module(new BitNetMatrixMultiplier(matrixSize, dataWidth)))
  
  // 存储器（压缩权重 + 激活值）
  val memoryBlock = SyncReadMem(memoryDepth, UInt(32.W))
  
  // 性能计数器
  val perfCounters = Seq.fill(4)(RegInit(0.U(32.W)))
  
  // 控制寄存器
  val ctrlReg = RegInit(0.U(32.W))
  val statusReg = RegInit(0.U(32.W))
  
  // 数据寄存器（用于计算单元）
  val dataRegs = Seq.fill(numComputeUnits)(RegInit(0.S(dataWidth.W)))
  val weightRegs = Seq.fill(numComputeUnits)(RegInit(0.U(2.W)))
  
  // AXI 状态机
  val axi_idle :: axi_active :: Nil = Enum(2)
  val axi_state = RegInit(axi_idle)
  
  val addr_reg = RegInit(0.U(addrWidth.W))
  val data_reg = RegInit(0.U(32.W))
  
  // 工作计数器
  val workCounter = RegInit(0.U(16.W))
  workCounter := workCounter + 1.U
  
  // BitNet 计算单元连接
  for (i <- computeUnits.indices) {
    computeUnits(i).io.activation := dataRegs(i)
    computeUnits(i).io.weight := weightRegs(i)
    computeUnits(i).io.accumulator := (workCounter + i.U).asSInt
    
    // 简化的反馈
    when(computeUnits(i).io.valid && (i.U === (workCounter & 0xF.U))) {
      dataRegs(i) := computeUnits(i).io.result.asSInt
    }
  }
  
  // 矩阵乘法器连接
  // 地址映射：
  // 0-255: 矩阵0激活值
  // 256-511: 矩阵0权重
  // 512-767: 矩阵0结果
  // 768-1023: 矩阵1激活值/权重/结果
  
  for (i <- matrixUnits.indices) {
    val baseAddr = i * 512
    
    // 启动信号
    matrixUnits(i).io.start := ctrlReg(i)
    
    // 激活值接口
    val isActWrite = io.axi.awvalid && io.axi.wvalid && 
      (io.axi.awaddr >= baseAddr.U) && (io.axi.awaddr < (baseAddr + 256).U)
    val isActRead = io.axi.arvalid && 
      (io.axi.araddr >= baseAddr.U) && (io.axi.araddr < (baseAddr + 256).U)
    
    matrixUnits(i).io.activation.writeEn := isActWrite
    matrixUnits(i).io.activation.readEn := isActRead
    matrixUnits(i).io.activation.addr := Mux(io.axi.awvalid, 
      io.axi.awaddr - baseAddr.U, io.axi.araddr - baseAddr.U)
    matrixUnits(i).io.activation.writeData := io.axi.wdata.asSInt
    
    // 权重接口（2-bit 编码）
    val isWeightWrite = io.axi.awvalid && io.axi.wvalid && 
      (io.axi.awaddr >= (baseAddr + 256).U) && (io.axi.awaddr < (baseAddr + 512).U)
    val isWeightRead = io.axi.arvalid && 
      (io.axi.araddr >= (baseAddr + 256).U) && (io.axi.araddr < (baseAddr + 512).U)
    
    matrixUnits(i).io.weight.writeEn := isWeightWrite
    matrixUnits(i).io.weight.readEn := isWeightRead
    matrixUnits(i).io.weight.addr := Mux(io.axi.awvalid, 
      io.axi.awaddr - (baseAddr + 256).U, io.axi.araddr - (baseAddr + 256).U)
    matrixUnits(i).io.weight.writeData := io.axi.wdata(1, 0)
    
    // 结果接口
    val isResultWrite = io.axi.awvalid && io.axi.wvalid && 
      (io.axi.awaddr >= (baseAddr + 512).U) && (io.axi.awaddr < (baseAddr + 768).U)
    val isResultRead = io.axi.arvalid && 
      (io.axi.araddr >= (baseAddr + 512).U) && (io.axi.araddr < (baseAddr + 768).U)
    
    matrixUnits(i).io.result.writeEn := isResultWrite
    matrixUnits(i).io.result.readEn := isResultRead
    matrixUnits(i).io.result.addr := Mux(io.axi.awvalid, 
      io.axi.awaddr - (baseAddr + 512).U, io.axi.araddr - (baseAddr + 512).U)
    matrixUnits(i).io.result.writeData := io.axi.wdata.asSInt
  }
  
  // 控制寄存器地址
  val CTRL_REG_ADDR = 0x300.U
  val STATUS_REG_ADDR = 0x304.U
  
  // AXI 状态机
  switch(axi_state) {
    is(axi_idle) {
      when(io.axi.awvalid || io.axi.arvalid) {
        axi_state := axi_active
        addr_reg := Mux(io.axi.awvalid, io.axi.awaddr, io.axi.araddr)
        
        when(io.axi.awvalid && io.axi.wvalid) {
          data_reg := io.axi.wdata
          // 写控制寄存器
          when(io.axi.awaddr === CTRL_REG_ADDR) {
            ctrlReg := io.axi.wdata
          }
        }.otherwise {
          // 读操作
          data_reg := MuxCase(0.U, Seq(
            (io.axi.araddr === CTRL_REG_ADDR) -> ctrlReg,
            (io.axi.araddr === STATUS_REG_ADDR) -> statusReg,
            (io.axi.araddr < 256.U) -> matrixUnits(0).io.activation.readData.asUInt,
            (io.axi.araddr >= 256.U && io.axi.araddr < 512.U) -> matrixUnits(0).io.weight.readData,
            (io.axi.araddr >= 512.U && io.axi.araddr < 768.U) -> matrixUnits(0).io.result.readData.asUInt
          ))
        }
      }
    }
    
    is(axi_active) {
      axi_state := axi_idle
    }
  }
  
  // AXI 握手信号
  io.axi.awready := axi_state === axi_idle
  io.axi.wready := axi_state === axi_idle
  io.axi.arready := axi_state === axi_idle
  
  io.axi.bvalid := axi_state === axi_active && RegNext(io.axi.awvalid)
  io.axi.bresp := 0.U
  
  io.axi.rvalid := axi_state === axi_active && RegNext(io.axi.arvalid)
  io.axi.rdata := data_reg
  
  // 状态输出
  val computeBusy = computeUnits.map(_.io.valid).reduce(_ || _)
  val matrixBusy = matrixUnits.map(_.io.busy).reduce(_ || _)
  
  io.status.busy := matrixBusy || computeBusy
  io.status.done := matrixUnits.map(_.io.done).reduce(_ || _)
  
  // 性能计数器
  io.perf_counters := VecInit(perfCounters)
  
  when(io.status.busy) {
    perfCounters(0) := perfCounters(0) + 1.U
  }
  when(io.status.done) {
    perfCounters(1) := perfCounters(1) + 1.U
  }
  
  perfCounters(2) := PopCount(computeUnits.map(_.io.valid))
  perfCounters(3) := workCounter.asUInt
  
  // 状态寄存器
  statusReg := Cat(
    0.U(28.W),
    computeBusy,
    io.status.done,
    matrixBusy,
    ctrlReg(0)
  )
}
