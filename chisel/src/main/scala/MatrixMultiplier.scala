package riscv.ai

import chisel3._
import chisel3.util._

/**
 * MAC (Multiply-Accumulate) Unit
 * 执行 result = a * b + c 运算
 */
class MacUnit(dataWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(dataWidth.W))
    val b = Input(SInt(dataWidth.W))
    val c = Input(SInt(dataWidth.W))
    val result = Output(SInt(dataWidth.W))
    val valid = Output(Bool())
  })

  // MAC运算：result = a * b + c
  // 使用更宽的中间结果避免溢出
  val product = io.a * io.b
  val sum = product + io.c
  
  // 直接输出结果
  io.result := sum
  io.valid := true.B // 组合逻辑，立即有效
}

/**
 * 矩阵乘法器
 * 支持可配置大小的矩阵乘法运算
 */
class MatrixMultiplier(
  dataWidth: Int = 32,
  matrixSize: Int = 4
) extends Module {
  val addrWidth = log2Ceil(matrixSize * matrixSize)
  
  val io = IO(new Bundle {
    // 控制接口
    val start = Input(Bool())
    val done = Output(Bool())
    val busy = Output(Bool())
    
    // 矩阵A接口
    val matrixA = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(SInt(dataWidth.W))
      val readData = Output(SInt(dataWidth.W))
    }
    
    // 矩阵B接口
    val matrixB = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(SInt(dataWidth.W))
      val readData = Output(SInt(dataWidth.W))
    }
    
    // 结果矩阵接口
    val result = new Bundle {
      val writeEn = Input(Bool())
      val readEn = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(SInt(dataWidth.W))
      val readData = Output(SInt(dataWidth.W))
      val valid = Output(Bool())
    }
  })

  // 内部存储器
  val matrixA = Mem(matrixSize * matrixSize, SInt(dataWidth.W))
  val matrixB = Mem(matrixSize * matrixSize, SInt(dataWidth.W))
  val matrixResult = Mem(matrixSize * matrixSize, SInt(dataWidth.W))

  // MAC单元
  val macUnit = Module(new MacUnit(dataWidth))

  // FSM状态
  val sIdle :: sCompute :: sDone :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // 计算计数器 - 对于2x2矩阵，需要8个周期
  val totalCycles = matrixSize * matrixSize * matrixSize
  val cycleCounter = RegInit(0.U(log2Ceil(totalCycles + 1).W))
  val computationComplete = RegInit(false.B)

  // 索引计算 - 正确的矩阵乘法遍历顺序：i(行), j(列), k(累加维度)
  // 遍历顺序：对每个(i,j)，k从0到matrixSize-1
  val k = cycleCounter % matrixSize.U
  val temp = cycleCounter / matrixSize.U
  val j = temp % matrixSize.U
  val i = temp / matrixSize.U

  // 延迟寄存器用于流水线
  val iDelayed = RegNext(i)
  val jDelayed = RegNext(j)
  val kDelayed = RegNext(k)

  // 不需要单独的累加器，直接使用结果矩阵

  // 矩阵A读写
  when(io.matrixA.writeEn) {
    matrixA(io.matrixA.addr) := io.matrixA.writeData
  }
  io.matrixA.readData := Mux(io.matrixA.readEn, matrixA(io.matrixA.addr), 0.S)

  // 矩阵B读写
  when(io.matrixB.writeEn) {
    matrixB(io.matrixB.addr) := io.matrixB.writeData
  }
  io.matrixB.readData := Mux(io.matrixB.readEn, matrixB(io.matrixB.addr), 0.S)

  // 结果矩阵读写
  when(io.result.writeEn) {
    matrixResult(io.result.addr) := io.result.writeData
  }
  
  io.result.readData := Mux(io.result.readEn, matrixResult(io.result.addr), 0.S)
  io.result.valid := io.result.readEn

  // FSM状态转换
  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sCompute
        cycleCounter := 0.U
        computationComplete := false.B
        
        // 在开始计算时清零结果矩阵
        for (idx <- 0 until matrixSize * matrixSize) {
          matrixResult(idx) := 0.S
        }
      }
    }
    
    is(sCompute) {
      cycleCounter := cycleCounter + 1.U
      when(cycleCounter === (totalCycles - 1).U) {
        computationComplete := true.B
        state := sDone
      }
    }
    
    is(sDone) {
      when(!io.start) {
        state := sIdle
      }
    }
  }

  // MAC输入连接
  val aAddr = i * matrixSize.U + k
  val bAddr = k * matrixSize.U + j
  val cAddr = i * matrixSize.U + j

  macUnit.io.a := matrixA(aAddr)
  macUnit.io.b := matrixB(bAddr)
  
  // 累加逻辑 - 使用延迟寄存器避免组合环路
  val resultAddr = i * matrixSize.U + j
  val prevResultAddr = RegNext(resultAddr, 0.U)
  val prevK = RegNext(k, 0.U)
  val prevState = RegNext(state, sIdle)
  // 再延迟一级
  val prevResultAddr2 = RegNext(prevResultAddr, 0.U)
  val prevK2 = RegNext(prevK, 0.U)
  val prevState2 = RegNext(prevState, sIdle)
  
  // 保存上一个周期的MAC结果
  val lastMacResult = RegNext(macUnit.io.result, 0.S)
  // 再延迟一个周期，用于写入
  val lastMacResult2 = RegNext(lastMacResult, 0.S)
  
  // MAC输入
  when(state === sCompute) {
    when(k === 0.U) {
      // 新的(i,j)开始，从0开始累加
      macUnit.io.c := 0.S
    }.otherwise {
      // 继续累加 - 使用上一个周期的MAC结果
      // 因为上一个周期计算的是同一个(i,j)位置的k-1
      macUnit.io.c := lastMacResult
    }
  }.otherwise {
    macUnit.io.c := 0.S
  }
  
  // 更新结果矩阵 - 只在k到达最后一个值时写入最终结果
  // 使用延迟2个周期的信号，确保写入的是正确的最终结果
  when(prevState2 === sCompute && prevK2 === (matrixSize - 1).U) {
    matrixResult(prevResultAddr2) := lastMacResult2
  }

  // 输出信号
  io.busy := state =/= sIdle
  io.done := state === sDone
}

/**
 * RISC-V AI芯片顶层模块
 */
class RiscvAiChip(
  dataWidth: Int = 32,
  matrixSize: Int = 4,
  addrWidth: Int = 8
) extends Module {
  val io = IO(new Bundle {
    // AXI-Lite控制接口
    val ctrl = new Bundle {
      val valid = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val writeData = Input(UInt(dataWidth.W))
      val writeEn = Input(Bool())
      val readData = Output(UInt(dataWidth.W))
      val ready = Output(Bool())
    }
    
    // 状态输出
    val aiAcceleratorBusy = Output(Bool())
    val computationDone = Output(Bool())
    
    // 调试接口
    val debugState = Output(UInt(8.W))
    val debugCounter = Output(UInt(16.W))
  })

  // 寄存器映射地址
  val CTRL_REG = 0x00.U
  val STATUS_REG = 0x04.U
  val MATRIX_A_BASE = 0x10.U
  val MATRIX_B_BASE = 0x50.U
  val RESULT_BASE = 0x90.U

  // 矩阵乘法器实例
  val matrixMult = Module(new MatrixMultiplier(dataWidth, matrixSize))

  // 控制寄存器
  val ctrlRegister = RegInit(0.U(dataWidth.W))
  val statusRegister = Wire(UInt(dataWidth.W))

  // 保留寄存器（用于测试）
  val reservedRegs = Mem(16, UInt(dataWidth.W))

  // 控制寄存器写入
  when(io.ctrl.valid && io.ctrl.writeEn && io.ctrl.addr === CTRL_REG) {
    ctrlRegister := io.ctrl.writeData
  }

  // 保留寄存器写入
  when(io.ctrl.valid && io.ctrl.writeEn && 
       io.ctrl.addr < MATRIX_A_BASE && 
       io.ctrl.addr =/= CTRL_REG && 
       io.ctrl.addr =/= STATUS_REG) {
    reservedRegs(io.ctrl.addr(3, 0)) := io.ctrl.writeData
  }

  // 状态寄存器
  statusRegister := Cat(
    0.U((dataWidth-3).W),
    matrixMult.io.result.valid,
    matrixMult.io.done,
    matrixMult.io.busy
  )

  // 矩阵乘法器连接
  matrixMult.io.start := ctrlRegister(0)

  // 矩阵A接口
  matrixMult.io.matrixA.writeEn := io.ctrl.valid && io.ctrl.writeEn && 
    io.ctrl.addr >= MATRIX_A_BASE && io.ctrl.addr < MATRIX_B_BASE
  matrixMult.io.matrixA.readEn := io.ctrl.valid && !io.ctrl.writeEn && 
    io.ctrl.addr >= MATRIX_A_BASE && io.ctrl.addr < MATRIX_B_BASE
  matrixMult.io.matrixA.addr := io.ctrl.addr - MATRIX_A_BASE
  matrixMult.io.matrixA.writeData := io.ctrl.writeData.asSInt

  // 矩阵B接口
  matrixMult.io.matrixB.writeEn := io.ctrl.valid && io.ctrl.writeEn && 
    io.ctrl.addr >= MATRIX_B_BASE && io.ctrl.addr < RESULT_BASE
  matrixMult.io.matrixB.readEn := io.ctrl.valid && !io.ctrl.writeEn && 
    io.ctrl.addr >= MATRIX_B_BASE && io.ctrl.addr < RESULT_BASE
  matrixMult.io.matrixB.addr := io.ctrl.addr - MATRIX_B_BASE
  matrixMult.io.matrixB.writeData := io.ctrl.writeData.asSInt

  // 结果矩阵接口
  matrixMult.io.result.writeEn := io.ctrl.valid && io.ctrl.writeEn && 
    io.ctrl.addr >= RESULT_BASE
  matrixMult.io.result.readEn := io.ctrl.valid && !io.ctrl.writeEn && 
    io.ctrl.addr >= RESULT_BASE
  matrixMult.io.result.addr := io.ctrl.addr - RESULT_BASE
  matrixMult.io.result.writeData := io.ctrl.writeData.asSInt

  // 读数据多路复用器
  io.ctrl.readData := MuxLookup(io.ctrl.addr, 0.U)(
    Seq(
      CTRL_REG -> ctrlRegister,
      STATUS_REG -> statusRegister
    ) ++
    (0 until 16).map(i => (i.U -> Mux(io.ctrl.addr < MATRIX_A_BASE, reservedRegs(i.U), 0.U))) ++
    (0x1000 until 0x2000).map(addr => (addr.U -> matrixMult.io.matrixA.readData.asUInt)) ++
    (0x2000 until 0x3000).map(addr => (addr.U -> matrixMult.io.matrixB.readData.asUInt)) ++
    Seq((0x3000.U -> matrixMult.io.result.readData.asUInt))
  )

  // 输出信号
  io.ctrl.ready := true.B
  io.aiAcceleratorBusy := matrixMult.io.busy
  io.computationDone := matrixMult.io.done
  io.debugState := Cat(0.U(6.W), matrixMult.io.done, matrixMult.io.busy)
  io.debugCounter := 0x1234.U
}