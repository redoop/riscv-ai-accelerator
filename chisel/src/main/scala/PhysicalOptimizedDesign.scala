package riscv.ai

import chisel3._
import chisel3.util._

/**
 * 物理优化的MAC单元
 * 针对DRC违例进行优化
 */
class PhysicalOptimizedMacUnit(dataWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(dataWidth.W))
    val b = Input(SInt(dataWidth.W))
    val c = Input(SInt(dataWidth.W))
    val result = Output(SInt(dataWidth.W))
    val valid = Output(Bool())
    val clk_gate_en = Input(Bool()) // 时钟门控使能
  })

  // 添加流水线寄存器减少组合逻辑深度，避免长路径导致的布线拥塞
  val a_reg = RegEnable(io.a, io.clk_gate_en)
  val b_reg = RegEnable(io.b, io.clk_gate_en)
  val c_reg = RegEnable(io.c, io.clk_gate_en)
  
  // 分级乘法器减少单级逻辑复杂度
  val product_stage1 = RegEnable(a_reg * b_reg, io.clk_gate_en)
  val sum_stage2 = RegEnable(product_stage1 + c_reg, io.clk_gate_en)
  
  // 输出寄存器
  val result_reg = RegEnable(sum_stage2, io.clk_gate_en)
  val valid_reg = RegEnable(io.clk_gate_en, io.clk_gate_en)
  
  io.result := result_reg
  io.valid := valid_reg
}

/**
 * 物理优化的矩阵乘法器
 * 解决DRC违例问题
 */
class PhysicalOptimizedMatrixMultiplier(
  dataWidth: Int = 32,
  matrixSize: Int = 4
) extends Module {
  val addrWidth = log2Ceil(matrixSize * matrixSize)
  
  val io = IO(new Bundle {
    // 控制接口
    val start = Input(Bool())
    val done = Output(Bool())
    val busy = Output(Bool())
    
    // 时钟门控控制
    val power_mode = Input(UInt(2.W)) // 00: 低功耗, 01: 正常, 10: 高性能
    
    // 矩阵A接口 - 分离读写端口减少多路复用器复杂度
    val matrixA_write = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Input(SInt(dataWidth.W))
    }
    val matrixA_read = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Output(SInt(dataWidth.W))
    }
    
    // 矩阵B接口
    val matrixB_write = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Input(SInt(dataWidth.W))
    }
    val matrixB_read = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Output(SInt(dataWidth.W))
    }
    
    // 结果矩阵接口
    val result_write = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Input(SInt(dataWidth.W))
    }
    val result_read = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Output(SInt(dataWidth.W))
      val valid = Output(Bool())
    }
  })

  // 使用编译器存储器减少自定义存储器的DRC问题
  val matrixA = SyncReadMem(matrixSize * matrixSize, SInt(dataWidth.W))
  val matrixB = SyncReadMem(matrixSize * matrixSize, SInt(dataWidth.W))
  val matrixResult = SyncReadMem(matrixSize * matrixSize, SInt(dataWidth.W))

  // 时钟门控逻辑
  val clk_gate_en = Wire(Bool())
  clk_gate_en := io.power_mode =/= 0.U && (io.busy || io.start)

  // 优化的MAC单元
  val macUnit = Module(new PhysicalOptimizedMacUnit(dataWidth))
  macUnit.io.clk_gate_en := clk_gate_en

  // FSM状态 - 增加更多中间状态减少单周期逻辑复杂度
  val sIdle :: sInit :: sCompute :: sWriteback :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // 计算计数器
  val totalCycles = matrixSize * matrixSize * matrixSize
  val cycleCounter = RegInit(0.U(log2Ceil(totalCycles + 1).W))

  // 索引寄存器 - 减少组合逻辑
  val i_reg = RegInit(0.U(log2Ceil(matrixSize).W))
  val j_reg = RegInit(0.U(log2Ceil(matrixSize).W))
  val k_reg = RegInit(0.U(log2Ceil(matrixSize).W))

  // 存储器读写逻辑 - 分离以减少多路复用器
  // 矩阵A
  when(io.matrixA_write.en) {
    matrixA.write(io.matrixA_write.addr, io.matrixA_write.data)
  }
  io.matrixA_read.data := matrixA.read(io.matrixA_read.addr, io.matrixA_read.en)

  // 矩阵B
  when(io.matrixB_write.en) {
    matrixB.write(io.matrixB_write.addr, io.matrixB_write.data)
  }
  io.matrixB_read.data := matrixB.read(io.matrixB_read.addr, io.matrixB_read.en)

  // 结果矩阵
  when(io.result_write.en) {
    matrixResult.write(io.result_write.addr, io.result_write.data)
  }
  io.result_read.data := matrixResult.read(io.result_read.addr, io.result_read.en)
  io.result_read.valid := RegNext(io.result_read.en)

  // 累加寄存器
  val accValue = RegInit(0.S(dataWidth.W))

  // 优化的FSM - 减少单状态复杂度
  switch(state) {
    is(sIdle) {
      when(io.start) {
        state := sInit
        cycleCounter := 0.U
        i_reg := 0.U
        j_reg := 0.U
        k_reg := 0.U
      }
    }
    
    is(sInit) {
      // 初始化状态，准备计算
      accValue := 0.S
      state := sCompute
    }
    
    is(sCompute) {
      // 更新索引
      when(k_reg === (matrixSize - 1).U) {
        k_reg := 0.U
        when(j_reg === (matrixSize - 1).U) {
          j_reg := 0.U
          when(i_reg === (matrixSize - 1).U) {
            state := sWriteback
          }.otherwise {
            i_reg := i_reg + 1.U
          }
        }.otherwise {
          j_reg := j_reg + 1.U
        }
      }.otherwise {
        k_reg := k_reg + 1.U
      }
      
      cycleCounter := cycleCounter + 1.U
    }
    
    is(sWriteback) {
      // 写回状态
      state := sDone
    }
    
    is(sDone) {
      when(!io.start) {
        state := sIdle
      }
    }
  }

  // MAC连接 - 使用寄存器地址减少组合逻辑
  val aAddr_reg = RegNext(i_reg * matrixSize.U + k_reg)
  val bAddr_reg = RegNext(k_reg * matrixSize.U + j_reg)
  
  macUnit.io.a := matrixA.read(aAddr_reg, state === sCompute)
  macUnit.io.b := matrixB.read(bAddr_reg, state === sCompute)
  macUnit.io.c := Mux(k_reg === 0.U, 0.S, accValue)

  // 累加逻辑
  when(state === sCompute && macUnit.io.valid) {
    accValue := macUnit.io.result
    
    // 写回结果
    when(k_reg === (matrixSize - 1).U) {
      val resultAddr = i_reg * matrixSize.U + j_reg
      matrixResult.write(resultAddr, macUnit.io.result)
    }
  }

  // 输出信号
  io.busy := state =/= sIdle
  io.done := state === sDone
}

/**
 * 物理优化的RISC-V AI芯片顶层
 * 解决AXI接口和电源网络DRC问题
 */
class PhysicalOptimizedRiscvAiChip(
  dataWidth: Int = 32,
  matrixSize: Int = 4,
  addrWidth: Int = 8
) extends Module {
  val io = IO(new Bundle {
    // 标准AXI-Lite接口 - 完整实现减少接口违例
    val axi = new Bundle {
      // 写地址通道
      val awaddr = Input(UInt(addrWidth.W))
      val awvalid = Input(Bool())
      val awready = Output(Bool())
      
      // 写数据通道
      val wdata = Input(UInt(dataWidth.W))
      val wstrb = Input(UInt((dataWidth/8).W))
      val wvalid = Input(Bool())
      val wready = Output(Bool())
      
      // 写响应通道
      val bresp = Output(UInt(2.W))
      val bvalid = Output(Bool())
      val bready = Input(Bool())
      
      // 读地址通道
      val araddr = Input(UInt(addrWidth.W))
      val arvalid = Input(Bool())
      val arready = Output(Bool())
      
      // 读数据通道
      val rdata = Output(UInt(dataWidth.W))
      val rresp = Output(UInt(2.W))
      val rvalid = Output(Bool())
      val rready = Input(Bool())
    }
    
    // 电源管理接口
    val power_ctrl = new Bundle {
      val mode = Input(UInt(2.W))
      val voltage_ok = Input(Bool())
      val temp_ok = Input(Bool())
    }
    
    // 状态和调试接口
    val status = new Bundle {
      val busy = Output(Bool())
      val done = Output(Bool())
      val error = Output(Bool())
    }
  })

  // 地址解码常量
  val CTRL_REG_ADDR = 0x00.U
  val STATUS_REG_ADDR = 0x04.U
  val POWER_REG_ADDR = 0x08.U
  val MATRIX_A_BASE = 0x10.U
  val MATRIX_B_BASE = 0x50.U
  val RESULT_BASE = 0x90.U

  // 优化的矩阵乘法器
  val matrixMult = Module(new PhysicalOptimizedMatrixMultiplier(dataWidth, matrixSize))

  // AXI状态机 - 标准实现减少协议违例
  val axi_idle :: axi_write :: axi_read :: axi_resp :: Nil = Enum(4)
  val axi_state = RegInit(axi_idle)

  // 寄存器
  val ctrl_reg = RegInit(0.U(dataWidth.W))
  val power_reg = RegInit(0.U(dataWidth.W))
  val status_reg = Wire(UInt(dataWidth.W))

  // 地址和数据寄存器
  val write_addr_reg = RegInit(0.U(addrWidth.W))
  val write_data_reg = RegInit(0.U(dataWidth.W))
  val read_addr_reg = RegInit(0.U(addrWidth.W))
  val read_data_reg = RegInit(0.U(dataWidth.W))

  // AXI写事务处理
  val write_transaction = io.axi.awvalid && io.axi.wvalid
  val read_transaction = io.axi.arvalid

  // AXI状态机
  switch(axi_state) {
    is(axi_idle) {
      when(write_transaction) {
        axi_state := axi_write
        write_addr_reg := io.axi.awaddr
        write_data_reg := io.axi.wdata
      }.elsewhen(read_transaction) {
        axi_state := axi_read
        read_addr_reg := io.axi.araddr
      }
    }
    
    is(axi_write) {
      // 执行写操作
      axi_state := axi_resp
    }
    
    is(axi_read) {
      // 执行读操作
      axi_state := axi_resp
    }
    
    is(axi_resp) {
      when((io.axi.bready && io.axi.bvalid) || (io.axi.rready && io.axi.rvalid)) {
        axi_state := axi_idle
      }
    }
  }

  // AXI握手信号
  io.axi.awready := axi_state === axi_idle
  io.axi.wready := axi_state === axi_idle
  io.axi.arready := axi_state === axi_idle && !write_transaction
  
  io.axi.bvalid := axi_state === axi_resp && RegNext(axi_state === axi_write)
  io.axi.bresp := 0.U // OKAY
  
  io.axi.rvalid := axi_state === axi_resp && RegNext(axi_state === axi_read)
  io.axi.rresp := 0.U // OKAY
  io.axi.rdata := read_data_reg

  // 寄存器写入
  when(axi_state === axi_write) {
    switch(write_addr_reg) {
      is(CTRL_REG_ADDR) { ctrl_reg := write_data_reg }
      is(POWER_REG_ADDR) { power_reg := write_data_reg }
    }
  }

  // 矩阵乘法器连接
  matrixMult.io.start := ctrl_reg(0)
  matrixMult.io.power_mode := io.power_ctrl.mode

  // 状态寄存器
  status_reg := Cat(
    0.U((dataWidth-4).W),
    !io.power_ctrl.voltage_ok || !io.power_ctrl.temp_ok, // error
    matrixMult.io.done,
    matrixMult.io.busy,
    ctrl_reg(0) // start bit
  )

  // 读数据多路复用
  when(axi_state === axi_read) {
    switch(read_addr_reg) {
      is(CTRL_REG_ADDR) { read_data_reg := ctrl_reg }
      is(STATUS_REG_ADDR) { read_data_reg := status_reg }
      is(POWER_REG_ADDR) { read_data_reg := power_reg }
    }
  }

  // 输出状态
  io.status.busy := matrixMult.io.busy
  io.status.done := matrixMult.io.done
  io.status.error := !io.power_ctrl.voltage_ok || !io.power_ctrl.temp_ok

  // 矩阵存储器接口连接（简化版本）
  matrixMult.io.matrixA_write.en := false.B
  matrixMult.io.matrixA_write.addr := 0.U
  matrixMult.io.matrixA_write.data := 0.S
  matrixMult.io.matrixA_read.en := false.B
  matrixMult.io.matrixA_read.addr := 0.U

  matrixMult.io.matrixB_write.en := false.B
  matrixMult.io.matrixB_write.addr := 0.U
  matrixMult.io.matrixB_write.data := 0.S
  matrixMult.io.matrixB_read.en := false.B
  matrixMult.io.matrixB_read.addr := 0.U

  matrixMult.io.result_write.en := false.B
  matrixMult.io.result_write.addr := 0.U
  matrixMult.io.result_write.data := 0.S
  matrixMult.io.result_read.en := false.B
  matrixMult.io.result_read.addr := 0.U
}