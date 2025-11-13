package riscv.ai

import chisel3._
import chisel3.util._

/**
 * CompactScaleAiChip - 紧凑规模AI芯片
 * 严格目标：控制在10万个instances以内
 * 配置：16个MAC单元 + 1个矩阵乘法器 + 最小化设计
 */
class CompactScaleAiChip(
  dataWidth: Int = 32,
  matrixSize: Int = 8,         // 8x8矩阵 (减小矩阵规模)
  numMacUnits: Int = 16,       // 16个MAC单元 (进一步减少)
  // numMatrixUnits: Int = 1,     // 1个矩阵乘法器 (最小配置) - 固定为1
  memoryDepth: Int = 512,      // 512深度存储器 (进一步减少)
  addrWidth: Int = 10          // 减少地址位宽
) extends Module {
  
  val io = IO(new Bundle {
    // 简化的AXI-Lite接口
    val axi = new Bundle {
      val awaddr = Input(UInt(addrWidth.W))
      val awvalid = Input(Bool())
      val awready = Output(Bool())
      val wdata = Input(UInt(dataWidth.W))
      val wvalid = Input(Bool())
      val wready = Output(Bool())
      val bresp = Output(UInt(2.W))
      val bvalid = Output(Bool())
      val bready = Input(Bool())
      val araddr = Input(UInt(addrWidth.W))
      val arvalid = Input(Bool())
      val arready = Output(Bool())
      val rdata = Output(UInt(dataWidth.W))
      val rvalid = Output(Bool())
      val rready = Input(Bool())
    }
    
    // 最小状态接口
    val status = new Bundle {
      val busy = Output(Bool())
      val done = Output(Bool())
    }
    
    // 最小性能监控 (只保留4个)
    val perf_counters = Output(Vec(4, UInt(32.W)))
  })

  // 16个并行MAC单元阵列
  val macUnits = Seq.fill(numMacUnits)(Module(new MacUnit(dataWidth)))
  
  // 1个矩阵乘法器 (8x8规模)
  val matrixUnit = Module(new MatrixMultiplier(dataWidth, matrixSize))
 
  // 1个存储器块 (最小配置)
  val memoryBlock = SyncReadMem(memoryDepth, UInt(dataWidth.W))
  
  // 最小性能计数器
  val perfCounters = Seq.fill(4)(RegInit(0.U(32.W)))
  
  // 最小控制寄存器
  val ctrlReg = RegInit(0.U(dataWidth.W))
  val statusReg = RegInit(0.U(dataWidth.W))
  
  // 最小数据寄存器
  val dataRegs = Seq.fill(numMacUnits)(RegInit(0.S(dataWidth.W)))
  
  // 简化的AXI状态机
  val axi_idle :: axi_active :: Nil = Enum(2)
  val axi_state = RegInit(axi_idle)
  
  // 最小寄存器组
  val addr_reg = RegInit(0.U(addrWidth.W))
  val data_reg = RegInit(0.U(dataWidth.W))
  
  // 工作计数器
  val workCounter = RegInit(0.U(16.W))  // 减少位宽
  workCounter := workCounter + 1.U
  
  // MAC单元连接 - 简化连接
  for (i <- macUnits.indices) {
    macUnits(i).io.a := dataRegs(i)
    macUnits(i).io.b := (workCounter + i.U).asSInt  
    macUnits(i).io.c := workCounter.asSInt
    
    // 简化的反馈
    when(macUnits(i).io.valid && (i.U === (workCounter & 0xF.U))) {
      dataRegs(i) := macUnits(i).io.result
    }
  }
  
  // 矩阵乘法器连接
  // 地址映射: 0-255: 矩阵A, 256-511: 矩阵B, 512-767: 结果C, 768+: 控制寄存器
  matrixUnit.io.start := ctrlReg(0)
  
  // 写操作检测
  val isMatrixAWrite = io.axi.awvalid && io.axi.wvalid && (io.axi.awaddr < 256.U)
  val isMatrixBWrite = io.axi.awvalid && io.axi.wvalid && (io.axi.awaddr >= 256.U) && (io.axi.awaddr < 512.U)
  val isResultWrite = io.axi.awvalid && io.axi.wvalid && (io.axi.awaddr >= 512.U) && (io.axi.awaddr < 768.U)
  
  // 读操作检测
  val isMatrixARead = io.axi.arvalid && (io.axi.araddr < 256.U)
  val isMatrixBRead = io.axi.arvalid && (io.axi.araddr >= 256.U) && (io.axi.araddr < 512.U)
  val isResultRead = io.axi.arvalid && (io.axi.araddr >= 512.U) && (io.axi.araddr < 768.U)
  
  // 矩阵A接口
  matrixUnit.io.matrixA.writeEn := isMatrixAWrite
  matrixUnit.io.matrixA.readEn := isMatrixARead
  matrixUnit.io.matrixA.addr := Mux(io.axi.awvalid, io.axi.awaddr, io.axi.araddr)
  matrixUnit.io.matrixA.writeData := io.axi.wdata.asSInt
  
  // 矩阵B接口
  matrixUnit.io.matrixB.writeEn := isMatrixBWrite
  matrixUnit.io.matrixB.readEn := isMatrixBRead
  matrixUnit.io.matrixB.addr := Mux(io.axi.awvalid, io.axi.awaddr - 256.U, io.axi.araddr - 256.U)
  matrixUnit.io.matrixB.writeData := io.axi.wdata.asSInt
  
  // 结果矩阵接口
  matrixUnit.io.result.writeEn := isResultWrite
  matrixUnit.io.result.readEn := isResultRead
  matrixUnit.io.result.addr := Mux(io.axi.awvalid, io.axi.awaddr - 512.U, io.axi.araddr - 512.U)
  matrixUnit.io.result.writeData := io.axi.wdata.asSInt

  // 存储器连接 - 简化
  val memBaseAddr = 0x300.U
  val isMemWrite = io.axi.awvalid && io.axi.wvalid && (io.axi.awaddr >= memBaseAddr)
  val isMemRead = io.axi.arvalid && (io.axi.araddr >= memBaseAddr)
  
  when(isMemWrite) {
    memoryBlock.write(io.axi.awaddr - memBaseAddr, io.axi.wdata)
  }
  
  val memReadData = memoryBlock.read(io.axi.araddr - memBaseAddr, isMemRead)
  
  // 控制寄存器地址
  val CTRL_REG_ADDR = 0x300.U
  val STATUS_REG_ADDR = 0x304.U
  
  // 简化的AXI状态机
  switch(axi_state) {
    is(axi_idle) {
      when(io.axi.awvalid || io.axi.arvalid) {
        axi_state := axi_active
        addr_reg := Mux(io.axi.awvalid, io.axi.awaddr, io.axi.araddr)
        when(io.axi.awvalid && io.axi.wvalid) {
          data_reg := io.axi.wdata
          // 写操作 - 控制寄存器
          when(io.axi.awaddr === CTRL_REG_ADDR) {
            ctrlReg := io.axi.wdata
          }
        }.otherwise {
          // 读操作 - 根据地址返回不同数据
          data_reg := MuxCase(0.U, Seq(
            (io.axi.araddr === CTRL_REG_ADDR) -> ctrlReg,
            (io.axi.araddr === STATUS_REG_ADDR) -> statusReg,
            isMatrixARead -> matrixUnit.io.matrixA.readData.asUInt,
            isMatrixBRead -> matrixUnit.io.matrixB.readData.asUInt,
            isResultRead -> matrixUnit.io.result.readData.asUInt,
            isMemRead -> memReadData
          ))
        }
      }
    }
    
    is(axi_active) {
      axi_state := axi_idle
    }
  }
  
  // AXI握手信号 - 简化
  io.axi.awready := axi_state === axi_idle
  io.axi.wready := axi_state === axi_idle
  io.axi.arready := axi_state === axi_idle
  
  io.axi.bvalid := axi_state === axi_active && RegNext(io.axi.awvalid)
  io.axi.bresp := 0.U
  
  io.axi.rvalid := axi_state === axi_active && RegNext(io.axi.arvalid)
  io.axi.rdata := data_reg
  
  // 状态输出 - 最小化
  val macBusy = macUnits.map(_.io.valid).reduce(_ || _)
  
  io.status.busy := matrixUnit.io.busy || macBusy
  io.status.done := matrixUnit.io.done
  
  // 最小性能计数器
  io.perf_counters := VecInit(perfCounters)
  
  // 性能计数器更新 - 简化
  when(io.status.busy) {
    perfCounters(0) := perfCounters(0) + 1.U
  }
  when(io.status.done) {
    perfCounters(1) := perfCounters(1) + 1.U
  }
  
  perfCounters(2) := PopCount(macUnits.map(_.io.valid))
  perfCounters(3) := workCounter.asUInt
  
  // 状态寄存器更新 - 简化
  statusReg := Cat(
    0.U(28.W),
    macBusy,
    io.status.done,
    matrixUnit.io.busy,
    ctrlReg(0)
  )
}