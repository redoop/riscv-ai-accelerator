// SPIFlash.scala - SPI Flash Controller (W25Q128)
// Flash/PSRAM Extension - Phase 1 Day 1
// Target: 16MB SPI Flash support

package riscv.ai.peripherals

import chisel3._
import chisel3.util._

/**
 * SPI Flash 控制器 (W25Q128 兼容)
 * 
 * 特性：
 * - 容量：16MB (128Mbit)
 * - 接口：标准 SPI (最高 25MHz)
 * - 页大小：256 字节
 * - 扇区大小：4KB
 * 
 * 寄存器映射：
 * 0x00: CMD    - 命令寄存器 (W)
 * 0x04: ADDR   - 地址寄存器 (R/W) [23:0]
 * 0x08: DATA   - 数据寄存器 (R/W)
 * 0x0C: CTRL   - 控制寄存器 (R/W)
 *       bit 0: START - 启动操作
 *       bit 1: BUSY  - 忙标志 (R)
 *       bit 2: DONE  - 完成标志 (R)
 * 0x10: STATUS - Flash 状态寄存器 (R)
 * 
 * 支持的命令：
 * 0x03: READ          - 标准读取
 * 0x0B: FAST_READ     - 快速读取
 * 0x02: PAGE_PROGRAM  - 页编程
 * 0x20: SECTOR_ERASE  - 扇区擦除 (4KB)
 * 0x06: WRITE_ENABLE  - 写使能
 * 0x05: READ_STATUS   - 读状态
 */
class SPIFlash(
  clockFreq: Int = 100000000,  // 100MHz 系统时钟
  spiFreq: Int = 25000000       // 25MHz SPI 时钟
) extends Module {
  val io = IO(new Bundle {
    // 寄存器接口
    val addr = Input(UInt(32.W))
    val wdata = Input(UInt(32.W))
    val rdata = Output(UInt(32.W))
    val wen = Input(Bool())
    val ren = Input(Bool())
    val valid = Input(Bool())
    val ready = Output(Bool())
    
    // SPI 物理接口
    val spi_clk = Output(Bool())
    val spi_mosi = Output(Bool())
    val spi_miso = Input(Bool())
    val spi_cs = Output(Bool())
  })
  
  // ============================================================================
  // 寄存器定义
  // ============================================================================
  
  val cmdReg = RegInit(0.U(8.W))
  val addrReg = RegInit(0.U(24.W))
  val dataReg = RegInit(0.U(32.W))
  val statusReg = RegInit(0.U(32.W))
  
  // 控制位 - 分开定义
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)
  val startReq = WireDefault(false.B)
  
  // ============================================================================
  // SPI 时钟生成 (100MHz → 25MHz)
  // ============================================================================
  
  val spiDivider = (clockFreq / spiFreq / 2).U
  val spiCounter = RegInit(0.U(8.W))
  val spiClkReg = RegInit(false.B)
  
  when(busyReg && spiCounter >= spiDivider - 1.U) {
    spiCounter := 0.U
    spiClkReg := !spiClkReg
  }.elsewhen(busyReg) {
    spiCounter := spiCounter + 1.U
  }.otherwise {
    spiCounter := 0.U
    spiClkReg := false.B
  }
  
  // ============================================================================
  // 状态机
  // ============================================================================
  
  val sIdle :: sCommand :: sAddress :: sDummy :: sData :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)
  
  val bitCounter = RegInit(0.U(8.W))
  val byteCounter = RegInit(0.U(8.W))
  val shiftReg = RegInit(0.U(32.W))
  
  val csReg = RegInit(true.B)  // CS 高电平（未选中）
  val mosiReg = RegInit(false.B)
  
  // SPI 时钟边沿检测
  val spiClkLast = RegNext(spiClkReg)
  val spiPosEdge = spiClkReg && !spiClkLast
  val spiNegEdge = !spiClkReg && spiClkLast
  
  // ============================================================================
  // 状态机逻辑
  // ============================================================================
  
  switch(state) {
    is(sIdle) {
      csReg := true.B
      busyReg := false.B
      
      when(startReq) {
        state := sCommand
        csReg := false.B
        bitCounter := 0.U
        shiftReg := Cat(cmdReg, 0.U(24.W))
        busyReg := true.B
        doneReg := false.B
      }
    }
    
    is(sCommand) {
      when(spiNegEdge) {
        mosiReg := shiftReg(31)
        shiftReg := Cat(shiftReg(30, 0), false.B)
        bitCounter := bitCounter + 1.U
        
        when(bitCounter === 7.U) {
          // 命令发送完成
          when(cmdReg === 0x03.U || cmdReg === 0x0B.U || 
               cmdReg === 0x02.U || cmdReg === 0x20.U) {
            // 需要地址的命令
            state := sAddress
            bitCounter := 0.U
            shiftReg := Cat(addrReg, 0.U(8.W))
          }.elsewhen(cmdReg === 0x05.U) {
            // READ_STATUS - 直接读数据
            state := sData
            bitCounter := 0.U
            byteCounter := 0.U
          }.otherwise {
            // 其他命令（如 WRITE_ENABLE）
            state := sDone
          }
        }
      }
    }
    
    is(sAddress) {
      when(spiNegEdge) {
        mosiReg := shiftReg(31)
        shiftReg := Cat(shiftReg(30, 0), false.B)
        bitCounter := bitCounter + 1.U
        
        when(bitCounter === 23.U) {
          // 地址发送完成
          when(cmdReg === 0x0B.U) {
            // FAST_READ 需要 dummy cycles
            state := sDummy
            bitCounter := 0.U
          }.otherwise {
            // 其他命令直接进入数据阶段
            state := sData
            bitCounter := 0.U
            byteCounter := 0.U
            when(cmdReg === 0x02.U) {
              // PAGE_PROGRAM - 准备写数据
              shiftReg := dataReg
            }
          }
        }
      }
    }
    
    is(sDummy) {
      // Dummy cycles (8 个时钟)
      when(spiNegEdge) {
        bitCounter := bitCounter + 1.U
        when(bitCounter === 7.U) {
          state := sData
          bitCounter := 0.U
          byteCounter := 0.U
        }
      }
    }
    
    is(sData) {
      when(cmdReg === 0x03.U || cmdReg === 0x0B.U || cmdReg === 0x05.U) {
        // 读操作
        when(spiPosEdge) {
          shiftReg := Cat(shiftReg(30, 0), io.spi_miso)
          bitCounter := bitCounter + 1.U
          
          when(bitCounter === 31.U) {
            dataReg := Cat(shiftReg(30, 0), io.spi_miso)
            state := sDone
          }
        }
      }.otherwise {
        // 写操作
        when(spiNegEdge) {
          mosiReg := shiftReg(31)
          shiftReg := Cat(shiftReg(30, 0), false.B)
          bitCounter := bitCounter + 1.U
          
          when(bitCounter === 31.U) {
            state := sDone
          }
        }
      }
    }
    
    is(sDone) {
      csReg := true.B
      busyReg := false.B
      doneReg := true.B
      state := sIdle
    }
  }
  
  // ============================================================================
  // 寄存器读写
  // ============================================================================
  
  io.ready := true.B
  io.rdata := 0.U
  
  when(io.valid && io.wen) {
    switch(io.addr(7, 0)) {
      is(0x00.U) { cmdReg := io.wdata(7, 0) }
      is(0x04.U) { addrReg := io.wdata(23, 0) }
      is(0x08.U) { dataReg := io.wdata }
      is(0x0C.U) { 
        // 只允许写 START 位
        startReq := io.wdata(0)
      }
    }
  }
  
  when(io.valid && io.ren) {
    switch(io.addr(7, 0)) {
      is(0x00.U) { io.rdata := cmdReg }
      is(0x04.U) { io.rdata := addrReg }
      is(0x08.U) { io.rdata := dataReg }
      is(0x0C.U) { io.rdata := Cat(0.U(29.W), doneReg, busyReg, false.B) }
      is(0x10.U) { io.rdata := statusReg }
    }
  }
  
  // ============================================================================
  // 输出连接
  // ============================================================================
  
  io.spi_clk := spiClkReg
  io.spi_mosi := mosiReg
  io.spi_cs := csReg
}

/**
 * 顶层包装器 - 用于 Verilog 生成
 */
object SPIFlashMain extends App {
  emitVerilog(
    new SPIFlash(),
    Array("--target-dir", "generated/spiflash")
  )
}
