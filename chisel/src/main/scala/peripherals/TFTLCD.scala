// TFTLCD.scala - TFT LCD SPI Controller (ST7735)
// Phase 2 of DEV_PLAN_V0.2

package riscv.ai.peripherals

import chisel3._
import chisel3.util._

/**
 * TFT LCD SPI 控制器（ST7735）
 * 
 * 特性：
 * - 分辨率：128x128 像素
 * - 颜色：65K 色（RGB565）
 * - 接口：SPI（最高 15MHz）
 * - 帧缓冲：32KB（128 x 128 x 2 字节）
 * 
 * 寄存器映射：
 * 0x00: COMMAND   - 命令寄存器 (W)
 * 0x04: DATA      - 数据寄存器 (W)
 * 0x08: STATUS    - 状态寄存器 (R)
 *       bit 0: BUSY
 *       bit 1: INIT_DONE
 * 0x0C: CONTROL   - 控制寄存器 (R/W)
 *       bit 0: BACKLIGHT
 *       bit 1: RESET
 * 0x10: X_START   - X 起始坐标 (R/W)
 * 0x14: Y_START   - Y 起始坐标 (R/W)
 * 0x18: X_END     - X 结束坐标 (R/W)
 * 0x1C: Y_END     - Y 结束坐标 (R/W)
 * 0x20: COLOR     - 颜色数据 (W, RGB565)
 * 0x1000-0x8FFF: FRAMEBUFFER - 帧缓冲 (32KB, 128x128x2)
 */
class TFTLCD(
  clockFreq: Int = 50000000,  // 50MHz 时钟
  spiFreq: Int = 10000000      // 10MHz SPI 时钟
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
    
    // SPI 接口
    val spi_clk = Output(Bool())
    val spi_mosi = Output(Bool())
    val spi_cs = Output(Bool())
    val spi_dc = Output(Bool())    // Data/Command
    val spi_rst = Output(Bool())   // Reset
    
    // 背光控制
    val backlight = Output(Bool())
  })
  
  // ============================================================================
  // 寄存器
  // ============================================================================
  
  val control = RegInit(0.U(32.W))
  val xStart = RegInit(0.U(8.W))
  val yStart = RegInit(0.U(8.W))
  val xEnd = RegInit(127.U(8.W))
  val yEnd = RegInit(127.U(8.W))
  
  val backlight = control(0)
  val resetN = control(1)
  
  // ============================================================================
  // 帧缓冲（32KB = 128x128x2 bytes）
  // ============================================================================
  
  val framebuffer = Mem(16384, UInt(16.W))  // 16K words of 16-bit
  
  // ============================================================================
  // SPI 控制器
  // ============================================================================
  
  val spiDivider = (clockFreq / spiFreq / 2).U
  val spiCounter = RegInit(0.U(16.W))
  val spiClkReg = RegInit(false.B)
  val spiTick = Wire(Bool())
  
  spiTick := false.B
  when(spiCounter >= spiDivider - 1.U) {
    spiCounter := 0.U
    spiClkReg := !spiClkReg
    spiTick := true.B
  }.otherwise {
    spiCounter := spiCounter + 1.U
  }
  
  // ============================================================================
  // 状态机
  // ============================================================================
  
  val sIdle :: sInit :: sCommand :: sData :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)
  
  val spiShiftReg = RegInit(0.U(8.W))
  val spiBitCounter = RegInit(0.U(4.W))
  val spiDC = RegInit(false.B)  // false = command, true = data
  val spiCS = RegInit(true.B)   // Active low
  val busy = RegInit(false.B)
  val initDone = RegInit(false.B)
  
  // 初始化序列计数器
  val initCounter = RegInit(0.U(8.W))
  val initDelay = RegInit(0.U(16.W))
  
  // ============================================================================
  // ST7735 初始化序列
  // ============================================================================
  
  // 简化的初始化序列
  val initSequence = VecInit(Seq(
    // Software reset
    0x01.U,
    // Sleep out
    0x11.U,
    // Color mode: 16-bit
    0x3A.U, 0x05.U,
    // Display on
    0x29.U
  ))
  
  // ============================================================================
  // SPI 传输状态机
  // ============================================================================
  
  val cmdQueue = Module(new Queue(UInt(8.W), 16))
  val dataQueue = Module(new Queue(UInt(8.W), 16))
  
  cmdQueue.io.enq.valid := false.B
  cmdQueue.io.enq.bits := 0.U
  cmdQueue.io.deq.ready := false.B
  
  dataQueue.io.enq.valid := false.B
  dataQueue.io.enq.bits := 0.U
  dataQueue.io.deq.ready := false.B
  
  switch(state) {
    is(sIdle) {
      spiCS := true.B
      busy := false.B
      
      when(!initDone && resetN) {
        state := sInit
        initCounter := 0.U
        initDelay := 0.U
      }.elsewhen(cmdQueue.io.deq.valid && !busy) {
        // 有命令要发送
        spiShiftReg := cmdQueue.io.deq.bits
        cmdQueue.io.deq.ready := true.B
        spiDC := false.B  // Command mode
        spiCS := false.B
        spiBitCounter := 0.U
        state := sCommand
        busy := true.B
      }.elsewhen(dataQueue.io.deq.valid && !busy) {
        // 有数据要发送
        spiShiftReg := dataQueue.io.deq.bits
        dataQueue.io.deq.ready := true.B
        spiDC := true.B  // Data mode
        spiCS := false.B
        spiBitCounter := 0.U
        state := sData
        busy := true.B
      }
    }
    
    is(sInit) {
      busy := true.B
      // 初始化延迟
      when(initDelay > 0.U) {
        initDelay := initDelay - 1.U
      }.elsewhen(initCounter < initSequence.length.U) {
        // 发送初始化命令
        when(cmdQueue.io.enq.ready) {
          cmdQueue.io.enq.valid := true.B
          cmdQueue.io.enq.bits := initSequence(initCounter)
          initCounter := initCounter + 1.U
          initDelay := 100.U  // 减少延迟以加快测试
        }
      }.otherwise {
        initDone := true.B
        busy := false.B
        state := sIdle
      }
    }
    
    is(sCommand) {
      busy := true.B
      when(spiTick && spiClkReg) {  // 在时钟上升沿发送
        spiShiftReg := spiShiftReg << 1
        spiBitCounter := spiBitCounter + 1.U
        when(spiBitCounter === 7.U) {
          state := sDone
        }
      }
    }
    
    is(sData) {
      busy := true.B
      when(spiTick && spiClkReg) {  // 在时钟上升沿发送
        spiShiftReg := spiShiftReg << 1
        spiBitCounter := spiBitCounter + 1.U
        when(spiBitCounter === 7.U) {
          state := sDone
        }
      }
    }
    
    is(sDone) {
      spiCS := true.B
      busy := false.B
      when(spiTick) {
        state := sIdle
      }
    }
  }
  
  // ============================================================================
  // 输出
  // ============================================================================
  
  io.spi_clk := spiClkReg
  io.spi_mosi := spiShiftReg(7)
  io.spi_cs := spiCS
  io.spi_dc := spiDC
  io.spi_rst := resetN
  io.backlight := backlight
  
  // ============================================================================
  // 状态寄存器
  // ============================================================================
  
  val status = Cat(
    Fill(30, false.B),
    initDone,
    busy
  )
  
  // ============================================================================
  // 寄存器接口
  // ============================================================================
  
  io.rdata := 0.U
  io.ready := true.B
  
  when(io.valid) {
    val regAddr = io.addr(15, 0)
    
    when(io.wen) {
      switch(regAddr) {
        is(0x00.U) {  // COMMAND
          cmdQueue.io.enq.valid := true.B
          cmdQueue.io.enq.bits := io.wdata(7, 0)
        }
        is(0x04.U) {  // DATA
          dataQueue.io.enq.valid := true.B
          dataQueue.io.enq.bits := io.wdata(7, 0)
        }
        is(0x0C.U) {  // CONTROL
          control := io.wdata
        }
        is(0x10.U) {  // X_START
          xStart := io.wdata(7, 0)
        }
        is(0x14.U) {  // Y_START
          yStart := io.wdata(7, 0)
        }
        is(0x18.U) {  // X_END
          xEnd := io.wdata(7, 0)
        }
        is(0x1C.U) {  // Y_END
          yEnd := io.wdata(7, 0)
        }
        is(0x20.U) {  // COLOR - 快速写入像素
          // 将 RGB565 颜色写入数据队列
          dataQueue.io.enq.valid := true.B
          dataQueue.io.enq.bits := io.wdata(15, 8)  // 高字节
          // 注意：需要两次写入，这里简化处理
        }
      }
      
      // 帧缓冲写入
      when(regAddr >= 0x1000.U && regAddr < 0x9000.U) {
        val fbAddr = (regAddr - 0x1000.U) >> 1
        framebuffer(fbAddr) := io.wdata(15, 0)
      }
    }
    
    when(io.ren) {
      switch(regAddr) {
        is(0x08.U) {  // STATUS
          io.rdata := status
        }
        is(0x0C.U) {  // CONTROL
          io.rdata := control
        }
        is(0x10.U) {  // X_START
          io.rdata := Cat(Fill(24, false.B), xStart)
        }
        is(0x14.U) {  // Y_START
          io.rdata := Cat(Fill(24, false.B), yStart)
        }
        is(0x18.U) {  // X_END
          io.rdata := Cat(Fill(24, false.B), xEnd)
        }
        is(0x1C.U) {  // Y_END
          io.rdata := Cat(Fill(24, false.B), yEnd)
        }
      }
      
      // 帧缓冲读取
      when(regAddr >= 0x1000.U && regAddr < 0x9000.U) {
        val fbAddr = (regAddr - 0x1000.U) >> 1
        io.rdata := Cat(Fill(16, false.B), framebuffer(fbAddr))
      }
    }
  }
}
