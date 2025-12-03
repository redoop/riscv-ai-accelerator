package riscv.ai.peripherals

import chisel3._
import chisel3.util._

class PSRAM extends Module {
  val io = IO(new Bundle {
    // 寄存器接口
    val reg_wen = Input(Bool())
    val reg_ren = Input(Bool())
    val reg_addr = Input(UInt(32.W))
    val reg_wdata = Input(UInt(32.W))
    val reg_rdata = Output(UInt(32.W))
    
    // SPI/QPI 接口
    val spi_clk = Output(Bool())
    val spi_cs = Output(Bool())
    val spi_mosi = Output(Bool())  // SIO0 输出
    val spi_miso = Input(Bool())   // SIO1 输入
    // Quad SPI 额外信号
    val spi_sio2_out = Output(Bool())
    val spi_sio2_oe = Output(Bool())
    val spi_sio2_in = Input(Bool())
    val spi_sio3_out = Output(Bool())
    val spi_sio3_oe = Output(Bool())
    val spi_sio3_in = Input(Bool())
  })
  
  // 寄存器定义
  val cmdReg = RegInit(0.U(8.W))      // 0x00: 命令寄存器
  val addrReg = RegInit(0.U(24.W))    // 0x04: 地址寄存器 (24-bit)
  val dataReg = RegInit(0.U(32.W))    // 0x08: 数据寄存器
  val ctrlReg = RegInit(0.U(32.W))    // 0x0C: 控制寄存器
  val statusReg = RegInit(0.U(32.W))  // 0x10: 状态寄存器
  val configReg = RegInit(0.U(32.W))  // 0x14: 配置寄存器
  
  // PSRAM 命令定义
  val CMD_READ       = 0x03.U(8.W)  // 标准读取
  val CMD_FAST_READ  = 0x0B.U(8.W)  // 快速读取
  val CMD_WRITE      = 0x02.U(8.W)  // 写入
  val CMD_QUAD_READ  = 0xEB.U(8.W)  // Quad 读取
  val CMD_QUAD_WRITE = 0x38.U(8.W)  // Quad 写入
  val CMD_ENTER_QPI  = 0x35.U(8.W)  // 进入 QPI 模式
  val CMD_EXIT_QPI   = 0xF5.U(8.W)  // 退出 QPI 模式
  
  // 状态机
  val sIdle :: sCommand :: sAddress :: sWait :: sData :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)
  
  // SPI 时钟分频 (100MHz -> 50MHz)
  val clkDiv = RegInit(0.U(1.W))
  val spiClk = RegInit(false.B)
  val spiClkEn = RegInit(false.B)
  
  when(spiClkEn) {
    when(clkDiv === 0.U) {
      spiClk := ~spiClk
      clkDiv := 1.U
    }.otherwise {
      clkDiv := 0.U
    }
  }
  
  // 位计数器
  val bitCnt = RegInit(0.U(6.W))
  
  // 移位寄存器
  val shiftReg = RegInit(0.U(32.W))
  val dataOut = RegInit(0.U(32.W))
  
  // MOSI 输出
  val mosiReg = RegInit(false.B)
  
  // CS 信号
  val csReg = RegInit(true.B)
  
  // QPI 模式标志 (bit 0 of configReg)
  val qpiMode = configReg(0)
  
  // Quad SPI 输出使能和数据
  val sio2OutReg = RegInit(false.B)
  val sio2OeReg = RegInit(false.B)
  val sio3OutReg = RegInit(false.B)
  val sio3OeReg = RegInit(false.B)
  
  // 状态机逻辑
  switch(state) {
    is(sIdle) {
      spiClkEn := false.B
      csReg := true.B  // CS 高电平 (未选中)
      bitCnt := 0.U
      
      // 默认: SIO2/3 输出使能关闭
      sio2OeReg := false.B
      sio3OeReg := false.B
      
      when(ctrlReg(0)) {  // start bit
        // 检查模式切换命令
        when(cmdReg === CMD_ENTER_QPI) {
          configReg := configReg | 1.U  // 设置 QPI 模式标志
          ctrlReg := ctrlReg & ~1.U  // 清除 start bit
          statusReg := statusReg | 2.U  // 设置 done flag
          // 保持在 idle 状态
        }.elsewhen(cmdReg === CMD_EXIT_QPI) {
          configReg := configReg & ~1.U  // 清除 QPI 模式标志
          ctrlReg := ctrlReg & ~1.U  // 清除 start bit
          statusReg := statusReg | 2.U  // 设置 done flag
          // 保持在 idle 状态
        }.otherwise {
          state := sCommand
          spiClkEn := true.B
          csReg := false.B  // CS 低电平 (选中)
          shiftReg := Cat(cmdReg, 0.U(24.W))
          
          // 检查是否需要启用 Quad 模式
          when(cmdReg === CMD_QUAD_READ) {
            sio2OeReg := false.B  // 读模式: SIO2/3 输入
            sio3OeReg := false.B
          }.elsewhen(cmdReg === CMD_QUAD_WRITE) {
            sio2OeReg := true.B   // 写模式: SIO2/3 输出
            sio3OeReg := true.B
          }
        }
      }
    }
    
    is(sCommand) {
      csReg := false.B
      
      when(spiClk && clkDiv === 0.U) {  // 上升沿发送数据
        mosiReg := shiftReg(31)
        shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
        bitCnt := bitCnt + 1.U
        
        when(bitCnt === 7.U) {
          state := sAddress
          bitCnt := 0.U
          shiftReg := Cat(addrReg, 0.U(8.W))
        }
      }
    }
    
    is(sAddress) {
      csReg := false.B
      
      when(spiClk && clkDiv === 0.U) {
        mosiReg := shiftReg(31)
        shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
        bitCnt := bitCnt + 1.U
        
        when(bitCnt === 23.U) {
          // 检查是否需要 dummy cycles
          when(cmdReg === CMD_FAST_READ) {
            state := sWait
            bitCnt := 0.U
          }.otherwise {
            state := sData
            bitCnt := 0.U
            shiftReg := dataReg
          }
        }
      }
    }
    
    is(sWait) {
      csReg := false.B
      
      // Dummy cycles (8 clocks for fast read)
      when(spiClk && clkDiv === 0.U) {
        bitCnt := bitCnt + 1.U
        
        when(bitCnt === 7.U) {
          state := sData
          bitCnt := 0.U
        }
      }
    }
    
    is(sData) {
      csReg := false.B
      
      when(spiClk && clkDiv === 0.U) {
        // Quad 模式: 4-bit 并行传输
        when(cmdReg === CMD_QUAD_READ || cmdReg === CMD_QUAD_WRITE) {
          when(cmdReg === CMD_QUAD_READ) {
            // Quad 读取: 从 SIO[3:0] 读取 4 bits
            val quadIn = Cat(io.spi_sio3_in, io.spi_sio2_in, io.spi_miso, mosiReg)
            dataOut := Cat(dataOut(27, 0), quadIn)
            bitCnt := bitCnt + 4.U
          }.otherwise {
            // Quad 写入: 向 SIO[3:0] 发送 4 bits
            sio3OutReg := shiftReg(31)
            sio2OutReg := shiftReg(30)
            mosiReg := shiftReg(29)  // SIO0
            // SIO1 在写模式下也是输出
            shiftReg := Cat(shiftReg(27, 0), 0.U(4.W))
            bitCnt := bitCnt + 4.U
          }
          
          when(bitCnt >= 28.U) {  // 32 bits / 4 = 8 cycles
            state := sDone
            dataReg := dataOut
          }
        }.otherwise {
          // 标准 SPI 模式: 1-bit 传输
          when(cmdReg === CMD_READ || cmdReg === CMD_FAST_READ) {
            dataOut := Cat(dataOut(30, 0), io.spi_miso)
          }.otherwise {
            mosiReg := shiftReg(31)
            shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
          }
          
          bitCnt := bitCnt + 1.U
          
          when(bitCnt === 31.U) {
            state := sDone
            dataReg := dataOut
          }
        }
      }
    }
    
    is(sDone) {
      spiClkEn := false.B
      csReg := true.B
      ctrlReg := ctrlReg & ~1.U  // 清除 start bit
      statusReg := statusReg | 2.U  // 设置 done flag
      state := sIdle
    }
  }
  
  // 寄存器读写接口
  when(io.reg_wen) {
    switch(io.reg_addr(4, 0)) {
      is(0x00.U) { cmdReg := io.reg_wdata(7, 0) }
      is(0x04.U) { addrReg := io.reg_wdata(23, 0) }
      is(0x08.U) { dataReg := io.reg_wdata }
      is(0x0C.U) { 
        ctrlReg := io.reg_wdata
        statusReg := statusReg & ~2.U  // 清除 done flag
      }
      is(0x14.U) { configReg := io.reg_wdata }
    }
  }
  
  when(io.reg_ren) {
    io.reg_rdata := MuxLookup(io.reg_addr(4, 0), 0.U)(Seq(
      0x00.U -> cmdReg,
      0x04.U -> addrReg,
      0x08.U -> dataReg,
      0x0C.U -> ctrlReg,
      0x10.U -> statusReg,
      0x14.U -> configReg
    ))
  }.otherwise {
    io.reg_rdata := 0.U
  }
  
  // 输出信号
  io.spi_clk := Mux(spiClkEn, spiClk, false.B)
  io.spi_cs := csReg
  io.spi_mosi := mosiReg
  
  // Quad SPI 信号
  io.spi_sio2_out := sio2OutReg
  io.spi_sio2_oe := sio2OeReg
  io.spi_sio3_out := sio3OutReg
  io.spi_sio3_oe := sio3OeReg
}
