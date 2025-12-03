package riscv.ai.peripherals

import chisel3._
import chisel3.util._

class SPIFlash extends Module {
  val io = IO(new Bundle {
    // 寄存器接口
    val reg = new Bundle {
      val valid = Input(Bool())
      val wen = Input(Bool())
      val ren = Input(Bool())
      val addr = Input(UInt(32.W))
      val wdata = Input(UInt(32.W))
      val rdata = Output(UInt(32.W))
      val ready = Output(Bool())
    }
    
    // SPI 物理接口
    val spi_clk = Output(Bool())
    val spi_mosi = Output(Bool())
    val spi_miso = Input(Bool())
    val spi_cs = Output(Bool())
  })
  
  // 寄存器
  val cmdReg = RegInit(0.U(8.W))      // 0x00: 命令
  val addrReg = RegInit(0.U(24.W))    // 0x04: 地址
  val dataReg = RegInit(0.U(32.W))    // 0x08: 数据
  val ctrlReg = RegInit(0.U(32.W))    // 0x0C: 控制 (bit 0: start, bit 1: busy, bit 2: done)
  val statusReg = RegInit(0.U(32.W))  // 0x10: 状态
  
  // SPI 命令定义
  val CMD_READ = 0x03.U
  val CMD_FAST_READ = 0x0B.U
  val CMD_PAGE_PROGRAM = 0x02.U
  val CMD_SECTOR_ERASE = 0x20.U
  
  // 状态机
  val sIdle :: sCommand :: sAddress :: sDummy :: sData :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)
  
  // SPI 时钟分频 (100MHz -> 25MHz, 分频系数 4)
  val clkDiv = RegInit(0.U(2.W))
  val spiClk = RegInit(false.B)
  val clkEn = Wire(Bool())
  
  when(clkDiv === 1.U) {
    clkDiv := 0.U
    spiClk := ~spiClk
    clkEn := spiClk  // 在下降沿采样
  }.otherwise {
    clkDiv := clkDiv + 1.U
    clkEn := false.B
  }
  
  // 位计数器和字节计数器
  val bitCnt = RegInit(0.U(6.W))
  val byteCnt = RegInit(0.U(3.W))
  
  // 移位寄存器
  val shiftReg = RegInit(0.U(32.W))
  val dataOut = RegInit(0.U(32.W))
  
  // 默认输出
  io.reg.ready := true.B
  io.reg.rdata := 0.U
  io.spi_cs := (state =/= sIdle)
  io.spi_clk := spiClk
  io.spi_mosi := shiftReg(31)
  
  // 状态机
  switch(state) {
    is(sIdle) {
      when(ctrlReg(0)) {  // start bit
        state := sCommand
        bitCnt := 0.U
        byteCnt := 0.U
        shiftReg := Cat(cmdReg, 0.U(24.W))
        ctrlReg := ctrlReg | 2.U  // set busy
      }
    }
    
    is(sCommand) {
      when(clkEn) {
        when(bitCnt === 7.U) {
          state := sAddress
          bitCnt := 0.U
          shiftReg := Cat(addrReg, 0.U(8.W))
        }.otherwise {
          bitCnt := bitCnt + 1.U
          shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
        }
      }
    }
    
    is(sAddress) {
      when(clkEn) {
        when(bitCnt === 23.U) {
          // 判断是否需要 dummy cycles
          when(cmdReg === CMD_FAST_READ) {
            state := sDummy
            bitCnt := 0.U
          }.otherwise {
            state := sData
            bitCnt := 0.U
            byteCnt := 0.U
            when(cmdReg === CMD_READ || cmdReg === CMD_FAST_READ) {
              shiftReg := 0.U
            }.otherwise {
              shiftReg := dataReg
            }
          }
        }.otherwise {
          bitCnt := bitCnt + 1.U
          shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
        }
      }
    }
    
    is(sDummy) {
      when(clkEn) {
        when(bitCnt === 7.U) {
          state := sData
          bitCnt := 0.U
          byteCnt := 0.U
          shiftReg := 0.U
        }.otherwise {
          bitCnt := bitCnt + 1.U
        }
      }
    }
    
    is(sData) {
      when(clkEn) {
        when(cmdReg === CMD_READ || cmdReg === CMD_FAST_READ) {
          // 读取数据
          shiftReg := Cat(shiftReg(30, 0), io.spi_miso)
          when(bitCnt === 31.U) {
            dataOut := Cat(shiftReg(30, 0), io.spi_miso)
            state := sDone
          }.otherwise {
            bitCnt := bitCnt + 1.U
          }
        }.otherwise {
          // 写入数据
          when(bitCnt === 31.U) {
            state := sDone
          }.otherwise {
            bitCnt := bitCnt + 1.U
            shiftReg := Cat(shiftReg(30, 0), 0.U(1.W))
          }
        }
      }
    }
    
    is(sDone) {
      dataReg := dataOut
      ctrlReg := (ctrlReg & ~3.U) | 4.U  // clear start/busy, set done
      statusReg := statusReg | 1.U  // operation complete
      state := sIdle
    }
  }
  
  // 寄存器读写
  when(io.reg.valid && io.reg.wen) {
    switch(io.reg.addr(4, 0)) {
      is(0x00.U) { cmdReg := io.reg.wdata(7, 0) }
      is(0x04.U) { addrReg := io.reg.wdata(23, 0) }
      is(0x08.U) { dataReg := io.reg.wdata }
      is(0x0C.U) { 
        ctrlReg := io.reg.wdata
        when(io.reg.wdata(0)) {
          statusReg := 0.U  // clear status on start
        }
      }
      is(0x10.U) { statusReg := io.reg.wdata }
    }
  }
  
  when(io.reg.valid && io.reg.ren) {
    io.reg.rdata := MuxLookup(io.reg.addr(4, 0), 0.U)(Seq(
      0x00.U -> cmdReg,
      0x04.U -> addrReg,
      0x08.U -> dataReg,
      0x0C.U -> ctrlReg,
      0x10.U -> statusReg
    ))
  }
}
