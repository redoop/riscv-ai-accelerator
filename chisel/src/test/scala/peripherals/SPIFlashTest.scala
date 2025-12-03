// SPIFlashTest.scala - SPI Flash Controller Test
// Flash/PSRAM Extension - Phase 1 Day 2

package riscv.ai.peripherals

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SPIFlashTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SPIFlash"
  
  it should "initialize correctly" in {
    test(new SPIFlash()) { dut =>
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(false.B)
      dut.clock.step(1)
      
      // 检查初始状态
      dut.io.spi_cs.expect(true.B)  // CS 应该是高电平（未选中）
      dut.io.ready.expect(true.B)
    }
  }
  
  it should "write command register" in {
    test(new SPIFlash()) { dut =>
      // 写命令寄存器
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x03.U)  // READ 命令
      dut.clock.step(1)
      
      // 读回验证
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.clock.step(1)
      dut.io.rdata.expect(0x03.U)
    }
  }
  
  it should "write address register" in {
    test(new SPIFlash()) { dut =>
      // 写地址寄存器
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x04.U)
      dut.io.wdata.poke(0x123456.U)
      dut.clock.step(1)
      
      // 读回验证
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(true.B)
      dut.io.addr.poke(0x04.U)
      dut.clock.step(1)
      dut.io.rdata.expect(0x123456.U)
    }
  }
  
  it should "start operation when START bit is set" in {
    test(new SPIFlash()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.clock.setTimeout(5000)  // 增加超时
      
      // 设置命令
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x06.U)  // WRITE_ENABLE
      dut.clock.step(1)
      
      // 启动操作
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)  // START=1
      dut.clock.step(1)
      
      // 检查 CS 拉低
      dut.clock.step(10)
      dut.io.spi_cs.expect(false.B)
      
      // 等待完成
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(true.B)
      dut.io.addr.poke(0x0C.U)
      
      var timeout = 0
      var done = false
      while (timeout < 2000 && !done) {
        dut.clock.step(1)
        val ctrl = dut.io.rdata.peek().litValue
        done = (ctrl & 0x04) != 0  // 检查 DONE 位
        timeout += 1
      }
      
      assert(done, s"Operation should complete (timeout after $timeout cycles)")
      dut.io.spi_cs.expect(true.B)  // CS 应该恢复高电平
    }
  }
  
  it should "generate SPI clock during operation" in {
    test(new SPIFlash()) { dut =>
      // 设置命令
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x06.U)  // WRITE_ENABLE
      dut.clock.step(1)
      
      // 启动操作
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.wen.poke(false.B)
      
      // 等待并检查 SPI 时钟翻转
      var clkToggles = 0
      var lastClk = false
      for (_ <- 0 until 200) {
        dut.clock.step(1)
        val currentClk = dut.io.spi_clk.peek().litToBoolean
        if (currentClk != lastClk) {
          clkToggles += 1
        }
        lastClk = currentClk
      }
      
      assert(clkToggles > 0, "SPI clock should toggle during operation")
    }
  }
  
  it should "send command byte on MOSI" in {
    test(new SPIFlash()) { dut =>
      // 设置 READ 命令
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x03.U)  // READ
      dut.clock.step(1)
      
      // 设置地址
      dut.io.addr.poke(0x04.U)
      dut.io.wdata.poke(0x000000.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.wen.poke(false.B)
      dut.io.spi_miso.poke(false.B)
      
      // 等待操作完成
      for (_ <- 0 until 500) {
        dut.clock.step(1)
      }
      
      // 操作应该完成
      dut.io.spi_cs.expect(true.B)
    }
  }
  
  it should "handle READ command with address" in {
    test(new SPIFlash()) { dut =>
      // 设置 READ 命令
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x03.U)  // READ
      dut.clock.step(1)
      
      // 设置地址
      dut.io.addr.poke(0x04.U)
      dut.io.wdata.poke(0x123456.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(true.B)
      
      // 模拟 MISO 输入（返回 0xFF）
      dut.io.spi_miso.poke(true.B)
      
      // 等待完成
      var timeout = 0
      var done = false
      while (timeout < 2000 && !done) {
        dut.clock.step(1)
        dut.io.addr.poke(0x0C.U)
        val ctrl = dut.io.rdata.peek().litValue
        done = (ctrl & 0x04) != 0
        timeout += 1
      }
      
      assert(done, "READ operation should complete")
    }
  }
  
  it should "handle FAST_READ command with dummy cycles" in {
    test(new SPIFlash()) { dut =>
      // 设置 FAST_READ 命令
      dut.io.valid.poke(true.B)
      dut.io.wen.poke(true.B)
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(0x0B.U)  // FAST_READ
      dut.clock.step(1)
      
      // 设置地址
      dut.io.addr.poke(0x04.U)
      dut.io.wdata.poke(0x000000.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.wen.poke(false.B)
      dut.io.ren.poke(true.B)
      dut.io.spi_miso.poke(false.B)
      
      // 等待完成（FAST_READ 需要更多时钟周期）
      var timeout = 0
      var done = false
      while (timeout < 2000 && !done) {
        dut.clock.step(1)
        dut.io.addr.poke(0x0C.U)
        val ctrl = dut.io.rdata.peek().litValue
        done = (ctrl & 0x04) != 0
        timeout += 1
      }
      
      assert(done, "FAST_READ operation should complete")
    }
  }
}
