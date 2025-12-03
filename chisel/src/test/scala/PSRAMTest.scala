package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import riscv.ai.peripherals.PSRAM

class PSRAMTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "PSRAM Controller"
  
  it should "initialize correctly" in {
    test(new PSRAM()) { dut =>
      // 检查初始状态
      dut.io.spi_cs.expect(true.B)  // CS 应该是高电平 (未选中)
      dut.io.spi_clk.expect(false.B)
      
      // 读取状态寄存器
      dut.io.reg_ren.poke(true.B)
      dut.io.reg_addr.poke(0x10.U)
      dut.clock.step(1)
      dut.io.reg_rdata.expect(0.U)
    }
  }
  
  it should "write and read command register" in {
    test(new PSRAM()) { dut =>
      // 写入命令寄存器
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x03.U)  // READ command
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 读取命令寄存器
      dut.io.reg_ren.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.clock.step(1)
      dut.io.reg_rdata.expect(0x03.U)
    }
  }
  
  it should "write and read address register" in {
    test(new PSRAM()) { dut =>
      // 写入地址寄存器
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x123456.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 读取地址寄存器
      dut.io.reg_ren.poke(true.B)
      dut.io.reg_addr.poke(0x04.U)
      dut.clock.step(1)
      dut.io.reg_rdata.expect(0x123456.U)
    }
  }
  
  it should "start read operation" in {
    test(new PSRAM()) { dut =>
      // 配置读取操作
      dut.io.reg_wen.poke(true.B)
      
      // 设置命令
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x03.U)  // READ
      dut.clock.step(1)
      
      // 设置地址
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x001000.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)  // start bit
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 验证 CS 拉低
      dut.clock.step(2)
      dut.io.spi_cs.expect(false.B)
      
      // 等待操作完成
      var cycles = 0
      var done = false
      while(cycles < 400 && !done) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)  // 状态寄存器
        dut.clock.step(1)
        
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "Operation should complete within 400 cycles")
      
      // 验证 CS 恢复高电平
      dut.io.spi_cs.expect(true.B)
    }
  }
  
  it should "generate SPI clock during operation" in {
    test(new PSRAM()) { dut =>
      // 启动读取操作
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x03.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待几个周期
      dut.clock.step(5)
      
      // 检查 SPI 时钟是否在切换
      var clkToggled = false
      var lastClk = dut.io.spi_clk.peek().litToBoolean
      
      for(_ <- 0 until 20) {
        dut.clock.step(1)
        val currentClk = dut.io.spi_clk.peek().litToBoolean
        if(currentClk != lastClk) {
          clkToggled = true
        }
        lastClk = currentClk
      }
      
      assert(clkToggled, "SPI clock should toggle during operation")
    }
  }
  
  it should "handle fast read command" in {
    test(new PSRAM()) { dut =>
      // 配置快速读取
      dut.io.reg_wen.poke(true.B)
      
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x0B.U)  // FAST_READ
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x002000.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成 (快速读取需要更多周期因为有 dummy cycles)
      var cycles = 0
      var done = false
      while(cycles < 500 && !done) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "Fast read should complete within 500 cycles")
    }
  }
  
  it should "handle write operation" in {
    test(new PSRAM()) { dut =>
      // 配置写入操作
      dut.io.reg_wen.poke(true.B)
      
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x02.U)  // WRITE
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x003000.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x08.U)
      dut.io.reg_wdata.poke("hDEADBEEF".U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while(cycles < 400 && !done) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "Write operation should complete within 400 cycles")
    }
  }
  
  it should "clear done flag on new operation" in {
    test(new PSRAM()) { dut =>
      // 第一次操作
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x03.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      while(cycles < 200) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          cycles = 200
        }
        cycles += 1
      }
      
      // 启动第二次操作 - done flag 应该被清除
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 检查 done flag 被清除
      dut.io.reg_ren.poke(true.B)
      dut.io.reg_addr.poke(0x10.U)
      dut.clock.step(1)
      
      val status = dut.io.reg_rdata.peek().litValue
      assert((status & 2) == 0, "Done flag should be cleared on new operation")
    }
  }
  
  it should "support quad read operation" in {
    test(new PSRAM()) { dut =>
      // 配置 Quad 读取
      dut.io.reg_wen.poke(true.B)
      
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0xEB.U)  // QUAD_READ
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x004000.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成 (Quad 模式应该更快)
      var cycles = 0
      var done = false
      while(cycles < 300 && !done) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "Quad read should complete within 300 cycles")
    }
  }
  
  it should "support quad write operation" in {
    test(new PSRAM()) { dut =>
      // 配置 Quad 写入
      dut.io.reg_wen.poke(true.B)
      
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x38.U)  // QUAD_WRITE
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x04.U)
      dut.io.reg_wdata.poke(0x005000.U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x08.U)
      dut.io.reg_wdata.poke("hCAFEBABE".U)
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while(cycles < 300 && !done) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "Quad write should complete within 300 cycles")
    }
  }
  
  it should "enter and exit QPI mode" in {
    test(new PSRAM()) { dut =>
      // 进入 QPI 模式
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0x35.U)  // ENTER_QPI
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      while(cycles < 200) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          cycles = 200
        }
        cycles += 1
      }
      
      // 检查 QPI 模式标志
      dut.io.reg_addr.poke(0x14.U)  // CONFIG
      dut.clock.step(1)
      val config = dut.io.reg_rdata.peek().litValue
      assert((config & 1) != 0, "QPI mode should be enabled")
      
      // 退出 QPI 模式
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0xF5.U)  // EXIT_QPI
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      // 等待完成
      cycles = 0
      while(cycles < 200) {
        dut.io.reg_ren.poke(true.B)
        dut.io.reg_addr.poke(0x10.U)
        dut.clock.step(1)
        if((dut.io.reg_rdata.peek().litValue & 2) != 0) {
          cycles = 200
        }
        cycles += 1
      }
      
      // 检查 QPI 模式标志已清除
      dut.io.reg_addr.poke(0x14.U)
      dut.clock.step(1)
      val config2 = dut.io.reg_rdata.peek().litValue
      assert((config2 & 1) == 0, "QPI mode should be disabled")
    }
  }
  
  it should "verify quad mode output enables" in {
    test(new PSRAM()) { dut =>
      // Quad 读取: SIO2/3 应该是输入 (OE=0)
      dut.io.reg_wen.poke(true.B)
      dut.io.reg_addr.poke(0x00.U)
      dut.io.reg_wdata.poke(0xEB.U)  // QUAD_READ
      dut.clock.step(1)
      
      dut.io.reg_addr.poke(0x0C.U)
      dut.io.reg_wdata.poke(0x01.U)
      dut.clock.step(1)
      dut.io.reg_wen.poke(false.B)
      
      dut.clock.step(5)
      
      // 在数据阶段，SIO2/3 应该是输入
      dut.io.spi_sio2_oe.expect(false.B)
      dut.io.spi_sio3_oe.expect(false.B)
    }
  }
}
