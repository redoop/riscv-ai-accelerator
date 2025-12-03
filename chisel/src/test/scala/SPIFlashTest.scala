package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import riscv.ai.peripherals.SPIFlash

class SPIFlashTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SPIFlash"
  
  it should "initialize correctly" in {
    test(new SPIFlash()) { dut =>
      dut.io.spi_cs.expect(false.B)
      dut.io.reg.ready.expect(true.B)
      println("✓ SPIFlash 初始化正常")
    }
  }
  
  it should "perform READ command" in {
    test(new SPIFlash()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      println("\n=== SPI Flash READ 测试 ===")
      
      // 写入命令
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.addr.poke(0x00.U)
      dut.io.reg.wdata.poke(0x03.U)  // READ command
      dut.clock.step(1)
      
      // 写入地址
      dut.io.reg.addr.poke(0x04.U)
      dut.io.reg.wdata.poke(0x123456.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.reg.addr.poke(0x0C.U)
      dut.io.reg.wdata.poke(0x01.U)  // start
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待操作完成
      var cycles = 0
      var done = false
      while (!done && cycles < 500) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x0C.U)
        dut.clock.step(1)
        
        val ctrl = dut.io.reg.rdata.peek().litValue
        if ((ctrl & 4) != 0) {  // done bit
          done = true
          println(s"✓ READ 操作完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      assert(done, "READ 操作超时")
      
      // 验证 CS 信号已释放
      dut.io.spi_cs.expect(false.B)
      println("✓ CS 信号正确释放")
    }
  }
  
  it should "perform FAST_READ command with dummy cycles" in {
    test(new SPIFlash()) { dut =>
      println("\n=== SPI Flash FAST_READ 测试 ===")
      
      // 写入命令
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.addr.poke(0x00.U)
      dut.io.reg.wdata.poke(0x0B.U)  // FAST_READ command
      dut.clock.step(1)
      
      // 写入地址
      dut.io.reg.addr.poke(0x04.U)
      dut.io.reg.wdata.poke(0xABCDEF.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.reg.addr.poke(0x0C.U)
      dut.io.reg.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while (!done && cycles < 500) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x0C.U)
        dut.clock.step(1)
        
        if ((dut.io.reg.rdata.peek().litValue & 4) != 0) {
          done = true
          println(s"✓ FAST_READ 操作完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      assert(done, "FAST_READ 操作超时")
      println("✓ FAST_READ 测试通过 (包含 dummy cycles)")
    }
  }
  
  it should "perform PAGE_PROGRAM command" in {
    test(new SPIFlash()) { dut =>
      println("\n=== SPI Flash PAGE_PROGRAM 测试 ===")
      
      // 写入命令
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.addr.poke(0x00.U)
      dut.io.reg.wdata.poke(0x02.U)  // PAGE_PROGRAM command
      dut.clock.step(1)
      
      // 写入地址
      dut.io.reg.addr.poke(0x04.U)
      dut.io.reg.wdata.poke(0x100000.U)
      dut.clock.step(1)
      
      // 写入数据
      dut.io.reg.addr.poke(0x08.U)
      dut.io.reg.wdata.poke(0x12345678.U)
      dut.clock.step(1)
      
      // 启动操作
      dut.io.reg.addr.poke(0x0C.U)
      dut.io.reg.wdata.poke(0x01.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while (!done && cycles < 500) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x0C.U)
        dut.clock.step(1)
        
        if ((dut.io.reg.rdata.peek().litValue & 4) != 0) {
          done = true
          println(s"✓ PAGE_PROGRAM 操作完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      assert(done, "PAGE_PROGRAM 操作超时")
      println("✓ PAGE_PROGRAM 测试通过")
    }
  }
  
  it should "handle multiple operations sequentially" in {
    test(new SPIFlash()) { dut =>
      println("\n=== SPI Flash 连续操作测试 ===")
      
      for (i <- 0 until 3) {
        println(s"\n操作 ${i + 1}:")
        
        // 写入命令和地址
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.wen.poke(true.B)
        dut.io.reg.addr.poke(0x00.U)
        dut.io.reg.wdata.poke(0x03.U)
        dut.clock.step(1)
        
        dut.io.reg.addr.poke(0x04.U)
        dut.io.reg.wdata.poke((0x100000 + i * 0x1000).U)
        dut.clock.step(1)
        
        // 启动
        dut.io.reg.addr.poke(0x0C.U)
        dut.io.reg.wdata.poke(0x01.U)
        dut.clock.step(1)
        
        dut.io.reg.valid.poke(false.B)
        dut.io.reg.wen.poke(false.B)
        
        // 等待完成
        var cycles = 0
        var done = false
        while (!done && cycles < 500) {
          dut.io.reg.valid.poke(true.B)
          dut.io.reg.ren.poke(true.B)
          dut.io.reg.addr.poke(0x0C.U)
          dut.clock.step(1)
          
          if ((dut.io.reg.rdata.peek().litValue & 4) != 0) {
            done = true
          }
          cycles += 1
        }
        
        assert(done, s"操作 ${i + 1} 超时")
        println(s"  ✓ 操作 ${i + 1} 完成")
      }
      
      println("\n✓ 连续操作测试通过")
    }
  }
}
