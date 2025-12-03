package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class PSRAMSoCTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "PSRAM SoC Integration"
  
  it should "compile SoC with PSRAM" in {
    test(new SimpleEdgeAiSoC()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // 基本初始化测试
      dut.clock.step(5)
      
      // 验证 PSRAM 信号初始状态
      // CS 应该是高电平 (未选中)
      dut.io.psram_spi_cs.expect(true.B)
      
      // 时钟应该是低电平 (未启动)
      dut.io.psram_spi_clk.expect(false.B)
      
      // Quad SPI 输出使能应该关闭
      dut.io.psram_spi_sio2_oe.expect(false.B)
      dut.io.psram_spi_sio3_oe.expect(false.B)
    }
  }
  
  it should "maintain PSRAM idle state" in {
    test(new SimpleEdgeAiSoC()) { dut =>
      dut.clock.step(10)
      
      // 在没有访问的情况下，PSRAM 应该保持空闲
      dut.io.psram_spi_cs.expect(true.B)
      dut.io.psram_spi_clk.expect(false.B)
      
      dut.clock.step(50)
      
      // 继续保持空闲
      dut.io.psram_spi_cs.expect(true.B)
      dut.io.psram_spi_clk.expect(false.B)
    }
  }
  
  it should "have correct memory map" in {
    test(new SimpleEdgeAiSoC()) { dut =>
      // 验证内存映射常量
      assert(SimpleMemoryMap.PSRAM_BASE == 0x04000000L)
      assert(SimpleMemoryMap.PSRAM_SIZE == 0x00800000L)  // 8 MB
      
      dut.clock.step(5)
    }
  }
}
