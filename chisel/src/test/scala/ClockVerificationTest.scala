package riscv.ai

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import riscv.ai.peripherals.TFTLCD

/**
 * 时钟验证测试套件 - 简化版
 * 
 * 验证内容:
 * 1. SPI 时钟频率
 * 2. SPI 时钟占空比
 */
class ClockVerificationTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SPI Clock"
  
  it should "generate correct SPI clock frequency from 100MHz main clock" in {
    test(new TFTLCD(clockFreq = 100000000, spiFreq = 10000000))
      .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      
      dut.clock.setTimeout(0)
      
      println("=" * 60)
      println("测试: SPI 时钟频率验证")
      println("=" * 60)
      println(s"主时钟频率: 100 MHz")
      println(s"目标 SPI 频率: 10 MHz")
      println()
      
      // 测量 SPI 时钟周期
      var spiClkCycles = 0
      var mainClkCycles = 0
      var lastSpiClk = false
      val maxCycles = 1000
      
      // 等待 SPI 时钟翻转
      while (spiClkCycles < 20 && mainClkCycles < maxCycles) {
        val currentSpiClk = dut.io.spi_clk.peek().litToBoolean
        
        if (currentSpiClk && !lastSpiClk) {
          if (spiClkCycles < 5) {
            println(f"  SPI 时钟上升沿 #$spiClkCycles: 主时钟周期 $mainClkCycles")
          }
          spiClkCycles += 1
        }
        
        lastSpiClk = currentSpiClk
        mainClkCycles += 1
        dut.clock.step(1)
      }
      
      // 计算平均周期
      val avgPeriod = mainClkCycles.toDouble / spiClkCycles
      val frequency = 100000000.0 / avgPeriod
      
      println()
      println(f"测量结果:")
      println(f"  SPI 时钟周期数: $spiClkCycles")
      println(f"  主时钟周期数: $mainClkCycles")
      println(f"  平均 SPI 周期: $avgPeriod%.2f 个主时钟周期")
      println(f"  测量频率: ${frequency/1000000}%.3f MHz")
      println(f"  误差: ${(frequency - 10000000)/10000000 * 100}%.2f%%")
      
      // 验证频率在合理范围内 (9.5-10.5 MHz, 允许 ±5% 误差)
      assert(frequency >= 9500000 && frequency <= 10500000, 
        f"SPI 频率 ${frequency/1000000}%.3f MHz 超出范围 [9.5, 10.5] MHz")
      
      println()
      println("✓ 频率测试通过")
      println()
    }
  }
  
  it should "have approximately 50%% duty cycle" in {
    test(new TFTLCD(clockFreq = 100000000, spiFreq = 10000000)) { dut =>
      
      dut.clock.setTimeout(0)
      
      println("=" * 60)
      println("测试: SPI 时钟占空比验证")
      println("=" * 60)
      
      var highCycles = 0
      var lowCycles = 0
      val maxCycles = 2000
      
      // 跳过初始化阶段
      for (_ <- 0 until 100) {
        dut.clock.step(1)
      }
      
      // 测量占空比
      for (_ <- 0 until maxCycles) {
        if (dut.io.spi_clk.peek().litToBoolean) {
          highCycles += 1
        } else {
          lowCycles += 1
        }
        dut.clock.step(1)
      }
      
      val totalCycles = highCycles + lowCycles
      val dutyCycle = highCycles.toDouble / totalCycles * 100
      val deviation = Math.abs(dutyCycle - 50.0)
      
      println(f"占空比测量:")
      println(f"  高电平周期: $highCycles")
      println(f"  低电平周期: $lowCycles")
      println(f"  总周期: $totalCycles")
      println(f"  占空比: $dutyCycle%.2f%%")
      println(f"  偏差: $deviation%.2f%%")
      
      // 验证占空比在 45-55% 范围内
      assert(dutyCycle >= 45 && dutyCycle <= 55,
        f"占空比 $dutyCycle%.2f%% 超出范围 [45%%, 55%%]")
      
      println()
      println("✓ 占空比测试通过")
      println()
    }
  }
}
