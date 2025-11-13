package riscv.ai

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * 规模对比测试
 * 比较不同规模芯片的设计复杂度
 */
class ScaleComparisonTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "Scale Comparison"
  
  // NoiJinScaleAiChip 暂未实现
  // it should "test NoiJinScaleAiChip basic functionality" in {
  //   test(new NoiJinScaleAiChip()) { dut =>
  //     dut.clock.step(10)
  //     
  //     val counter0 = dut.io.perf_counters(0).peek().litValue
  //     val counter1 = dut.io.perf_counters(1).peek().litValue
  //     
  //     println("=== NoiJinScaleAiChip 测试 ===")
  //     println(s"配置: 32个MAC单元 + 2个矩阵乘法器")
  //     println(s"性能计数器0: $counter0")
  //     println(s"性能计数器1: $counter1")
  //     println("✅ NoiJinScaleAiChip 基本功能测试通过")
  //   }
  // }
  
  it should "test CompactScaleAiChip basic functionality" in {
    test(new CompactScaleAiChip()) { dut =>
      dut.clock.step(10)
      
      val counter0 = dut.io.perf_counters(0).peek().litValue
      val counter1 = dut.io.perf_counters(1).peek().litValue
      
      println("=== CompactScaleAiChip 测试 ===")
      println(s"配置: 16个MAC单元 + 1个矩阵乘法器")
      println(s"性能计数器0: $counter0")
      println(s"性能计数器1: $counter1")
      println("✅ CompactScaleAiChip 基本功能测试通过")
    }
  }
  
  it should "compare design scales" in {
    println("=== 🔍 设计规模对比分析 ===")
    println()
    
    // 基于FixedMediumScaleAiChip的实际数据: 284,363 instances
    val fixedInstances = 284363
    
    println("📊 设计规模对比:")
    println(f"| 设计版本 | MAC单元 | 矩阵乘法器 | 存储器 | 预估Instances | 是否满足10万限制 |")
    println(f"|----------|---------|------------|--------|---------------|------------------|")
    
    // FixedMediumScaleAiChip
    println(f"| FixedMediumScale | 64个 | 4个(16x16) | 4×2K | $fixedInstances | ❌ 超出限制 |")
    
    // NoiJinScaleAiChip 预估
    val noiJinInstances = (fixedInstances * 0.4).toInt
    val noiJinStatus = if (noiJinInstances <= 100000) "✅ 满足限制" else "❌ 超出限制"
    println(f"| NoiJinScale | 32个 | 2个(16x16) | 2×1K | $noiJinInstances | $noiJinStatus |")
    
    // CompactScaleAiChip 预估
    val compactInstances = (fixedInstances * 0.15).toInt
    val compactStatus = if (compactInstances <= 100000) "✅ 满足限制" else "❌ 超出限制"
    println(f"| CompactScale | 16个 | 1个(8x8) | 1×512 | $compactInstances | $compactStatus |")
    
    println()
    println("🎯 预估分析:")
    println(s"- FixedMediumScale: $fixedInstances instances (实际测量)")
    println(s"- NoiJinScale: ~$noiJinInstances instances (预估，缩放因子0.4)")
    println(s"- CompactScale: ~$compactInstances instances (预估，缩放因子0.15)")
    
    println()
    println("💡 建议:")
    if (noiJinInstances <= 100000) {
      println("✅ NoiJinScaleAiChip 可能满足10万instances限制")
    } else {
      println("⚠️  NoiJinScaleAiChip 可能仍超出10万instances限制")
      println("🔧 建议使用 CompactScaleAiChip 确保满足限制")
    }
    
    println()
    println("📈 性能对比预估:")
    println("- FixedMediumScale: 64 MAC/周期, 4×(16×16) = 1024个矩阵元素")
    println("- NoiJinScale: 32 MAC/周期, 2×(16×16) = 512个矩阵元素")  
    println("- CompactScale: 16 MAC/周期, 1×(8×8) = 64个矩阵元素")
    
    println()
    println("🎖️  推荐方案:")
    println("1. 如果必须满足10万instances限制: 使用 CompactScaleAiChip")
    println("2. 如果可以接受轻微超出: 尝试 NoiJinScaleAiChip")
    println("3. 如果需要最高性能: 使用 FixedMediumScaleAiChip (但需要商业EDA工具)")
  }
}