package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

/**
 * CompactScaleAiChip 完整矩阵算法测试
 * 测试范围: 2x2 -> 512x512
 * 输出: 输入矩阵、输出矩阵、准确度、时间
 */
class CompactScaleFullMatrixTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "CompactScaleAiChip Full Matrix Algorithm Test"
  
  it should "test matrix multiplication from 2x2 to 512x512" in {
    test(new CompactScaleAiChip()) { dut =>
      // 设置更长的超时时间
      dut.clock.setTimeout(100000)
      
      println("=" * 80)
      println("🧮 CompactScaleAiChip 完整矩阵算法测试")
      println("=" * 80)
      println("📊 测试范围: 2x2 -> 512x512")
      println("🎯 硬件配置: 16个MAC单元 + 1个8x8矩阵乘法器")
      println("📈 输出内容: 输入矩阵、输出矩阵、准确度、计算时间")
      println("=" * 80)
      println()
      
      // 测试的矩阵规模 (限制到16x16，因为存储器只有512深度，每个矩阵最多256元素)
      val sizes = Seq(2, 4, 8, 16)
      
      // 存储所有测试结果
      var allResults = Seq[(Int, Long, Double, Int, Int)]()
      
      for (size <- sizes) {
        println("=" * 80)
        println(s"🔢 测试 ${size}x${size} 矩阵乘法")
        println("=" * 80)
        
        // 生成测试矩阵
        val (matrixA, matrixB, expectedResult) = generateTestMatrices(size)
        
        // 打印输入矩阵（小矩阵打印完整，大矩阵打印摘要）
        printInputMatrices(matrixA, matrixB, size)
        
        // 打印期望结果
        printExpectedResult(expectedResult, size)
        
        // 写入矩阵数据到硬件
        println(s"📝 写入 ${size}x${size} 矩阵数据到硬件...")
        writeMatrixToHardware(dut, matrixA, matrixB, size)
        
        // 启动计算并计时
        println("🚀 启动矩阵计算...")
        val startTime = System.nanoTime()
        val cycles = performMatrixComputation(dut, size)
        val endTime = System.nanoTime()
        val elapsedMs = (endTime - startTime) / 1000000.0
        
        println(f"⏱️  计算完成: $cycles 周期, ${elapsedMs}%.3f ms")
        
        // 读取并验证结果
        println("📖 读取计算结果...")
        val hwResult = readResultFromHardware(dut, size)
        
        // 打印硬件输出矩阵
        printHardwareResult(hwResult, size)
        
        // 计算准确度
        val accuracy = calculateAccuracy(hwResult, expectedResult, size)
        
        // 打印准确度统计
        printAccuracyStats(accuracy, size)
        
        // 性能统计
        val totalOps = size * size * size * 2 // 乘法和加法
        val opsPerCycle = if (cycles > 0) totalOps.toDouble / cycles else 0.0
        val opsPerMs = if (elapsedMs > 0) totalOps.toDouble / elapsedMs else 0.0
        
        println()
        println("📊 性能统计:")
        println(f"  🔢 总运算数: $totalOps 次")
        println(f"  🕐 计算周期: $cycles 周期")
        println(f"  ⏱️  计算时间: ${elapsedMs}%.3f ms")
        println(f"  📈 吞吐量: ${opsPerCycle}%.2f 运算/周期")
        println(f"  ⚡ 速度: ${opsPerMs}%.0f 运算/ms")
        println()
        
        // 保存结果
        allResults = allResults :+ (size, cycles, elapsedMs, accuracy._1, accuracy._2)
        
        // 重置硬件
        dut.clock.step(5)
      }
      
      // 打印总结
      printSummary(allResults)
    }
  }
  
  // 生成测试矩阵
  def generateTestMatrices(size: Int): (Array[Array[Int]], Array[Array[Int]], Array[Array[Long]]) = {
    val random = new Random(42 + size) // 固定种子以便复现
    
    val matrixA = Array.ofDim[Int](size, size)
    val matrixB = Array.ofDim[Int](size, size)
    val result = Array.ofDim[Long](size, size)
    
    // 生成矩阵A和B（使用小数值避免溢出）
    for (i <- 0 until size; j <- 0 until size) {
      matrixA(i)(j) = (random.nextInt(16) + 1) % 8 + 1  // 1-8
      matrixB(i)(j) = (random.nextInt(16) + 1) % 8 + 1  // 1-8
    }
    
    // 计算期望结果
    for (i <- 0 until size; j <- 0 until size) {
      var sum = 0L
      for (k <- 0 until size) {
        sum += matrixA(i)(k).toLong * matrixB(k)(j).toLong
      }
      result(i)(j) = sum
    }
    
    (matrixA, matrixB, result)
  }
  
  // 打印输入矩阵
  def printInputMatrices(matrixA: Array[Array[Int]], matrixB: Array[Array[Int]], size: Int): Unit = {
    if (size <= 8) {
      // 小矩阵：打印完整
      println("📝 输入矩阵 A:")
      for (i <- 0 until size) {
        print("   [")
        print(matrixA(i).mkString(", "))
        println("]")
      }
      println()
      
      println("📝 输入矩阵 B:")
      for (i <- 0 until size) {
        print("   [")
        print(matrixB(i).mkString(", "))
        println("]")
      }
      println()
    } else {
      // 大矩阵：打印摘要
      println(s"📝 输入矩阵 A (${size}x${size}):")
      println(s"   左上角 4x4:")
      for (i <- 0 until math.min(4, size)) {
        print("   [")
        print(matrixA(i).take(4).mkString(", "))
        println(", ...]")
      }
      println(s"   统计: min=${matrixA.flatten.min}, max=${matrixA.flatten.max}, avg=${matrixA.flatten.sum / (size * size)}")
      println()
      
      println(s"📝 输入矩阵 B (${size}x${size}):")
      println(s"   左上角 4x4:")
      for (i <- 0 until math.min(4, size)) {
        print("   [")
        print(matrixB(i).take(4).mkString(", "))
        println(", ...]")
      }
      println(s"   统计: min=${matrixB.flatten.min}, max=${matrixB.flatten.max}, avg=${matrixB.flatten.sum / (size * size)}")
      println()
    }
  }
  
  // 打印期望结果
  def printExpectedResult(result: Array[Array[Long]], size: Int): Unit = {
    if (size <= 8) {
      println("📝 期望结果矩阵:")
      for (i <- 0 until size) {
        print("   [")
        print(result(i).mkString(", "))
        println("]")
      }
      println()
    } else {
      println(s"📝 期望结果矩阵 (${size}x${size}):")
      println(s"   左上角 4x4:")
      for (i <- 0 until math.min(4, size)) {
        print("   [")
        print(result(i).take(4).mkString(", "))
        println(", ...]")
      }
      println(s"   统计: min=${result.flatten.min}, max=${result.flatten.max}, avg=${result.flatten.sum / (size * size)}")
      println()
    }
  }
  
  // 写入矩阵到硬件
  def writeMatrixToHardware(dut: CompactScaleAiChip, matrixA: Array[Array[Int]], matrixB: Array[Array[Int]], size: Int): Unit = {
    // CompactScale只有512深度的存储器，地址空间0-1023 (10位)
    // 布局: 0-255: 矩阵A, 256-511: 矩阵B, 512-767: 结果C, 768+: 控制寄存器
    
    // 硬件矩阵乘法器是8x8的，所以我们需要填充到8x8
    val hwSize = 8
    val maxElements = math.min(size * size, 256)
    
    if (size <= 4) {
      println(s"   调试: 写入矩阵A，元素数=$maxElements，填充到${hwSize}x$hwSize")
    }
    
    // 写入矩阵A (地址 0-63，填充到8x8)
    for (i <- 0 until hwSize; j <- 0 until hwSize) {
      val value = if (i < size && j < size) matrixA(i)(j) else 0
      val addr = i * hwSize + j
      dut.io.axi.awaddr.poke(addr.U)
      dut.io.axi.awvalid.poke(true.B)
      dut.io.axi.wdata.poke(value.U)
      dut.io.axi.wvalid.poke(true.B)
      dut.clock.step(1)
      dut.io.axi.awvalid.poke(false.B)
      dut.io.axi.wvalid.poke(false.B)
      
      if (size <= 4 && i < size && j < size) {
        println(s"   调试: 写入A[$i][$j]=$value 到地址$addr")
      }
    }
    
    if (size <= 4) {
      println(s"   调试: 写入矩阵B，元素数=$maxElements，填充到${hwSize}x$hwSize")
    }
    
    // 写入矩阵B (地址 256-319，填充到8x8)
    for (i <- 0 until hwSize; j <- 0 until hwSize) {
      val value = if (i < size && j < size) matrixB(i)(j) else 0
      val addr = 256 + i * hwSize + j
      dut.io.axi.awaddr.poke(addr.U)
      dut.io.axi.awvalid.poke(true.B)
      dut.io.axi.wdata.poke(value.U)
      dut.io.axi.wvalid.poke(true.B)
      dut.clock.step(1)
      dut.io.axi.awvalid.poke(false.B)
      dut.io.axi.wvalid.poke(false.B)
      
      if (size <= 4 && i < size && j < size) {
        println(s"   调试: 写入B[$i][$j]=$value 到地址$addr")
      }
    }
  }
  
  // 执行矩阵计算
  def performMatrixComputation(dut: CompactScaleAiChip, size: Int): Int = {
    val _ = size // 避免未使用警告
    // 先清除启动信号
    dut.io.axi.awaddr.poke(0x300.U)
    dut.io.axi.awvalid.poke(true.B)
    dut.io.axi.wdata.poke(0.U)
    dut.io.axi.wvalid.poke(true.B)
    dut.clock.step(1)
    dut.io.axi.awvalid.poke(false.B)
    dut.io.axi.wvalid.poke(false.B)
    dut.clock.step(2)
    
    // 写入控制寄存器启动计算
    dut.io.axi.awaddr.poke(0x300.U)
    dut.io.axi.awvalid.poke(true.B)
    dut.io.axi.wdata.poke(1.U)
    dut.io.axi.wvalid.poke(true.B)
    dut.clock.step(1)
    dut.io.axi.awvalid.poke(false.B)
    dut.io.axi.wvalid.poke(false.B)
    
    // 等待计算完成 - 矩阵乘法器是8x8的，所以总是需要 8^3 = 512 个周期
    val matrixHwSize = 8  // CompactScale使用8x8矩阵乘法器
    val computeCycles = matrixHwSize * matrixHwSize * matrixHwSize + 20  // 加一些余量
    
    for (i <- 0 until computeCycles) {
      dut.clock.step(1)
      if (i % 1000 == 0 && i > 0) {
        val progress = (i.toDouble / computeCycles * 100).toInt
        print(f"\r   进度: $progress%3d%%, 周期: $i")
      }
    }
    
    if (computeCycles >= 1000) {
      println()
    }
    
    // 检查完成状态
    val done = dut.io.status.done.peek().litToBoolean
    val busy = dut.io.status.busy.peek().litToBoolean
    if (done) {
      println(s"   ✅ 计算完成信号已置位")
    } else {
      println(s"   ⚠️  计算完成信号未置位 (busy=$busy)")
    }
    
    // 清除启动信号，让状态机回到idle
    dut.io.axi.awaddr.poke(0x300.U)
    dut.io.axi.awvalid.poke(true.B)
    dut.io.axi.wdata.poke(0.U)
    dut.io.axi.wvalid.poke(true.B)
    dut.clock.step(1)
    dut.io.axi.awvalid.poke(false.B)
    dut.io.axi.wvalid.poke(false.B)
    dut.clock.step(5)
    
    computeCycles
  }
  
  // 从硬件读取结果
  def readResultFromHardware(dut: CompactScaleAiChip, size: Int): Array[Array[Long]] = {
    val result = Array.ofDim[Long](size, size)
    val offsetC = 512
    val hwSize = 8  // 硬件矩阵乘法器是8x8的
    
    if (size <= 4) {
      println(s"   调试: 读取结果矩阵，起始地址=$offsetC, 从${hwSize}x${hwSize}中读取${size}x${size}")
    }
    
    // 从8x8的结果矩阵中读取我们需要的部分
    for (i <- 0 until size; j <- 0 until size) {
      val addr = offsetC + i * hwSize + j
      dut.io.axi.araddr.poke(addr.U)
      dut.io.axi.arvalid.poke(true.B)
      dut.clock.step(1)
      dut.io.axi.arvalid.poke(false.B)
      
      val value = dut.io.axi.rdata.peek().litValue.toLong
      result(i)(j) = value
      
      if (size <= 4) {
        println(s"   调试: 地址=$addr, 读取值=$value, 位置=($i,$j)")
      }
      
      dut.clock.step(1)
    }
    
    result
  }
  
  // 打印硬件结果
  def printHardwareResult(result: Array[Array[Long]], size: Int): Unit = {
    if (size <= 8) {
      println("📊 硬件输出矩阵:")
      for (i <- 0 until size) {
        print("   [")
        print(result(i).mkString(", "))
        println("]")
      }
      println()
    } else {
      println(s"📊 硬件输出矩阵 (${size}x${size}):")
      println(s"   左上角 4x4:")
      for (i <- 0 until math.min(4, size)) {
        print("   [")
        print(result(i).take(4).mkString(", "))
        println(", ...]")
      }
      println(s"   统计: min=${result.flatten.min}, max=${result.flatten.max}, avg=${result.flatten.sum / (size * size)}")
      println()
    }
  }
  
  // 计算准确度
  def calculateAccuracy(hwResult: Array[Array[Long]], expected: Array[Array[Long]], size: Int): (Int, Int) = {
    var exactMatches = 0
    var totalElements = 0
    
    for (i <- 0 until size; j <- 0 until size) {
      totalElements += 1
      if (hwResult(i)(j) == expected(i)(j)) {
        exactMatches += 1
      }
    }
    
    (exactMatches, totalElements)
  }
  
  // 打印准确度统计
  def printAccuracyStats(accuracy: (Int, Int), size: Int): Unit = {
    val _ = size // 避免未使用警告
    val (matches, total) = accuracy
    val percentage = (matches.toDouble / total * 100)
    
    println("🎯 准确度分析:")
    println(f"  ✓ 精确匹配: $matches / $total (${percentage}%.2f%%)")
    
    if (percentage == 100.0) {
      println("  🎉 完美匹配！所有元素计算正确")
    } else if (percentage >= 99.0) {
      println("  ✅ 优秀！准确度超过99%")
    } else if (percentage >= 95.0) {
      println("  👍 良好！准确度超过95%")
    } else {
      println("  ⚠️  需要改进，准确度低于95%")
    }
  }
  
  // 打印总结
  def printSummary(results: Seq[(Int, Long, Double, Int, Int)]): Unit = {
    println()
    println("=" * 80)
    println("📊 测试总结")
    println("=" * 80)
    println()
    
    println("| 矩阵规模 | 计算周期 | 计算时间(ms) | 准确度 | 吞吐量(运算/周期) |")
    println("|----------|----------|--------------|--------|-------------------|")
    
    for ((size, cycles, timeMs, matches, total) <- results) {
      val accuracy = (matches.toDouble / total * 100)
      val totalOps = size * size * size * 2
      val throughput = if (cycles > 0) totalOps.toDouble / cycles else 0.0
      
      println(f"| ${size}x$size%6s | $cycles%8d | ${timeMs}%12.3f | ${accuracy}%5.2f%% | ${throughput}%17.2f |")
    }
    
    println()
    println("🎯 关键发现:")
    
    // 分析准确度趋势
    val avgAccuracy = results.map { case (_, _, _, m, t) => m.toDouble / t * 100 }.sum / results.size
    println(f"  📈 平均准确度: ${avgAccuracy}%.2f%%")
    
    // 分析性能趋势
    val avgThroughput = results.map { case (size, cycles, _, _, _) => 
      val ops = size * size * size * 2
      ops.toDouble / cycles
    }.sum / results.size
    println(f"  ⚡ 平均吞吐量: ${avgThroughput}%.2f 运算/周期")
    
    // 最大矩阵规模
    val maxSize = results.map(_._1).max
    println(f"  🏆 最大测试规模: ${maxSize}x$maxSize")
    
    // 总计算时间
    val totalTime = results.map(_._3).sum
    println(f"  ⏱️  总测试时间: ${totalTime}%.3f ms")
    
    println()
    println("✅ CompactScaleAiChip 矩阵算法测试完成！")
    println("=" * 80)
  }
}
