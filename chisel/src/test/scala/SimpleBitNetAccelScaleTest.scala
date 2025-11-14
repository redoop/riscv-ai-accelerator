package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * SimpleBitNetAccel 规模测试
 * 测试从 2x2 到 16x16 的所有矩阵规模
 */
class SimpleBitNetAccelScaleTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SimpleBitNetAccel Scale Test"
  
  /**
   * 软件矩阵乘法（用于验证）
   */
  def matrixMultiply(a: Array[Array[Int]], b: Array[Array[Int]]): Array[Array[Int]] = {
    val n = a.length
    val result = Array.ofDim[Int](n, n)
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        var sum = 0
        for (k <- 0 until n) {
          sum += a(i)(k) * b(k)(j)
        }
        result(i)(j) = sum
      }
    }
    result
  }
  
  /**
   * 生成测试矩阵
   */
  def generateTestMatrix(size: Int, seed: Int): Array[Array[Int]] = {
    val matrix = Array.ofDim[Int](size, size)
    var value = seed
    for (i <- 0 until size) {
      for (j <- 0 until size) {
        matrix(i)(j) = (value % 10) + 1  // 1-10 的值
        value = (value * 7 + 13) % 100
      }
    }
    matrix
  }
  
  /**
   * 测试指定规模的矩阵乘法
   */
  def testMatrixSize(size: Int): Unit = {
    it should s"correctly compute ${size}x${size} matrix multiplication" in {
      test(new SimpleBitNetAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(5)
        
        println(s"\n=== BitNetAccel 测试 ${size}x${size} 矩阵乘法 ===")
        
        // 生成测试矩阵
        val activation = generateTestMatrix(size, 42)
        val weight = generateTestMatrix(size, 17)
        val expected = matrixMultiply(activation, weight)
        
        println(s"激活值矩阵 (${size}x${size}):")
        for (i <- 0 until math.min(size, 4)) {
          print("  ")
          for (j <- 0 until math.min(size, 4)) {
            print(f"${activation(i)(j)}%3d ")
          }
          if (size > 4) print("...")
          println()
        }
        if (size > 4) println("  ...")
        
        println(s"\n权重矩阵 (${size}x${size}):")
        for (i <- 0 until math.min(size, 4)) {
          print("  ")
          for (j <- 0 until math.min(size, 4)) {
            print(f"${weight(i)(j)}%3d ")
          }
          if (size > 4) print("...")
          println()
        }
        if (size > 4) println("  ...")
        
        // 写入激活值
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.wen.poke(true.B)
        dut.io.reg.ren.poke(false.B)
        
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            val addr = 0x100 + i * 0x40 + j * 4
            dut.io.reg.addr.poke(addr.U)
            dut.io.reg.wdata.poke(activation(i)(j).U)
            dut.clock.step(1)
          }
        }
        
        // 写入权重
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            val addr = 0x300 + i * 0x40 + j * 4
            dut.io.reg.addr.poke(addr.U)
            dut.io.reg.wdata.poke(weight(i)(j).U)
            dut.clock.step(1)
          }
        }
        
        // 设置矩阵大小
        dut.io.reg.addr.poke(0x01C.U)
        dut.io.reg.wdata.poke(size.U)
        dut.clock.step(1)
        
        // 配置
        dut.io.reg.addr.poke(0x020.U)
        dut.io.reg.wdata.poke((size << 24 | size << 16 | size << 8 | size).U)
        dut.clock.step(1)
        
        // 启动计算
        println(s"\n启动 ${size}x${size} 矩阵计算...")
        dut.io.reg.addr.poke(0x000.U)
        dut.io.reg.wdata.poke(1.U)
        dut.clock.step(1)
        
        dut.io.reg.valid.poke(false.B)
        dut.io.reg.wen.poke(false.B)
        
        // 等待完成
        var cycles = 0
        var done = false
        val maxCycles = size * size * size + 100
        
        while (!done && cycles < maxCycles) {
          dut.io.reg.valid.poke(true.B)
          dut.io.reg.ren.poke(true.B)
          dut.io.reg.addr.poke(0x004.U)
          dut.clock.step(1)
          
          if (dut.io.reg.rdata.peek().litValue == 2) {
            done = true
          }
          cycles += 1
        }
        
        if (!done) {
          println(s"✗ 超时：计算未在 $maxCycles 周期内完成")
          fail(s"Computation timeout for ${size}x${size} matrix")
        }
        
        println(s"✓ 计算完成，用时 $cycles 周期")
        
        // 读取性能计数器
        dut.io.reg.addr.poke(0x028.U)
        dut.clock.step(1)
        val perfCycles = dut.io.reg.rdata.peek().litValue
        println(s"性能计数器: $perfCycles 周期")
        
        // 理论周期数
        val theoreticalCycles = size * size * size
        println(s"理论周期数: $theoreticalCycles (${size}³)")
        
        // 验证结果
        println(s"\n验证结果矩阵 (${size}x${size}):")
        var errors = 0
        var checked = 0
        
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            val addr = 0x500 + i * 0x40 + j * 4
            dut.io.reg.addr.poke(addr.U)
            dut.clock.step(1)
            
            val result = dut.io.reg.rdata.peek().litValue.toInt
            val exp = expected(i)(j)
            
            if (result != exp) {
              if (errors < 10) {  // 只显示前10个错误
                println(f"  ✗ Result[$i][$j] = $result%5d (期望 $exp%5d)")
              }
              errors += 1
            }
            checked += 1
          }
        }
        
        if (errors == 0) {
          println(s"✓ 所有 $checked 个元素验证通过")
          println(s"\n✓✓✓ BitNetAccel ${size}x${size} 矩阵乘法测试通过 ✓✓✓")
        } else {
          println(s"\n✗ 发现 $errors 个错误（共 $checked 个元素）")
          fail(s"${size}x${size} matrix multiplication failed with $errors errors")
        }
        
        // 性能分析
        val throughput = (size * size * size).toDouble / perfCycles.toDouble
        println(f"\n性能分析:")
        println(f"  吞吐量: $throughput%.2f 次乘加/周期")
        println(f"  效率: ${(throughput * 100)}%.1f%%")
      }
    }
  }
  
  // 测试所有规模 (2x2 到 16x16)
  testMatrixSize(2)
  testMatrixSize(3)
  testMatrixSize(4)
  testMatrixSize(5)
  testMatrixSize(6)
  testMatrixSize(7)
  testMatrixSize(8)
  testMatrixSize(9)
  testMatrixSize(10)
  testMatrixSize(11)
  testMatrixSize(12)
  testMatrixSize(13)
  testMatrixSize(14)
  testMatrixSize(15)
  testMatrixSize(16)
  
  it should "handle maximum size (16x16) with identity matrix" in {
    test(new SimpleBitNetAccel()) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== BitNetAccel 测试最大规模 16x16 矩阵（单位矩阵）===")
      
      // 使用单位矩阵测试
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入单位矩阵作为激活值
      for (i <- 0 until 16) {
        for (j <- 0 until 16) {
          val addr = 0x100 + i * 0x40 + j * 4
          val value = if (i == j) 1 else 0
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 写入测试矩阵作为权重
      for (i <- 0 until 16) {
        for (j <- 0 until 16) {
          val addr = 0x300 + i * 0x40 + j * 4
          val value = i * 16 + j + 1
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 设置矩阵大小
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(16.U)
      dut.clock.step(1)
      
      // 配置
      dut.io.reg.addr.poke(0x020.U)
      dut.io.reg.wdata.poke(0x10101010L.U)
      dut.clock.step(1)
      
      // 启动计算
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while (!done && cycles < 5000) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        if (dut.io.reg.rdata.peek().litValue == 2) {
          done = true
        }
        cycles += 1
      }
      
      assert(done, "16x16 matrix computation should complete")
      println(s"✓ 16x16 矩阵计算完成，用时 $cycles 周期")
      
      // 验证结果（单位矩阵 * 权重 = 权重）
      var allCorrect = true
      for (i <- 0 until 16) {
        for (j <- 0 until 16) {
          val addr = 0x500 + i * 0x40 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.clock.step(1)
          
          val result = dut.io.reg.rdata.peek().litValue.toInt
          val expected = i * 16 + j + 1
          
          if (result != expected) {
            allCorrect = false
          }
        }
      }
      
      assert(allCorrect, "16x16 matrix result should be correct")
      println("✓ 所有 256 个元素验证通过")
    }
  }
  
  it should "report correct performance metrics for different sizes" in {
    test(new SimpleBitNetAccel()) { dut =>
      println("\n=== BitNetAccel 性能对比测试 ===")
      
      val sizes = Array(2, 4, 8, 12, 16)
      
      for (size <- sizes) {
        dut.reset.poke(true.B)
        dut.clock.step(5)
        dut.reset.poke(false.B)
        dut.clock.step(5)
        
        // 写入简单矩阵
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.wen.poke(true.B)
        dut.io.reg.ren.poke(false.B)
        
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            val addr = 0x100 + i * 0x40 + j * 4
            dut.io.reg.addr.poke(addr.U)
            dut.io.reg.wdata.poke(1.U)
            dut.clock.step(1)
          }
        }
        
        for (i <- 0 until size) {
          for (j <- 0 until size) {
            val addr = 0x300 + i * 0x40 + j * 4
            dut.io.reg.addr.poke(addr.U)
            dut.io.reg.wdata.poke(1.U)
            dut.clock.step(1)
          }
        }
        
        dut.io.reg.addr.poke(0x01C.U)
        dut.io.reg.wdata.poke(size.U)
        dut.clock.step(1)
        
        dut.io.reg.addr.poke(0x020.U)
        dut.io.reg.wdata.poke((size << 24 | size << 16 | size << 8 | size).U)
        dut.clock.step(1)
        
        dut.io.reg.addr.poke(0x000.U)
        dut.io.reg.wdata.poke(1.U)
        dut.clock.step(1)
        
        dut.io.reg.valid.poke(false.B)
        dut.io.reg.wen.poke(false.B)
        
        var cycles = 0
        var done = false
        val maxCycles = size * size * size + 100
        while (!done && cycles < maxCycles) {
          dut.io.reg.valid.poke(true.B)
          dut.io.reg.ren.poke(true.B)
          dut.io.reg.addr.poke(0x004.U)
          dut.clock.step(1)
          
          if (dut.io.reg.rdata.peek().litValue == 2) {
            done = true
          }
          cycles += 1
        }
        
        dut.io.reg.addr.poke(0x028.U)
        dut.clock.step(1)
        val perfCycles = dut.io.reg.rdata.peek().litValue
        
        val theoretical = size * size * size
        val throughput = theoretical.toDouble / perfCycles.toDouble
        println(f"${size}x${size}: $perfCycles%5d 周期 (理论: $theoretical%5d, 吞吐量: $throughput%.2f)")
      }
      
      println("\n✓ BitNetAccel 性能对比测试完成")
    }
  }
  
  it should "compare CompactAccel vs BitNetAccel performance" in {
    println("\n=== CompactAccel vs BitNetAccel 性能对比 ===")
    
    // 测试 CompactAccel (8x8)
    test(new SimpleCompactAccel()) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x100 + i * 0x20 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(1.U)
          dut.clock.step(1)
        }
      }
      
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x300 + i * 0x20 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(1.U)
          dut.clock.step(1)
        }
      }
      
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(8.U)
      dut.clock.step(1)
      
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      var cycles = 0
      var done = false
      while (!done && cycles < 1000) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        if (dut.io.reg.rdata.peek().litValue == 2) {
          done = true
        }
        cycles += 1
      }
      
      dut.io.reg.addr.poke(0x028.U)
      dut.clock.step(1)
      val compactCycles = dut.io.reg.rdata.peek().litValue
      
      println(f"CompactAccel 8x8: $compactCycles%4d 周期")
    }
    
    // 测试 BitNetAccel (8x8)
    test(new SimpleBitNetAccel()) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x100 + i * 0x40 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(1.U)
          dut.clock.step(1)
        }
      }
      
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x300 + i * 0x40 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(1.U)
          dut.clock.step(1)
        }
      }
      
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(8.U)
      dut.clock.step(1)
      
      dut.io.reg.addr.poke(0x020.U)
      dut.io.reg.wdata.poke(0x08080808L.U)
      dut.clock.step(1)
      
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      var cycles = 0
      var done = false
      while (!done && cycles < 1000) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        if (dut.io.reg.rdata.peek().litValue == 2) {
          done = true
        }
        cycles += 1
      }
      
      dut.io.reg.addr.poke(0x028.U)
      dut.clock.step(1)
      val bitnetCycles = dut.io.reg.rdata.peek().litValue
      
      println(f"BitNetAccel  8x8: $bitnetCycles%4d 周期")
    }
    
    println("\n✓ 性能对比完成")
  }
}
