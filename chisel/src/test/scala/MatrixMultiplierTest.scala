package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MatrixMultiplierTest extends AnyFlatSpec with ChiselScalatestTester {
  
  "MacUnit" should "perform multiply-accumulate correctly" in {
    test(new MacUnit()) { dut =>
      // 测试 MAC: 3 * 4 + 5 = 17
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.c.poke(5.S)
      dut.clock.step(1)
      dut.io.result.expect(17.S)
    }
  }

  "MatrixMultiplier" should "perform matrix multiplication correctly" in {
    test(new MatrixMultiplier(matrixSize = 2)) { dut =>
      // 测试2x2矩阵乘法
      // A = [[1, 2], [3, 4]]
      // B = [[5, 6], [7, 8]]
      // C = [[19, 22], [43, 50]]
      
      // 写入矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      dut.io.matrixA.addr.poke(0.U) // A[0,0]
      dut.io.matrixA.writeData.poke(1.S)
      dut.clock.step(1)
      
      dut.io.matrixA.addr.poke(1.U) // A[0,1]
      dut.io.matrixA.writeData.poke(2.S)
      dut.clock.step(1)
      
      dut.io.matrixA.addr.poke(2.U) // A[1,0]
      dut.io.matrixA.writeData.poke(3.S)
      dut.clock.step(1)
      
      dut.io.matrixA.addr.poke(3.U) // A[1,1]
      dut.io.matrixA.writeData.poke(4.S)
      dut.clock.step(1)
      
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B
      dut.io.matrixB.writeEn.poke(true.B)
      dut.io.matrixB.addr.poke(0.U) // B[0,0]
      dut.io.matrixB.writeData.poke(5.S)
      dut.clock.step(1)
      
      dut.io.matrixB.addr.poke(1.U) // B[0,1]
      dut.io.matrixB.writeData.poke(6.S)
      dut.clock.step(1)
      
      dut.io.matrixB.addr.poke(2.U) // B[1,0]
      dut.io.matrixB.writeData.poke(7.S)
      dut.clock.step(1)
      
      dut.io.matrixB.addr.poke(3.U) // B[1,1]
      dut.io.matrixB.writeData.poke(8.S)
      dut.clock.step(1)
      
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 清零结果矩阵
      dut.io.result.writeEn.poke(true.B)
      for (i <- 0 until 4) {
        dut.io.result.addr.poke(i.U)
        dut.io.result.writeData.poke(0.S)
        dut.clock.step(1)
      }
      dut.io.result.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.busy.expect(true.B)
      dut.io.done.expect(false.B)
      
      // 等待计算完成，并打印调试信息
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
        
        // 每个周期打印状态信息
        if (cycles <= 10) {
          println(s"周期 $cycles: busy=${dut.io.busy.peek().litToBoolean}, done=${dut.io.done.peek().litToBoolean}")
        }
      }
      
      println(s"计算完成，用时 $cycles 个周期")
      dut.io.done.expect(true.B)
      
      // 验证结果
      dut.io.result.readEn.poke(true.B)
      
      dut.io.result.addr.poke(0.U) // C[0,0]
      dut.clock.step(1)
      dut.io.result.readData.expect(19.S)
      
      dut.io.result.addr.poke(1.U) // C[0,1]
      dut.clock.step(1)
      dut.io.result.readData.expect(22.S)
      
      dut.io.result.addr.poke(2.U) // C[1,0]
      dut.clock.step(1)
      dut.io.result.readData.expect(43.S)
      
      dut.io.result.addr.poke(3.U) // C[1,1]
      dut.clock.step(1)
      dut.io.result.readData.expect(50.S)
      
      println("✅ 2x2矩阵乘法测试通过")
    }
  }

  "RiscvAiChip" should "provide correct register interface" in {
    test(new RiscvAiChip()) { dut =>
      // 测试寄存器读写
      
      // 写入控制寄存器
      dut.io.ctrl.valid.poke(true.B)
      dut.io.ctrl.writeEn.poke(true.B)
      dut.io.ctrl.addr.poke(0x00.U)
      dut.io.ctrl.writeData.poke(0x12345678.U)
      dut.clock.step(1)
      
      // 读取控制寄存器
      dut.io.ctrl.writeEn.poke(false.B)
      dut.io.ctrl.addr.poke(0x00.U)
      dut.clock.step(1)
      dut.io.ctrl.readData.expect(0x12345678.U)
      
      // 测试矩阵A写入和读取
      dut.io.ctrl.writeEn.poke(true.B)
      dut.io.ctrl.addr.poke(0x10.U) // 矩阵A[0,0]
      dut.io.ctrl.writeData.poke(42.U)
      dut.clock.step(1)
      
      dut.io.ctrl.writeEn.poke(false.B)
      dut.io.ctrl.addr.poke(0x10.U)
      dut.clock.step(1)
      dut.io.ctrl.readData.expect(42.U)
      
      println("✅ 寄存器接口测试通过")
    }
  }

  "MatrixMultiplier" should "support 4x4 matrix multiplication with 10 random test cases" in {
    test(new MatrixMultiplier(matrixSize = 4)) { dut =>
      println("=== 4x4矩阵乘法随机测试开始 (10个测试用例) ===")
      
      val random = new scala.util.Random(42) // 固定种子确保可重现
      
      for (testCase <- 1 to 10) {
        println(s"\n--- 测试用例 $testCase ---")
        
        // 生成随机矩阵A和B
        val matrixA = Array.ofDim[Int](4, 4)
        val matrixB = Array.ofDim[Int](4, 4)
        
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            matrixA(i)(j) = random.nextInt(10) - 5 // -5到4的随机数
            matrixB(i)(j) = random.nextInt(10) - 5
          }
        }
        
        println("输入矩阵A:")
        for (i <- 0 until 4) {
          println(matrixA(i).mkString("[", ", ", "]"))
        }
        
        println("输入矩阵B:")
        for (i <- 0 until 4) {
          println(matrixB(i).mkString("[", ", ", "]"))
        }
        
        // 计算期望结果
        val expectedResult = Array.ofDim[Int](4, 4)
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            expectedResult(i)(j) = 0
            for (k <- 0 until 4) {
              expectedResult(i)(j) += matrixA(i)(k) * matrixB(k)(j)
            }
          }
        }
        
        println("期望结果矩阵:")
        for (i <- 0 until 4) {
          println(expectedResult(i).mkString("[", ", ", "]"))
        }
        
        // 写入矩阵A
        dut.io.matrixA.writeEn.poke(true.B)
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            val addr = i * 4 + j
            val value = matrixA(i)(j)
            dut.io.matrixA.addr.poke(addr.U)
            dut.io.matrixA.writeData.poke(value.S)
            dut.clock.step(1)
          }
        }
        dut.io.matrixA.writeEn.poke(false.B)
        
        // 写入矩阵B
        dut.io.matrixB.writeEn.poke(true.B)
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            val addr = i * 4 + j
            val value = matrixB(i)(j)
            dut.io.matrixB.addr.poke(addr.U)
            dut.io.matrixB.writeData.poke(value.S)
            dut.clock.step(1)
          }
        }
        dut.io.matrixB.writeEn.poke(false.B)
        
        // 清零结果矩阵
        dut.io.result.writeEn.poke(true.B)
        for (i <- 0 until 16) {
          dut.io.result.addr.poke(i.U)
          dut.io.result.writeData.poke(0.S)
          dut.clock.step(1)
        }
        dut.io.result.writeEn.poke(false.B)
        
        // 启动计算
        dut.io.start.poke(true.B)
        dut.clock.step(1)
        dut.io.start.poke(false.B)
        
        // 等待计算完成
        var cycles = 0
        while (!dut.io.done.peek().litToBoolean && cycles < 100) {
          dut.clock.step(1)
          cycles += 1
        }
        
        println(s"计算完成，用时 $cycles 个周期")
        
        // 读取结果矩阵
        val resultMatrix = Array.ofDim[Int](4, 4)
        dut.io.result.readEn.poke(true.B)
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            val addr = i * 4 + j
            dut.io.result.addr.poke(addr.U)
            dut.clock.step(1)
            resultMatrix(i)(j) = dut.io.result.readData.peek().litValue.toInt
          }
        }
        dut.io.result.readEn.poke(false.B)
        
        println("实际结果矩阵:")
        for (i <- 0 until 4) {
          println(resultMatrix(i).mkString("[", ", ", "]"))
        }
        
        // 验证结果
        var allCorrect = true
        for (i <- 0 until 4) {
          for (j <- 0 until 4) {
            if (resultMatrix(i)(j) != expectedResult(i)(j)) {
              println(s"错误: C[$i][$j] = ${resultMatrix(i)(j)}, 期望值 = ${expectedResult(i)(j)}")
              allCorrect = false
            }
          }
        }
        
        if (allCorrect) {
          println(s"测试用例 $testCase 通过!")
        } else {
          fail(s"测试用例 $testCase 失败!")
        }
      }
      
      println("\n=== 所有10个4x4矩阵乘法随机测试用例通过 ===")
    }
  }

  "MatrixMultiplier" should "support 8x8 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 8)) { dut =>
      // 测试8x8矩阵乘法
      // 使用单位矩阵和简单矩阵进行测试
      
      // 写入单位矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = i * 8 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B（简单递增序列）
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = i * 8 + j
          val value = addr + 1
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（8x8需要512个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 1000) {
        dut.clock.step(1)
        cycles += 1
      }
      
      println(s"8x8矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果（I * B = B）
      dut.io.result.readEn.poke(true.B)
      // 验证对角线和几个关键位置
      val testPositions = Seq((0,0), (1,1), (2,2), (7,7), (0,7), (7,0))
      for ((i, j) <- testPositions) {
        val addr = i * 8 + j
        val expectedValue = addr + 1
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 8x8矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 16x16 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 16)) { dut =>
      dut.clock.setTimeout(10000)
      // 测试16x16矩阵乘法
      // 由于规模较大，只测试部分关键功能
      
      // 写入单位矩阵A的对角线元素
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 16) {
        for (j <- 0 until 16) {
          val addr = i * 16 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B的部分元素（为了节省时间，只写入关键位置）
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 16) {
        for (j <- 0 until 16) {
          val addr = i * 16 + j
          val value = (i + 1) * 10 + (j + 1)  // 简单的值模式
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（16x16需要4096个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 5000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 1000 == 0) {
          println(s"16x16计算进行中... $cycles 周期")
        }
      }
      
      println(s"16x16矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (15,15), (0,15), (15,0))
      for ((i, j) <- testPositions) {
        val addr = i * 16 + j
        val expectedValue = (i + 1) * 10 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 16x16矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 32x32 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 32)) { dut =>
      dut.clock.setTimeout(50000)
      // 测试32x32矩阵乘法
      
      // 写入单位矩阵A（只写对角线元素以节省时间）
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 32) {
        for (j <- 0 until 32) {
          val addr = i * 32 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B（简单模式）
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 32) {
        for (j <- 0 until 32) {
          val addr = i * 32 + j
          val value = (i + 1) * 100 + (j + 1)
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（32x32需要32768个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 40000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 5000 == 0) {
          println(s"32x32计算进行中... $cycles 周期")
        }
      }
      
      println(s"32x32矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (31,31))
      for ((i, j) <- testPositions) {
        val addr = i * 32 + j
        val expectedValue = (i + 1) * 100 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 32x32矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 64x64 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 64)) { dut =>
      dut.clock.setTimeout(300000)
      // 测试64x64矩阵乘法
      
      // 写入单位矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 64) {
        for (j <- 0 until 64) {
          val addr = i * 64 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 64) {
        for (j <- 0 until 64) {
          val addr = i * 64 + j
          val value = (i + 1) * 1000 + (j + 1)
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（64x64需要262144个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 270000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 20000 == 0) {
          println(s"64x64计算进行中... $cycles 周期")
        }
      }
      
      println(s"64x64矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (63,63))
      for ((i, j) <- testPositions) {
        val addr = i * 64 + j
        val expectedValue = (i + 1) * 1000 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 64x64矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 128x128 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 128)) { dut =>
      dut.clock.setTimeout(2500000)
      // 测试128x128矩阵乘法
      
      // 写入单位矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 128) {
        for (j <- 0 until 128) {
          val addr = i * 128 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 128) {
        for (j <- 0 until 128) {
          val addr = i * 128 + j
          val value = (i + 1) * 10000 + (j + 1)
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（128x128需要2097152个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 2200000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 100000 == 0) {
          println(s"128x128计算进行中... $cycles 周期")
        }
      }
      
      println(s"128x128矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (127,127))
      for ((i, j) <- testPositions) {
        val addr = i * 128 + j
        val expectedValue = (i + 1) * 10000 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 128x128矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 256x256 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 256)) { dut =>
      dut.clock.setTimeout(20000000)
      // 测试256x256矩阵乘法
      
      // 写入单位矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 256) {
        for (j <- 0 until 256) {
          val addr = i * 256 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 256) {
        for (j <- 0 until 256) {
          val addr = i * 256 + j
          val value = (i + 1) * 100000 + (j + 1)
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（256x256需要16777216个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 17000000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 500000 == 0) {
          println(s"256x256计算进行中... $cycles 周期")
        }
      }
      
      println(s"256x256矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (255,255))
      for ((i, j) <- testPositions) {
        val addr = i * 256 + j
        val expectedValue = (i + 1) * 100000 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 256x256矩阵乘法测试通过")
    }
  }

  "MatrixMultiplier" should "support 512x512 matrix multiplication" in {
    test(new MatrixMultiplier(matrixSize = 512)) { dut =>
      dut.clock.setTimeout(150000000)
      // 测试512x512矩阵乘法
      
      // 写入单位矩阵A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 512) {
        for (j <- 0 until 512) {
          val addr = i * 512 + j
          val value = if (i == j) 1 else 0
          dut.io.matrixA.addr.poke(addr.U)
          dut.io.matrixA.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 写入矩阵B
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 512) {
        for (j <- 0 until 512) {
          val addr = i * 512 + j
          val value = (i + 1) * 1000000 + (j + 1)
          dut.io.matrixB.addr.poke(addr.U)
          dut.io.matrixB.writeData.poke(value.S)
          dut.clock.step(1)
        }
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待计算完成（512x512需要134217728个周期）
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 135000000) {
        dut.clock.step(1)
        cycles += 1
        if (cycles % 5000000 == 0) {
          println(s"512x512计算进行中... $cycles 周期")
        }
      }
      
      println(s"512x512矩阵乘法计算完成，用时 $cycles 个周期")
      
      // 验证几个关键结果
      dut.io.result.readEn.poke(true.B)
      val testPositions = Seq((0,0), (1,1), (511,511))
      for ((i, j) <- testPositions) {
        val addr = i * 512 + j
        val expectedValue = (i + 1) * 1000000 + (j + 1)
        dut.io.result.addr.poke(addr.U)
        dut.clock.step(1)
        dut.io.result.readData.expect(expectedValue.S)
      }
      dut.io.result.readEn.poke(false.B)
      
      println("✅ 512x512矩阵乘法测试通过")
    }
  }
}