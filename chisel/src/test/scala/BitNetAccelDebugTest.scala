package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class BitNetAccelDebugTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "BitNetAccel Debug"
  
  it should "test 8x8 with BitNet weights" in {
    test(new SimpleBitNetAccel()) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== BitNet 8x8 矩阵乘法测试 ===")
      println("激活值 = 单位矩阵 (对角线为1，其余为0)")
      println("权重   = BitNet 模式 (交替 +1/-1/0)")
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入单位矩阵作为激活值
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x100 + i * 0x40 + j * 4
          val value = if (i == j) 1 else 0
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 写入 BitNet 权重矩阵 (交替模式: +1, -1, 0)
      val expectedResults = Array.ofDim[Int](8, 8)
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x300 + i * 0x40 + j * 4
          // BitNet 权重模式：根据位置交替使用 +1, -1, 0
          val value = (i + j) % 3 match {
            case 0 => 1   // +1
            case 1 => -1  // -1
            case 2 => 0   // 0
          }
          expectedResults(i)(j) = value
          dut.io.reg.addr.poke(addr.U)
          // 将负数转换为 32 位补码表示
          val unsignedValue: BigInt = if (value < 0) {
            BigInt(0x100000000L) + BigInt(value)
          } else {
            BigInt(value)
          }
          dut.io.reg.wdata.poke(unsignedValue.U)
          dut.clock.step(1)
        }
      }
      
      // 设置矩阵大小
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(8.U)
      dut.clock.step(1)
      
      // 配置
      dut.io.reg.addr.poke(0x020.U)
      dut.io.reg.wdata.poke(0x08080808L.U)
      dut.clock.step(1)
      
      // 启动计算
      println("\n启动计算...")
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
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
      
      println(s"✓ 计算完成，用时 $cycles 周期")
      
      // 读取稀疏性统计
      dut.io.reg.addr.poke(0x02C.U)
      dut.clock.step(1)
      val sparsitySkipped = dut.io.reg.rdata.peek().litValue
      println(s"稀疏性优化: 跳过了 $sparsitySkipped 次零权重计算")
      
      // 额外等待几个周期
      dut.io.reg.valid.poke(false.B)
      dut.clock.step(5)
      
      // 验证结果（单位矩阵 * 权重 = 权重的对角线）
      println("\n验证结果:")
      var allCorrect = true
      var errorCount = 0
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          val addr = 0x500 + i * 0x40 + j * 4
          dut.io.reg.valid.poke(true.B)
          dut.io.reg.ren.poke(true.B)
          dut.io.reg.addr.poke(addr.U)
          dut.clock.step(1)
          
          val result = dut.io.reg.rdata.peek().litValue.toInt
          // 处理负数
          val signedResult = if (result > 0x7FFFFFFF) {
            (result - 0x100000000L).toInt
          } else {
            result
          }
          
          // 单位矩阵 * 权重 = 权重矩阵本身
          val expected = expectedResults(i)(j)
          
          if (signedResult != expected) {
            if (errorCount < 10) {
              println(f"  ✗ [$i][$j] 地址=0x$addr%03X 结果=$signedResult%3d (期望 $expected%3d)")
            }
            errorCount += 1
            allCorrect = false
          }
        }
      }
      
      if (allCorrect) {
        println("✓ 所有 64 个元素验证通过")
        println("\n✓✓✓ BitNet 8x8 测试通过 ✓✓✓")
      } else {
        println(f"\n✗✗✗ BitNet 8x8 测试失败（$errorCount 个错误）✗✗✗")
      }
    }
  }
  
  it should "test 2x2 with BitNet weights" in {
    test(new SimpleBitNetAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== BitNet 2x2 矩阵乘法测试 ===")
      println("激活值 = [[1, 2], [3, 4]]")
      println("权重   = [[1, -1], [1, 0]] (BitNet: {-1, 0, +1})")
      println("期望   = [[3, -1], [7, -3]]")
      println("计算过程:")
      println("  [0][0] = 1*1 + 2*1 = 1 + 2 = 3")
      println("  [0][1] = 1*(-1) + 2*0 = -1 + 0 = -1")
      println("  [1][0] = 3*1 + 4*1 = 3 + 4 = 7")
      println("  [1][1] = 3*(-1) + 4*0 = -3 + 0 = -3")
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入激活值 [[1, 2], [3, 4]]
      println("\n写入激活值:")
      val activationData = Array(
        (0x100, 0, 0, 1),
        (0x104, 0, 1, 2),
        (0x140, 1, 0, 3),
        (0x144, 1, 1, 4)
      )
      
      for ((addr, i, j, value) <- activationData) {
        println(f"  地址 0x$addr%03X [${i}][${j}] = $value")
        dut.io.reg.addr.poke(addr.U)
        dut.io.reg.wdata.poke(value.U)
        dut.clock.step(1)
      }
      
      // 写入 BitNet 权重 [[1, -1], [1, 0]]
      println("\n写入 BitNet 权重 (编码: 0=0, 1=+1, -1=-1):")
      val weightData = Array(
        (0x300, 0, 0, 1),   // [0][0] = +1
        (0x304, 0, 1, -1),  // [0][1] = -1
        (0x340, 1, 0, 1),   // [1][0] = +1
        (0x344, 1, 1, 0)    // [1][1] = 0
      )
      
      for ((addr, i, j, value) <- weightData) {
        println(f"  地址 0x$addr%03X [${i}][${j}] = $value%2d")
        dut.io.reg.addr.poke(addr.U)
        // 将负数转换为 32 位补码表示
        val unsignedValue: BigInt = if (value < 0) {
          BigInt(0x100000000L) + BigInt(value)
        } else {
          BigInt(value)
        }
        dut.io.reg.wdata.poke(unsignedValue.U)
        dut.clock.step(1)
      }
      
      // 设置矩阵大小
      println("\n设置矩阵大小 = 2")
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(2.U)
      dut.clock.step(1)
      
      // 配置
      dut.io.reg.addr.poke(0x020.U)
      dut.io.reg.wdata.poke(0x02020202L.U)
      dut.clock.step(1)
      
      // 启动计算
      println("\n启动计算...")
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while (!done && cycles < 100) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        if (dut.io.reg.rdata.peek().litValue == 2) {
          done = true
        }
        cycles += 1
      }
      
      println(s"✓ 计算完成，用时 $cycles 周期")
      
      // 读取稀疏性统计
      dut.io.reg.addr.poke(0x02C.U)
      dut.clock.step(1)
      val sparsitySkipped = dut.io.reg.rdata.peek().litValue
      println(s"稀疏性优化: 跳过了 $sparsitySkipped 次零权重计算")
      
      // 读取结果
      println("\n读取结果:")
      val resultData = Array(
        (0x500, 0, 0, 3),   // [0][0] = 1*1 + 2*1 = 3
        (0x504, 0, 1, -1),  // [0][1] = 1*(-1) + 2*0 = -1
        (0x540, 1, 0, 7),   // [1][0] = 3*1 + 4*1 = 7
        (0x544, 1, 1, -3)   // [1][1] = 3*(-1) + 4*0 = -3
      )
      
      var allCorrect = true
      for ((addr, i, j, expected) <- resultData) {
        dut.io.reg.addr.poke(addr.U)
        dut.clock.step(1)
        val result = dut.io.reg.rdata.peek().litValue.toInt
        // 处理负数（32位补码）
        val signedResult = if (result > 0x7FFFFFFF) {
          (result - 0x100000000L).toInt
        } else {
          result
        }
        
        val status = if (signedResult == expected) "✓" else "✗"
        println(f"  地址 0x$addr%03X [${i}][${j}] = $signedResult%3d (期望 $expected%3d) $status")
        
        if (signedResult != expected) {
          allCorrect = false
        }
      }
      
      if (allCorrect) {
        println("\n✓✓✓ BitNet 2x2 测试通过 ✓✓✓")
      } else {
        println("\n✗✗✗ BitNet 2x2 测试失败 ✗✗✗")
      }
    }
  }
}
