package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SimpleCompactAccelDebugTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SimpleCompactAccel Debug"
  
  it should "debug 2x2 matrix multiply step by step" in {
    test(new SimpleCompactAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== 调试 2x2 矩阵乘法 ===")
      println("A = [[1, 2], [3, 4]]")
      println("B = [[5, 6], [7, 8]]")
      println("期望 C = [[19, 22], [43, 50]]")
      println()
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入矩阵 A
      println("写入矩阵 A:")
      val matrixA = Array(
        (0x100, 1), (0x104, 2),  // 第0行
        (0x120, 3), (0x124, 4)   // 第1行
      )
      
      for ((addr, value) <- matrixA) {
        val idx = (addr - 0x100) / 4
        println(f"  地址 0x$addr%03X (索引 $idx%2d) = $value")
        dut.io.reg.addr.poke(addr.U)
        dut.io.reg.wdata.poke(value.U)
        dut.clock.step(1)
      }
      
      // 写入矩阵 B
      println("\n写入矩阵 B:")
      val matrixB = Array(
        (0x300, 5), (0x304, 6),  // 第0行
        (0x320, 7), (0x324, 8)   // 第1行
      )
      
      for ((addr, value) <- matrixB) {
        val idx = (addr - 0x300) / 4
        println(f"  地址 0x$addr%03X (索引 $idx%2d) = $value")
        dut.io.reg.addr.poke(addr.U)
        dut.io.reg.wdata.poke(value.U)
        dut.clock.step(1)
      }
      
      // 设置矩阵大小
      println("\n设置矩阵大小 = 2")
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(2.U)
      dut.clock.step(1)
      
      // 启动计算
      println("\n启动计算...")
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      println("等待计算完成...")
      var cycles = 0
      var done = false
      while (!done && cycles < 100) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        val status = dut.io.reg.rdata.peek().litValue
        if (status == 2) {
          done = true
          println(s"✓ 计算完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      // 读取所有结果
      println("\n读取结果矩阵 C:")
      val resultAddrs = Array(
        (0x500, 0, 0, 19),  // C[0][0]
        (0x504, 0, 1, 22),  // C[0][1]
        (0x520, 1, 0, 43),  // C[1][0]
        (0x524, 1, 1, 50)   // C[1][1]
      )
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.ren.poke(true.B)
      dut.io.reg.wen.poke(false.B)
      
      var allCorrect = true
      for ((addr, row, col, expected) <- resultAddrs) {
        dut.io.reg.addr.poke(addr.U)
        dut.clock.step(1)
        val result = dut.io.reg.rdata.peek().litValue.toInt
        val idx = (addr - 0x500) / 4
        
        val status = if (result == expected) "✓" else "✗"
        println(f"  C[$row][$col] (地址 0x$addr%03X, 索引 $idx%2d) = $result%3d (期望 $expected%3d) $status")
        
        if (result != expected) {
          allCorrect = false
          
          // 计算期望值的详细过程
          println(f"    计算过程: C[$row][$col] = A[$row][0]*B[0][$col] + A[$row][1]*B[1][$col]")
          if (row == 0 && col == 0) {
            println(f"                        = 1*5 + 2*7 = 5 + 14 = 19")
          } else if (row == 0 && col == 1) {
            println(f"                        = 1*6 + 2*8 = 6 + 16 = 22")
          } else if (row == 1 && col == 0) {
            println(f"                        = 3*5 + 4*7 = 15 + 28 = 43")
          } else if (row == 1 && col == 1) {
            println(f"                        = 3*6 + 4*8 = 18 + 32 = 50")
          }
        }
      }
      
      // 读取性能计数器
      dut.io.reg.addr.poke(0x028.U)
      dut.clock.step(1)
      val perfCycles = dut.io.reg.rdata.peek().litValue
      println(f"\n性能计数器: $perfCycles 周期")
      
      if (allCorrect) {
        println("\n✓✓✓ 测试通过 ✓✓✓")
      } else {
        println("\n✗✗✗ 测试失败 ✗✗✗")
      }
    }
  }
}
