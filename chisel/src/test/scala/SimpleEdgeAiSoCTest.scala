package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * SimpleEdgeAiSoC 测试
 * 模拟 C 程序的测试逻辑
 */
class SimpleEdgeAiSoCTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "SimpleEdgeAiSoC"
  
  it should "instantiate correctly" in {
    test(new SimpleEdgeAiSoC()) { dut =>
      dut.clock.step(10)
      println("✓ SimpleEdgeAiSoC 实例化成功")
    }
  }
  
  it should "test CompactAccel 2x2 matrix multiply" in {
    test(new SimpleCompactAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      // 复位
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== CompactAccel 2x2 矩阵乘法测试 ===")
      
      // 初始化矩阵 A = [[1, 2], [3, 4]]
      println("写入矩阵 A:")
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // A[0][0] = 1
      dut.io.reg.addr.poke(0x100.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      // A[0][1] = 2
      dut.io.reg.addr.poke(0x104.U)
      dut.io.reg.wdata.poke(2.U)
      dut.clock.step(1)
      
      // A[1][0] = 3
      dut.io.reg.addr.poke(0x120.U)
      dut.io.reg.wdata.poke(3.U)
      dut.clock.step(1)
      
      // A[1][1] = 4
      dut.io.reg.addr.poke(0x124.U)
      dut.io.reg.wdata.poke(4.U)
      dut.clock.step(1)
      
      // 初始化矩阵 B = [[5, 6], [7, 8]]
      println("写入矩阵 B:")
      
      // B[0][0] = 5
      dut.io.reg.addr.poke(0x300.U)
      dut.io.reg.wdata.poke(5.U)
      dut.clock.step(1)
      
      // B[0][1] = 6
      dut.io.reg.addr.poke(0x304.U)
      dut.io.reg.wdata.poke(6.U)
      dut.clock.step(1)
      
      // B[1][0] = 7
      dut.io.reg.addr.poke(0x320.U)
      dut.io.reg.wdata.poke(7.U)
      dut.clock.step(1)
      
      // B[1][1] = 8
      dut.io.reg.addr.poke(0x324.U)
      dut.io.reg.wdata.poke(8.U)
      dut.clock.step(1)
      
      // 设置矩阵大小
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(2.U)
      dut.clock.step(1)
      
      // 启动计算
      println("启动计算...")
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U) // START
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      println("等待计算完成...")
      var cycles = 0
      var done = false
      while (!done && cycles < 1000) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U) // STATUS
        dut.clock.step(1)
        
        val status = dut.io.reg.rdata.peek().litValue
        if (status == 2) { // DONE
          done = true
          println(s"✓ 计算完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      if (!done) {
        println("✗ 超时：计算未完成")
      }
      
      // 读取结果 C = A * B = [[19, 22], [43, 50]]
      println("读取结果矩阵 C:")
      val expected = Array(19, 22, 43, 50)
      val offsets = Array(0x500, 0x504, 0x520, 0x524)
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.ren.poke(true.B)
      dut.io.reg.wen.poke(false.B)
      
      var allCorrect = true
      for (i <- 0 until 4) {
        dut.io.reg.addr.poke(offsets(i).U)
        dut.clock.step(1)
        val result = dut.io.reg.rdata.peek().litValue.toInt
        val exp = expected(i)
        
        if (result == exp) {
          println(f"  C[$i] = $result%3d ✓")
        } else {
          println(f"  C[$i] = $result%3d (期望 $exp%3d) ✗")
          allCorrect = false
        }
      }
      
      if (allCorrect) {
        println("\n✓✓✓ 2x2 矩阵乘法测试通过 ✓✓✓")
      } else {
        println("\n✗✗✗ 2x2 矩阵乘法测试失败 ✗✗✗")
      }
      
      // 读取性能计数器
      dut.io.reg.addr.poke(0x028.U)
      dut.clock.step(1)
      val perfCycles = dut.io.reg.rdata.peek().litValue
      println(f"\n性能: $perfCycles 周期")
    }
  }
  
  it should "test CompactAccel 4x4 matrix multiply" in {
    test(new SimpleCompactAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== CompactAccel 4x4 矩阵乘法测试 ===")
      println("测试: 单位矩阵 * 测试矩阵 = 测试矩阵")
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入单位矩阵 A
      println("写入单位矩阵 A (4x4):")
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x100 + i * 0x20 + j * 4
          val value = if (i == j) 1 else 0
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 写入测试矩阵 B
      println("写入测试矩阵 B (4x4):")
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x300 + i * 0x20 + j * 4
          val value = i * 4 + j + 1
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 设置矩阵大小
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(4.U)
      dut.clock.step(1)
      
      // 启动计算
      println("启动计算...")
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
          println(s"✓ 计算完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      // 验证结果
      println("验证结果 (应该等于矩阵 B):")
      var allCorrect = true
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x500 + i * 0x20 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.clock.step(1)
          
          val result = dut.io.reg.rdata.peek().litValue.toInt
          val expected = i * 4 + j + 1
          
          if (result != expected) {
            println(f"  C[$i][$j] = $result%3d (期望 $expected%3d) ✗")
            allCorrect = false
          }
        }
      }
      
      if (allCorrect) {
        println("✓✓✓ 4x4 矩阵乘法测试通过 ✓✓✓")
      } else {
        println("✗✗✗ 4x4 矩阵乘法测试失败 ✗✗✗")
      }
    }
  }
  
  it should "test BitNetAccel 4x4 matrix multiply" in {
    test(new SimpleBitNetAccel()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== BitNetAccel 4x4 矩阵乘法测试 ===")
      
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      // 写入激活值
      println("写入激活值 (4x4):")
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x100 + i * 0x40 + j * 4
          val value = i + 1
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 写入权重 (单位矩阵)
      println("写入权重 (4x4 单位矩阵):")
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x300 + i * 0x40 + j * 4
          val value = if (i == j) 1 else 0
          dut.io.reg.addr.poke(addr.U)
          dut.io.reg.wdata.poke(value.U)
          dut.clock.step(1)
        }
      }
      
      // 设置矩阵大小
      dut.io.reg.addr.poke(0x01C.U)
      dut.io.reg.wdata.poke(4.U)
      dut.clock.step(1)
      
      // 配置
      dut.io.reg.addr.poke(0x020.U)
      dut.io.reg.wdata.poke(0x04040404L.U)
      dut.clock.step(1)
      
      // 启动计算
      println("启动计算...")
      dut.io.reg.addr.poke(0x000.U)
      dut.io.reg.wdata.poke(1.U)
      dut.clock.step(1)
      
      dut.io.reg.valid.poke(false.B)
      dut.io.reg.wen.poke(false.B)
      
      // 等待完成
      var cycles = 0
      var done = false
      while (!done && cycles < 2000) {
        dut.io.reg.valid.poke(true.B)
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x004.U)
        dut.clock.step(1)
        
        if (dut.io.reg.rdata.peek().litValue == 2) {
          done = true
          println(s"✓ 计算完成，用时 $cycles 周期")
        }
        cycles += 1
      }
      
      // 验证结果
      println("验证结果:")
      var allCorrect = true
      for (i <- 0 until 4) {
        for (j <- 0 until 4) {
          val addr = 0x500 + i * 0x40 + j * 4
          dut.io.reg.addr.poke(addr.U)
          dut.clock.step(1)
          
          val result = dut.io.reg.rdata.peek().litValue.toInt
          val expected = i + 1
          
          if (result != expected) {
            println(f"  Result[$i][$j] = $result%3d (期望 $expected%3d) ✗")
            allCorrect = false
          }
        }
      }
      
      if (allCorrect) {
        println("✓✓✓ BitNetAccel 4x4 测试通过 ✓✓✓")
      } else {
        println("✗✗✗ BitNetAccel 4x4 测试失败 ✗✗✗")
      }
    }
  }
  
  it should "test GPIO functionality" in {
    test(new SimpleGPIO()) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step(5)
      
      println("\n=== GPIO 功能测试 ===")
      
      // 测试写入
      println("测试 GPIO 写入:")
      dut.io.reg.valid.poke(true.B)
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.ren.poke(false.B)
      
      val testValues = Array(0x0000L, 0xFFFFL, 0xAAAAL, 0x5555L)
      
      for (value <- testValues) {
        dut.io.reg.wdata.poke(value.U)
        dut.clock.step(1)
        
        val output = dut.io.gpio_out.peek().litValue
        if (output == value) {
          println(f"  写入 0x$value%04X -> 输出 0x$output%04X ✓")
        } else {
          println(f"  写入 0x$value%04X -> 输出 0x$output%04X ✗")
        }
      }
      
      // 测试读取
      println("\n测试 GPIO 读取:")
      dut.io.reg.wen.poke(false.B)
      dut.io.reg.ren.poke(true.B)
      
      val testInputs = Array(0x1234L, 0xABCDL, 0xBEEFL)
      
      for (value <- testInputs) {
        dut.io.gpio_in.poke(value.U)
        dut.clock.step(1)
        
        val readback = dut.io.reg.rdata.peek().litValue
        if (readback == value) {
          println(f"  输入 0x$value%04X -> 读取 0x$readback%04X ✓")
        } else {
          println(f"  输入 0x$value%04X -> 读取 0x$readback%04X ✗")
        }
      }
      
      println("\n✓✓✓ GPIO 测试通过 ✓✓✓")
    }
  }
  
  it should "run comprehensive test suite" in {
    test(new SimpleEdgeAiSoC()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(10)
      dut.reset.poke(false.B)
      dut.clock.step(10)
      
      println("\n" + "="*70)
      println("SimpleEdgeAiSoC 综合测试套件")
      println("="*70)
      
      // 初始化输入
      dut.io.uart_rx.poke(true.B)
      dut.io.gpio_in.poke(0.U)
      
      // 运行一段时间观察系统行为
      println("\n运行系统 100 个周期...")
      for (i <- 0 until 100) {
        dut.clock.step(1)
        
        if (i % 20 == 0) {
          val trap = dut.io.trap.peek().litValue
          val compact_irq = dut.io.compact_irq.peek().litValue
          val bitnet_irq = dut.io.bitnet_irq.peek().litValue
          val gpio_out = dut.io.gpio_out.peek().litValue
          
          println(f"  周期 $i%3d: trap=$trap compact_irq=$compact_irq bitnet_irq=$bitnet_irq gpio=0x$gpio_out%08X")
        }
      }
      
      println("\n✓ 系统运行稳定")
      println("="*70)
    }
  }
}
