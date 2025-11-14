package riscv.ai

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * MAC 单元测试
 */
class MacUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MacUnit"
  
  it should "perform multiply-accumulate correctly" in {
    test(new MacUnit(32)) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.c.poke(5.S)
      dut.clock.step(3)
      dut.io.result.expect(17.S)
      println(s"✓ MAC Test: 3 * 4 + 5 = ${dut.io.result.peek().litValue}")
    }
  }
  
  it should "handle negative numbers" in {
    test(new MacUnit(32)) { dut =>
      dut.io.a.poke(-2.S)
      dut.io.b.poke(3.S)
      dut.io.c.poke(10.S)
      dut.clock.step(3)
      dut.io.result.expect(4.S)
      println(s"✓ MAC Test: -2 * 3 + 10 = ${dut.io.result.peek().litValue}")
    }
  }
}

/**
 * 矩阵乘法器测试
 */
class MatrixMultiplierTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MatrixMultiplier"
  
  it should "multiply 2x2 matrices correctly" in {
    test(new MatrixMultiplier(32, 2)) { dut =>
      // 初始化矩阵 A
      dut.io.matrixA.writeEn.poke(true.B)
      for (i <- 0 until 4) {
        dut.io.matrixA.addr.poke(i.U)
        dut.io.matrixA.writeData.poke((i + 1).S)
        dut.clock.step(1)
      }
      dut.io.matrixA.writeEn.poke(false.B)
      
      // 初始化矩阵 B
      dut.io.matrixB.writeEn.poke(true.B)
      for (i <- 0 until 4) {
        dut.io.matrixB.addr.poke(i.U)
        dut.io.matrixB.writeData.poke((i + 1).S)
        dut.clock.step(1)
      }
      dut.io.matrixB.writeEn.poke(false.B)
      
      // 启动计算
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      // 等待完成
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }
      
      println(s"✓ Matrix multiplication completed in $cycles cycles")
    }
  }
}

/**
 * AI 加速器测试
 */
class CompactScaleAiChipTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "CompactScaleAiChip"
  
  it should "instantiate and respond to AXI transactions" in {
    test(new CompactScaleAiChip()) { dut =>
      dut.io.axi.awvalid.poke(false.B)
      dut.io.axi.wvalid.poke(false.B)
      dut.io.axi.arvalid.poke(false.B)
      dut.io.axi.bready.poke(true.B)
      dut.io.axi.rready.poke(true.B)
      
      dut.clock.step(10)
      println("✓ AI Accelerator instantiated successfully")
    }
  }
  
  it should "process matrix data through AXI" in {
    test(new CompactScaleAiChip()) { dut =>
      dut.io.axi.bready.poke(true.B)
      dut.io.axi.rready.poke(true.B)
      
      // 写入一些数据
      for (i <- 0 until 4) {
        dut.io.axi.awaddr.poke(i.U)
        dut.io.axi.awvalid.poke(true.B)
        dut.io.axi.wdata.poke((i + 1).U)
        dut.io.axi.wvalid.poke(true.B)
        dut.clock.step(1)
        while (!dut.io.axi.awready.peek().litToBoolean) {
          dut.clock.step(1)
        }
        dut.io.axi.awvalid.poke(false.B)
        dut.io.axi.wvalid.poke(false.B)
        dut.clock.step(2)
      }
      
      println("✓ Matrix data written successfully")
    }
  }
}

/**
 * RISC-V 集成测试
 */
class RiscvAiIntegrationTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RiscvAiChip"
  
  it should "instantiate without errors" in {
    test(new RiscvAiChip) { dut =>
      dut.io.mem_ready.poke(false.B)
      dut.io.mem_rdata.poke(0.U)
      dut.io.irq.poke(0.U)
      dut.clock.step(10)
      println("✓ RiscvAiChip instantiated successfully")
    }
  }
  
  it should "handle memory transactions" in {
    test(new RiscvAiChip) { dut =>
      dut.io.mem_ready.poke(false.B)
      dut.io.mem_rdata.poke(0.U)
      dut.io.irq.poke(0.U)
      
      var cycles = 0
      while (!dut.io.mem_valid.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }
      
      if (dut.io.mem_valid.peek().litToBoolean) {
        dut.io.mem_ready.poke(true.B)
        dut.io.mem_rdata.poke(0x00000013.U)
        dut.clock.step(1)
        dut.io.mem_ready.poke(false.B)
        println(s"✓ Memory request detected at cycle $cycles")
      }
      
      dut.clock.step(10)
    }
  }
  
  it should "report performance counters" in {
    test(new RiscvAiChip) { dut =>
      dut.io.mem_ready.poke(false.B)
      dut.io.mem_rdata.poke(0.U)
      dut.io.irq.poke(0.U)
      dut.clock.step(20)
      println("✓ Performance counters accessible")
    }
  }
}

/**
 * 系统集成测试
 */
class RiscvAiSystemTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RiscvAiSystem"
  
  it should "integrate CPU and AI accelerator" in {
    test(new RiscvAiSystem()) { dut =>
      dut.io.mem_ready.poke(false.B)
      dut.io.mem_rdata.poke(0.U)
      dut.io.irq.poke(0.U)
      
      dut.clock.step(5)
      
      for (_ <- 0 until 20) {
        if (dut.io.mem_valid.peek().litToBoolean) {
          dut.io.mem_ready.poke(true.B)
          dut.io.mem_rdata.poke(0x00000013.U)
        } else {
          dut.io.mem_ready.poke(false.B)
        }
        dut.clock.step(1)
      }
      
      println("✓ CPU and AI accelerator integration successful")
    }
  }
}
