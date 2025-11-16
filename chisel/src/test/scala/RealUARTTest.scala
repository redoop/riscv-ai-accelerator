// RealUARTTest.scala - UART Controller Test
// Phase 1 of DEV_PLAN_V0.2

package riscv.ai.peripherals

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RealUARTTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "RealUART"
  
  it should "initialize correctly" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200)) { dut =>
      dut.io.tx.expect(true.B)  // TX idle high
      dut.io.ready.expect(true.B)
      
      // Read status register
      dut.io.addr.poke(0x04.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      
      // Should show TX not busy, RX FIFO empty
      val status = dut.io.rdata.peek().litValue
      assert((status & 0x8) != 0, "RX FIFO should be empty")
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "configure baud rate" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200)) { dut =>
      val newBaudDiv = 434  // For 115200 baud at 50MHz
      
      // Write baud divisor
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(newBaudDiv.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Read back baud divisor
      dut.io.addr.poke(0x0C.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(newBaudDiv.U)
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "enable TX and RX" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200)) { dut =>
      // Enable TX and RX
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x03.U)  // TX_ENABLE | RX_ENABLE
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Read back control register
      dut.io.addr.poke(0x08.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(0x03.U)
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "transmit a byte" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200, fifoDepth = 16)) { dut =>
      // Enable TX
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x01.U)  // TX_ENABLE
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Write byte to TX FIFO
      val testByte = 0x55  // 01010101
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(testByte.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for transmission to start
      dut.clock.step(10)
      
      // TX should go low (start bit)
      var foundStartBit = false
      for (_ <- 0 until 20) {
        if (!dut.io.tx.peek().litToBoolean) {
          foundStartBit = true
        }
        dut.clock.step(1)
      }
      assert(foundStartBit, "Should detect start bit")
      
      // Wait for transmission to complete
      dut.clock.step(100)
      
      // TX should return to idle (high)
      dut.io.tx.expect(true.B)
    }
  }
  
  it should "fill TX FIFO" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200, fifoDepth = 4)) { dut =>
      // Enable TX
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x01.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Fill FIFO
      for (i <- 0 until 4) {
        dut.io.addr.poke(0x00.U)
        dut.io.wdata.poke((0x30 + i).U)
        dut.io.wen.poke(true.B)
        dut.io.valid.poke(true.B)
        dut.clock.step(1)
        dut.io.valid.poke(false.B)
        dut.io.wen.poke(false.B)
      }
      
      // Check status - FIFO should be full
      dut.io.addr.poke(0x04.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      val status = dut.io.rdata.peek().litValue
      assert((status & 0x4) != 0, "TX FIFO should be full")
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "receive a byte" ignore {
    test(new RealUART(clockFreq = 100000, baudRate = 9600, fifoDepth = 16)) { dut =>
      // Enable RX
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x02.U)  // RX_ENABLE
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      val baudDiv = 100000 / 9600
      val testByte = 0xA5  // 10100101
      
      // Start with RX idle (high)
      dut.io.rx.poke(true.B)
      dut.clock.step(5)
      
      // Simulate receiving a byte
      // Start bit (low)
      dut.io.rx.poke(false.B)
      dut.clock.step(baudDiv + 5)  // Wait for start bit detection + extra cycles
      
      // Data bits (LSB first)
      for (i <- 0 until 8) {
        val bit = ((testByte >> i) & 1) != 0
        dut.io.rx.poke(bit.B)
        dut.clock.step(baudDiv)
      }
      
      // Stop bit (high)
      dut.io.rx.poke(true.B)
      dut.clock.step(baudDiv + 10)  // Extra cycles for processing
      
      // Check if byte was received
      dut.io.addr.poke(0x04.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      val status = dut.io.rdata.peek().litValue
      assert((status & 0x2) != 0, "RX should have data ready")
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
      
      // Read received byte
      dut.io.addr.poke(0x00.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(testByte.U)
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "generate TX interrupt" in {
    test(new RealUART(clockFreq = 1000000, baudRate = 115200)) { dut =>
      // Enable TX and TX interrupt
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x05.U)  // TX_ENABLE | TX_IRQ_ENABLE
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      dut.clock.step(5)
      
      // TX FIFO is not full, should generate interrupt
      dut.io.tx_irq.expect(true.B)
    }
  }
  
  it should "generate RX interrupt" in {
    test(new RealUART(clockFreq = 100000, baudRate = 9600)) { dut =>
      // Enable RX and RX interrupt
      dut.io.addr.poke(0x08.U)
      dut.io.wdata.poke(0x0A.U)  // RX_ENABLE | RX_IRQ_ENABLE
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      val baudDiv = 100000 / 9600
      val testByte = 0x42
      
      // Simulate receiving a byte
      dut.io.rx.poke(false.B)  // Start bit
      dut.clock.step(baudDiv * 2)
      
      for (i <- 0 until 8) {
        val bit = ((testByte >> i) & 1) != 0
        dut.io.rx.poke(bit.B)
        dut.clock.step(baudDiv)
      }
      
      dut.io.rx.poke(true.B)  // Stop bit
      dut.clock.step(baudDiv * 2)
      
      // Should generate RX interrupt
      dut.io.rx_irq.expect(true.B)
    }
  }
}
