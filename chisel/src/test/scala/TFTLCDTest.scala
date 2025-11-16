// TFTLCDTest.scala - TFT LCD Controller Test - Simplified Version
// Phase 2 of DEV_PLAN_V0.2

package riscv.ai.peripherals

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TFTLCDTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "TFTLCD (Simplified)"
  
  it should "initialize correctly" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      // Check initial state
      dut.io.spi_cs.expect(true.B)  // CS should be high (inactive)
      dut.io.backlight.expect(false.B)
      dut.io.spi_rst.expect(false.B)
      
      // Read status
      dut.io.addr.poke(0x08.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      
      val status = dut.io.rdata.peek().litValue
      assert((status & 0x1) == 0, "Should not be busy initially")
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "enable backlight" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      // Enable backlight
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x01.U)  // BACKLIGHT = 1
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      dut.clock.step(5)
      dut.io.backlight.expect(true.B)
    }
  }
  
  it should "set reset pin" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      // Set reset high
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x02.U)  // RESET = 1
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      dut.clock.step(5)
      dut.io.spi_rst.expect(true.B)
    }
  }
  

  
  it should "send SPI command" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.clock.setTimeout(0)  // Disable timeout
      
      // Send a command
      val cmd = 0x2A  // Column address set
      dut.io.addr.poke(0x00.U)
      dut.io.wdata.poke(cmd.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for transmission
      var foundActivity = false
      for (_ <- 0 until 200) {
        if (!dut.io.spi_cs.peek().litToBoolean) {
          foundActivity = true
        }
        dut.clock.step(1)
      }
      
      assert(foundActivity, "Should see SPI activity (CS low)")
    }
  }
  
  it should "send SPI data" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      dut.clock.setTimeout(0)  // Disable timeout
      
      // Send data
      val data = 0x55
      dut.io.addr.poke(0x04.U)
      dut.io.wdata.poke(data.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for transmission
      var foundActivity = false
      var foundDCHigh = false
      for (_ <- 0 until 200) {
        if (!dut.io.spi_cs.peek().litToBoolean) {
          foundActivity = true
          if (dut.io.spi_dc.peek().litToBoolean) {
            foundDCHigh = true
          }
        }
        dut.clock.step(1)
      }
      
      assert(foundActivity, "Should see SPI activity")
      assert(foundDCHigh, "DC should be high for data")
    }
  }
}
