// TFTLCDTest.scala - TFT LCD Controller Test
// Phase 2 of DEV_PLAN_V0.2

package riscv.ai.peripherals

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TFTLCDTest extends AnyFlatSpec with ChiselScalatestTester {
  
  behavior of "TFTLCD"
  
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
  
  it should "configure display window" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      // Set X start
      dut.io.addr.poke(0x10.U)
      dut.io.wdata.poke(10.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Set Y start
      dut.io.addr.poke(0x14.U)
      dut.io.wdata.poke(20.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Set X end
      dut.io.addr.poke(0x18.U)
      dut.io.wdata.poke(100.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Set Y end
      dut.io.addr.poke(0x1C.U)
      dut.io.wdata.poke(110.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Read back values
      dut.io.addr.poke(0x10.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(10.U)
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
      
      dut.io.addr.poke(0x14.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(20.U)
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "write to framebuffer" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      // Write RGB565 color to framebuffer
      val color = 0xF800  // Red in RGB565
      val fbAddr = 0x1000 + (10 * 128 + 20) * 2  // Pixel at (20, 10)
      
      dut.io.addr.poke(fbAddr.U)
      dut.io.wdata.poke(color.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Read back
      dut.io.addr.poke(fbAddr.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.rdata.expect(color.U)
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
  
  it should "send SPI command" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.clock.setTimeout(0)  // Disable timeout
      
      // Enable reset first
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x02.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for init to complete
      for (_ <- 0 until 5000) {
        dut.clock.step(1)
      }
      
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
      for (_ <- 0 until 500) {
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
      
      // Enable reset first
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x02.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for init to complete
      for (_ <- 0 until 5000) {
        dut.clock.step(1)
      }
      
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
      for (_ <- 0 until 500) {
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
  
  it should "initialize display on reset" in {
    test(new TFTLCD(clockFreq = 1000000, spiFreq = 100000)) { dut =>
      dut.clock.setTimeout(0)  // Disable timeout
      
      // Enable reset to trigger initialization
      dut.io.addr.poke(0x0C.U)
      dut.io.wdata.poke(0x02.U)
      dut.io.wen.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      dut.io.valid.poke(false.B)
      dut.io.wen.poke(false.B)
      
      // Wait for initialization (reduced cycles for faster test)
      for (_ <- 0 until 5000) {
        dut.clock.step(1)
      }
      
      // Check if init is done
      dut.io.addr.poke(0x08.U)
      dut.io.ren.poke(true.B)
      dut.io.valid.poke(true.B)
      dut.clock.step(1)
      val status = dut.io.rdata.peek().litValue
      assert((status & 0x2) != 0, "Init should be done")
      
      dut.io.valid.poke(false.B)
      dut.io.ren.poke(false.B)
    }
  }
}
