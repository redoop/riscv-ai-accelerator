package riscv.ai

import circt.stage.ChiselStage

/**
 * 生成 EdgeAiSoC (RISC-V + AI 加速器 SoC) 的 Verilog 代码
 * 
 * EdgeAiSoC 特点：
 * - PicoRV32 RISC-V 核心 (RV32I)
 * - CompactScale AI 加速器 (8x8 矩阵)
 * - BitNetScale AI 加速器 (16x16 矩阵)
 * - DMA 控制器
 * - 中断控制器
 * - UART/GPIO 外设
 * - AXI4-Lite 系统总线
 */
object EdgeAiSoCMain extends App {
  println("=" * 70)
  println("Generating EdgeAiSoC (RISC-V + AI Accelerator SoC) Verilog...")
  println("=" * 70)
  
  println("\n📋 Configuration:")
  println("  - RISC-V Core: PicoRV32 (RV32I)")
  println("  - CompactScale: 8x8 matrix accelerator")
  println("  - BitNetScale: 16x16 matrix accelerator")
  println("  - System Bus: AXI4-Lite")
  println("  - Peripherals: UART, GPIO")
  println("  - DMA: Memory-to-memory transfer")
  println("  - Interrupts: 32 sources")
  
  println("\n🗺️  Memory Map:")
  println(f"  RAM:         0x${MemoryMap.RAM_BASE}%08X - 0x${MemoryMap.RAM_BASE + MemoryMap.RAM_SIZE - 1}%08X (256 MB)")
  println(f"  CompactScale: 0x${MemoryMap.COMPACT_BASE}%08X - 0x${MemoryMap.COMPACT_BASE + MemoryMap.COMPACT_SIZE - 1}%08X (4 KB)")
  println(f"  BitNetScale:  0x${MemoryMap.BITNET_BASE}%08X - 0x${MemoryMap.BITNET_BASE + MemoryMap.BITNET_SIZE - 1}%08X (4 KB)")
  println(f"  DMA:          0x${MemoryMap.DMA_BASE}%08X - 0x${MemoryMap.DMA_BASE + MemoryMap.DMA_SIZE - 1}%08X (4 KB)")
  println(f"  IntCtrl:      0x${MemoryMap.INTC_BASE}%08X - 0x${MemoryMap.INTC_BASE + MemoryMap.INTC_SIZE - 1}%08X (4 KB)")
  println(f"  UART:         0x${MemoryMap.UART_BASE}%08X - 0x${MemoryMap.UART_BASE + MemoryMap.UART_SIZE - 1}%08X (64 KB)")
  println(f"  GPIO:         0x${MemoryMap.GPIO_BASE}%08X - 0x${MemoryMap.GPIO_BASE + MemoryMap.GPIO_SIZE - 1}%08X (64 KB)")
  println(f"  Flash:        0x${MemoryMap.FLASH_BASE}%08X - 0x${MemoryMap.FLASH_BASE + MemoryMap.FLASH_SIZE - 1}%08X (256 MB)")
  
  println("\n⚡ Interrupt Map:")
  println("  IRQ 16: CompactScale computation done")
  println("  IRQ 17: BitNetScale computation done")
  println("  IRQ 18: DMA transfer done")
  
  println("\n🔧 Generating SystemVerilog...")
  
  try {
    ChiselStage.emitSystemVerilogFile(
      new EdgeAiSoC(),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
      args = Array("--target-dir", "generated/edgeaisoc")
    )
    
    // 后处理: 清理生成的文件
    println("\n📝 Post-processing generated files...")
    PostProcessVerilog.cleanupVerilogFile("generated/edgeaisoc/EdgeAiSoC.sv")
    
    println("\n" + "=" * 70)
    println("✅ EdgeAiSoC Verilog generation complete!")
    println("=" * 70)
    println("\n📁 Output Files:")
    println("  Directory: generated/edgeaisoc/")
    println("  Main file: EdgeAiSoC.sv")
    println("  + PicoRV32 core (from resources/rtl/picorv32.v)")
    println("  + CompactScaleWrapper.sv")
    println("  + BitNetScaleWrapper.sv")
    println("  + MemoryInterfaceAdapter.sv")
    println("  + EdgeDMAController.sv")
    println("  + EdgeInterruptController.sv")
    println("  + UARTController.sv")
    println("  + GPIOController.sv")
    
    println("\n💡 Next Steps:")
    println("  1. Review generated SystemVerilog files")
    println("  2. Integrate with PicoRV32 core (picorv32.v)")
    println("  3. Add memory controller for RAM/Flash")
    println("  4. Synthesize for FPGA or ASIC")
    println("  5. Develop software drivers")
    
    println("\n📚 Documentation:")
    println("  - README: chisel/docs/EdgeAiSoC_README.md")
    println("  - Build Guide: chisel/docs/EdgeAiSoC_BUILD.md")
    println("  - Status: chisel/docs/EdgeAiSoC_STATUS.md")
    println("  - Integration Plan: chisel/docs/RISCV_INTEGRATION_PLAN.md")
    
    println("\n🎯 Performance Targets:")
    println("  - CPU: 100 MHz, 100 MIPS")
    println("  - CompactScale: 1.6 GOPS @ 100MHz")
    println("  - BitNetScale: 4.8 GOPS @ 100MHz")
    println("  - Power: <200 mW (estimated)")
    println("  - Area: ~5 mm² @ 40nm (estimated)")
    
    println("\n🚀 Application Scenarios:")
    println("  - Edge AI inference")
    println("  - IoT smart devices")
    println("  - BitNet-1B/3B model inference")
    println("  - Image classification")
    println("  - Natural language processing")
    
  } catch {
    case e: Exception =>
      println("\n❌ Error during generation:")
      println(s"  ${e.getMessage}")
      println("\n⚠️  Known Issues:")
      println("  - AXI4-Lite interface direction issues with Flipped bundles")
      println("  - Address decoder needs proper crossbar implementation")
      println("  - See chisel/docs/EdgeAiSoC_STATUS.md for details")
      println("\n💡 Workarounds:")
      println("  1. Use simplified point-to-point connections")
      println("  2. Implement custom address decoder without Flipped")
      println("  3. Use DecoupledIO instead of custom AXI bundles")
      println("  4. Reference working designs in generated/ directory")
      throw e
  }
}

// SimpleEdgeAiSoCMain 已移至独立文件: SimpleEdgeAiSoCMain.scala
