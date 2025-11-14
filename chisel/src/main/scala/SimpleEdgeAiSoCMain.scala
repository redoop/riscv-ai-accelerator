package riscv.ai

import circt.stage.ChiselStage

/**
 * 生成简化版 EdgeAiSoC 的 Verilog 代码
 * 
 * 使用简单寄存器接口替代 AXI4-Lite，避免接口方向问题
 * 
 * 特点：
 * - PicoRV32 RISC-V 核心 (RV32I)
 * - SimpleCompactAccel (8x8 矩阵加速器)
 * - SimpleBitNetAccel (16x16 矩阵加速器)
 * - 简单寄存器接口 (替代 AXI4-Lite)
 * - UART/GPIO 外设
 * - 地址解码器
 */
object SimpleEdgeAiSoCMain extends App {
  println("=" * 70)
  println("Generating Simple EdgeAiSoC (RISC-V + AI Accelerator SoC)")
  println("=" * 70)
  
  println("\n📋 Configuration:")
  println("  - RISC-V Core: PicoRV32 (RV32I)")
  println("  - CompactAccel: 8x8 matrix accelerator")
  println("  - BitNetAccel: 16x16 matrix accelerator")
  println("  - Interface: Simple Register (not AXI4-Lite)")
  println("  - Peripherals: UART, GPIO")
  println("  - Interrupts: 2 sources (compact + bitnet)")
  
  println("\n🗺️  Memory Map:")
  println(f"  RAM:         0x${SimpleMemoryMap.RAM_BASE}%08X - 0x${SimpleMemoryMap.RAM_BASE + SimpleMemoryMap.RAM_SIZE - 1}%08X (256 MB)")
  println(f"  CompactAccel: 0x${SimpleMemoryMap.COMPACT_BASE}%08X - 0x${SimpleMemoryMap.COMPACT_BASE + SimpleMemoryMap.COMPACT_SIZE - 1}%08X (4 KB)")
  println(f"  BitNetAccel:  0x${SimpleMemoryMap.BITNET_BASE}%08X - 0x${SimpleMemoryMap.BITNET_BASE + SimpleMemoryMap.BITNET_SIZE - 1}%08X (4 KB)")
  println(f"  UART:         0x${SimpleMemoryMap.UART_BASE}%08X - 0x${SimpleMemoryMap.UART_BASE + SimpleMemoryMap.UART_SIZE - 1}%08X (64 KB)")
  println(f"  GPIO:         0x${SimpleMemoryMap.GPIO_BASE}%08X - 0x${SimpleMemoryMap.GPIO_BASE + SimpleMemoryMap.GPIO_SIZE - 1}%08X (64 KB)")
  
  println("\n⚡ Interrupt Map:")
  println("  IRQ 16: CompactAccel computation done")
  println("  IRQ 17: BitNetAccel computation done")
  
  println("\n💡 Key Improvements:")
  println("  ✅ Simple register interface (no AXI complexity)")
  println("  ✅ No Flipped bundle direction issues")
  println("  ✅ Straightforward address decoding")
  println("  ✅ Easy to understand and modify")
  println("  ✅ Suitable for FPGA prototyping")
  
  println("\n🔧 Generating SystemVerilog...")
  
  try {
    ChiselStage.emitSystemVerilogFile(
      new SimpleEdgeAiSoC(),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
      args = Array("--target-dir", "generated/simple_edgeaisoc")
    )
    
    // 后处理
    println("\n📝 Post-processing generated files...")
    PostProcessVerilog.cleanupVerilogFile("generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv")
    
    println("\n" + "=" * 70)
    println("✅ Simple EdgeAiSoC Verilog generation complete!")
    println("=" * 70)
    
    println("\n📁 Output Files:")
    println("  Directory: generated/simple_edgeaisoc/")
    println("  Main file: SimpleEdgeAiSoC.sv")
    println("  + SimplePicoRV32 (BlackBox)")
    println("  + SimpleCompactAccel.sv")
    println("  + SimpleBitNetAccel.sv")
    println("  + SimpleMemAdapter.sv")
    println("  + SimpleAddressDecoder.sv")
    println("  + SimpleUART.sv")
    println("  + SimpleGPIO.sv")
    
    println("\n💡 Usage Example (C code):")
    println("""
  // 初始化 CompactAccel
  volatile uint32_t *compact = (uint32_t *)0x10000000;
  
  // 写入矩阵 A
  for (int i = 0; i < 64; i++) {
    compact[0x100/4 + i] = matrix_a[i];
  }
  
  // 写入矩阵 B
  for (int i = 0; i < 64; i++) {
    compact[0x300/4 + i] = matrix_b[i];
  }
  
  // 启动计算
  compact[0] = 0x1;  // CTRL = START
  
  // 等待完成
  while ((compact[1] & 0x2) == 0);  // STATUS & DONE
  
  // 读取结果
  for (int i = 0; i < 64; i++) {
    result[i] = compact[0x500/4 + i];
  }
    """)
    
    println("\n🎯 Next Steps:")
    println("  1. Review generated SystemVerilog")
    println("  2. Add PicoRV32 core (picorv32.v)")
    println("  3. Add memory controller")
    println("  4. Synthesize for FPGA")
    println("  5. Write software drivers")
    
    println("\n📊 Estimated Resources (FPGA):")
    println("  - LUTs: ~8,000")
    println("  - FFs: ~6,000")
    println("  - BRAMs: ~20")
    println("  - Frequency: 50-100 MHz")
    
    println("\n🚀 Performance:")
    println("  - CompactAccel: ~1.6 GOPS @ 100MHz")
    println("  - BitNetAccel: ~4.8 GOPS @ 100MHz")
    println("  - Total: ~6.4 GOPS")
    
    println("\n📚 Comparison with Original:")
    println("  Original EdgeAiSoC:")
    println("    ❌ AXI4-Lite interface (direction issues)")
    println("    ❌ Complex address decoder")
    println("    ❌ Cannot generate Verilog")
    println("  ")
    println("  Simple EdgeAiSoC:")
    println("    ✅ Simple register interface")
    println("    ✅ Straightforward decoder")
    println("    ✅ Successfully generates Verilog")
    println("    ✅ Same functionality, simpler design")
    
  } catch {
    case e: Exception =>
      println("\n❌ Error during generation:")
      println(s"  ${e.getMessage}")
      e.printStackTrace()
      throw e
  }
}
