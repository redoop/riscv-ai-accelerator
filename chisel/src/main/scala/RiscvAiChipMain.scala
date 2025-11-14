package riscv.ai

import circt.stage.ChiselStage

/**
 * 生成 RISC-V AI 加速器芯片的 Verilog 代码
 */
object RiscvAiChipMain extends App {
  println("Generating RISC-V AI Accelerator Chip Verilog...")
  
  ChiselStage.emitSystemVerilogFile(
    new RiscvAiChip,
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
    args = Array("--target-dir", "generated")
  )
  
  // 后处理: 清理生成的文件
  println("\nPost-processing generated files...")
  PostProcessVerilog.cleanupVerilogFile("generated/RiscvAiChip.sv")
  
  println("\n✅ Verilog generation complete!")
  println("Output directory: generated/")
  println("Main file: generated/RiscvAiChip.sv")
  println("\n💡 文件已优化，可直接用于综合")
}

/**
 * 生成 RISC-V AI 系统的 Verilog 代码 (包含更多细节)
 */
object RiscvAiSystemMain extends App {
  println("Generating RISC-V AI System Verilog...")
  
  ChiselStage.emitSystemVerilogFile(
    new RiscvAiSystem(),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
    args = Array("--target-dir", "generated")
  )
  
  // 后处理: 清理生成的文件
  println("\nPost-processing generated files...")
  PostProcessVerilog.cleanupVerilogFile("generated/RiscvAiSystem.sv")
  
  println("\n✅ Verilog generation complete!")
  println("Output directory: generated/")
  println("Main file: generated/RiscvAiSystem.sv")
}

/**
 * 生成独立的 AI 加速器 Verilog 代码
 */
object CompactScaleAiChipMain extends App {
  println("Generating Compact Scale AI Chip Verilog...")
  
  ChiselStage.emitSystemVerilogFile(
    new CompactScaleAiChip(),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
    args = Array("--target-dir", "generated")
  )
  
  // 后处理: 清理生成的文件
  println("\nPost-processing generated files...")
  PostProcessVerilog.cleanupVerilogFile("generated/CompactScaleAiChip.sv")
  
  println("\n✅ Verilog generation complete!")
  println("Output directory: generated/")
  println("Main file: generated/CompactScaleAiChip.sv")
}

/**
 * 生成 BitNet 专用 AI 加速芯片 Verilog 代码
 * 特点：
 * - 16个 BitNet 计算单元（无乘法器，只用加减法）
 * - 2个 16x16 BitNet 矩阵乘法器
 * - 压缩权重存储（2-bit/权重）
 * - 目标：控制在 50,000 instances 以内
 */
object BitNetScaleAiChipMain extends App {
  println("Generating BitNet Scale AI Chip Verilog...")
  println("Configuration:")
  println("  - 16 BitNet Compute Units (no multipliers)")
  println("  - 2x 16x16 Matrix Multipliers")
  println("  - 2-bit compressed weights")
  println("  - Target: <50K instances")
  
  ChiselStage.emitSystemVerilogFile(
    new BitNetScaleAiChip(
      dataWidth = 16,
      matrixSize = 16,
      numComputeUnits = 16,
      numMatrixUnits = 2,
      memoryDepth = 1024,
      addrWidth = 10
    ),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"),
    args = Array("--target-dir", "generated")
  )
  
  // 后处理: 清理生成的文件
  println("\nPost-processing generated files...")
  PostProcessVerilog.cleanupVerilogFile("generated/BitNetScaleAiChip.sv")
  
  println("\n✅ BitNet Verilog generation complete!")
  println("Output directory: generated/")
  println("Main file: generated/BitNetScaleAiChip.sv")
  println("\n💡 BitNet 芯片特点:")
  println("   - 无乘法器设计，功耗极低")
  println("   - 权重压缩至 2-bit，存储效率高")
  println("   - 专为 {-1, 0, +1} 权重优化")
  println("   - 可直接用于综合和流片")
}
