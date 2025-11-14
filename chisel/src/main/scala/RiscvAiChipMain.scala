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
