package riscv.ai

import circt.stage.ChiselStage

/**
 * 主程序 - 生成Verilog代码
 */
object Main extends App {
  println("生成RISC-V AI芯片...")
  
  // 生成 RISC-V AI 芯片 Verilog (Chisel 6.x 语法)
  println("\n🔧 生成 RISC-V AI 芯片...")
  ChiselStage.emitSystemVerilogFile(
    new RiscvAiChip,
    Array("--target-dir", "generated")
  )
  
  
  println("\n✅ 所有Verilog代码已生成到 generated/ 目录")
  println("\n📁 生成的文件:")
  println("\n🤖 AI矩阵乘法器:")
  println("  - RiscvAiChip.sv: AI芯片顶层模块")
  println("  - MatrixMultiplier.sv: 矩阵乘法器")
  println("  - MacUnit.sv: MAC单元")
  println("  - MessageSchedule.sv: 消息调度模块")
  println("\n🎯 特性说明:")
  println("  ✅ AXI-Lite总线接口")
  println("  ✅ 参数化设计，易于定制")
  println("  ✅ 完整的测试覆盖")
}