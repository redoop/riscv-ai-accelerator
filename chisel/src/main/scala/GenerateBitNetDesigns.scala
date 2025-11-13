package riscv.ai

import circt.stage.ChiselStage

/**
 * BitNet 设计生成器
 * 生成 BitNetScaleAiChip 的 Verilog 代码
 */
object GenerateBitNetDesigns extends App {
  
  println("=" * 80)
  println("🔧 生成 BitNet AI 芯片设计")
  println("=" * 80)
  println()
  
  // 创建输出目录
  val outputDir = new java.io.File("generated/bitnet")
  if (!outputDir.exists()) {
    outputDir.mkdirs()
    println(s"📁 创建输出目录: ${outputDir.getAbsolutePath}")
  }
  
  try {
    // 生成 BitNetScaleAiChip
    println("🔧 生成 BitNetScaleAiChip...")
    val bitnetVerilog = ChiselStage.emitSystemVerilog(
      new BitNetScaleAiChip(),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
    
    val bitnetFile = new java.io.File(outputDir, "BitNetScaleAiChip.sv")
    val bitnetWriter = new java.io.PrintWriter(bitnetFile)
    bitnetWriter.write(bitnetVerilog)
    bitnetWriter.close()
    
    println(s"✅ BitNetScaleAiChip.sv 生成完成")
    println(s"📄 文件位置: ${bitnetFile.getAbsolutePath}")
    
    // 统计代码行数
    val bitnetLines = scala.io.Source.fromFile(bitnetFile).getLines().size
    
    println()
    println("📊 BitNet 设计规模统计:")
    println(s"📄 BitNetScaleAiChip.sv: $bitnetLines 行")
    
    // 预估 instances 数量（基于代码行数的经验公式）
    val estimatedInstances = bitnetLines * 12  // 经验值：每行约12个instances
    
    println()
    println("🎯 预估 Instance 数量:")
    println(s"  - BitNetScale: ~$estimatedInstances instances")
    
    if (estimatedInstances <= 50000) {
      println(s"  ✅ 满足 5万 instances 限制 (余量: ${50000 - estimatedInstances})")
    } else {
      println(s"  ⚠️  超出 5万 instances 限制 (超出: ${estimatedInstances - 50000})")
    }
    
    println()
    println("💡 BitNet 设计特点:")
    println("  ✅ 无乘法器 - 只有加减法")
    println("  ✅ 权重压缩 - 2-bit 存储")
    println("  ✅ 稀疏性优化 - 跳过零权重")
    println("  ✅ 16x16 矩阵 - 4倍容量提升")
    println("  ✅ 双矩阵单元 - 2倍并行度")
    
    println()
    println("🔧 推荐使用:")
    println("  1. BitNet 模型推理: BitNetScaleAiChip")
    println("  2. 边缘 LLM 应用: BitNetScaleAiChip")
    println("  3. 低功耗 AI 推理: BitNetScaleAiChip")
    
    println()
    println("=" * 80)
    println("✅ BitNet 设计生成完成！")
    println("=" * 80)
    
  } catch {
    case e: Exception =>
      println(s"❌ 生成失败: ${e.getMessage}")
      e.printStackTrace()
      sys.exit(1)
  }
}
