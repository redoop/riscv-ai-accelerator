package riscv.ai

import java.io.{File, PrintWriter}
import scala.io.Source

/**
 * Verilog 后处理工具
 * 清理生成的 SystemVerilog 文件，移除不必要的标记
 */
object PostProcessVerilog {
  
  /**
   * 清理 SystemVerilog 文件
   * - 移除 FIRRTL 黑盒资源文件清单标记
   * - 确保文件以 endmodule 结束
   */
  def cleanupVerilogFile(filePath: String): Unit = {
    val file = new File(filePath)
    if (!file.exists()) {
      println(s"⚠️  文件不存在: $filePath")
      return
    }
    
    println(s"🔧 清理文件: $filePath")
    
    // 读取所有行
    val lines = Source.fromFile(file).getLines().toList
    
    // 过滤掉资源清单标记
    val cleanedLines = lines.takeWhile { line =>
      !line.contains("firrtl_black_box_resource_files")
    }
    
    // 写回文件
    val writer = new PrintWriter(file)
    try {
      cleanedLines.foreach(writer.println)
      println(s"✓ 清理完成: 从 ${lines.size} 行减少到 ${cleanedLines.size} 行")
    } finally {
      writer.close()
    }
  }
  
  /**
   * 批量清理目录中的所有 .sv 文件
   */
  def cleanupDirectory(dirPath: String): Unit = {
    val dir = new File(dirPath)
    if (!dir.exists() || !dir.isDirectory) {
      println(s"⚠️  目录不存在: $dirPath")
      return
    }
    
    println(s"\n🔧 清理目录: $dirPath")
    
    val svFiles = dir.listFiles().filter(_.getName.endsWith(".sv"))
    svFiles.foreach { file =>
      cleanupVerilogFile(file.getAbsolutePath)
    }
    
    println(s"✅ 清理完成: 处理了 ${svFiles.length} 个文件")
  }
}

/**
 * 独立运行的清理工具
 */
object CleanupVerilogMain extends App {
  println("="*60)
  println("🧹 SystemVerilog 文件清理工具")
  println("="*60)
  
  // 清理 generated 目录
  PostProcessVerilog.cleanupDirectory("generated")
  
  // 清理测试目录
  PostProcessVerilog.cleanupDirectory("test_results/synthesis")
  
  println("\n✅ 所有文件清理完成！")
}
