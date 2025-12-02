package riscv.ai

import java.io.{File, PrintWriter}
import scala.io.Source

/**
 * Verilog 后处理工具
 * 清理生成的 SystemVerilog 文件，移除不必要的标记
 */
object PostProcessVerilog {
  
  /**
   * 为模块名添加前缀
   * 
   * @param filePath 文件路径
   * @param prefix 前缀（如 "ip1_"）
   */
  def addModulePrefix(filePath: String, prefix: String): Unit = {
    val file = new File(filePath)
    if (!file.exists()) {
      println(s"⚠️  文件不存在: $filePath")
      return
    }
    
    println(s"🔧 添加模块前缀: $filePath (前缀: $prefix)")
    
    // 读取所有行
    val lines = Source.fromFile(file).getLines().toList
    
    // 添加前缀到模块定义和实例化（但不包括端口连接）
    val prefixedLines = lines.map { line =>
      // 匹配 module 定义: module ModuleName
      val modulePattern = """^(\s*module\s+)(\w+)(.*)$""".r
      // 匹配模块实例化: ModuleName instanceName (但不是端口连接 .portName)
      val instancePattern = """^(\s*)(\w+)(\s+\w+\s*\()""".r
      
      line match {
        case modulePattern(prefix1, moduleName, suffix) =>
          // 跳过已有前缀的模块和标准模块
          if (moduleName.startsWith(prefix) || moduleName.startsWith("picorv32")) {
            line
          } else {
            s"$prefix1$prefix$moduleName$suffix"
          }
        case instancePattern(indent, moduleName, suffix) =>
          // 跳过端口连接（以点开头）、已有前缀的实例化和标准模块
          if (line.trim.startsWith(".") || 
              moduleName.startsWith(prefix) || 
              moduleName.startsWith("picorv32")) {
            line
          } else {
            s"$indent$prefix$moduleName$suffix"
          }
        case _ => line
      }
    }
    
    // 写回文件
    val writer = new PrintWriter(file)
    try {
      prefixedLines.foreach(writer.println)
      println(s"✓ 前缀添加完成")
    } finally {
      writer.close()
    }
  }
  
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
