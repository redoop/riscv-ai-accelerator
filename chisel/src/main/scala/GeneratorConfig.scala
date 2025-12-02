package riscv.ai

/**
 * Verilog 生成器配置
 * 
 * 用于配置模块前缀等生成选项
 */
object GeneratorConfig {
  /**
   * 模块前缀配置
   * 
   * 根据项目编号设置不同的前缀：
   * - ip0_: project_1854
   * - ip1_: project_1984 (当前项目)
   * - ip2_: project_1839
   * - ip3_: ysyxSoCASIC
   * - ip4_: project_1988
   * - ip5_: project_1993
   */
  val MODULE_PREFIX = "ip1_"
  
  /**
   * 获取完整的 firtool 选项
   * 注意：firtool 1.62.0 不支持 --prefix-modules，使用后处理添加前缀
   */
  def getFirtoolOpts: Array[String] = Array(
    "-disable-all-randomization",
    "-strip-debug-info"
  )
  
  /**
   * 是否启用模块前缀（通过后处理）
   */
  val ENABLE_PREFIX = true
  
  /**
   * 获取带条件的 firtool 选项
   */
  def getFirtoolOptsConditional: Array[String] = {
    Array(
      "-disable-all-randomization",
      "-strip-debug-info"
    )
  }
}
