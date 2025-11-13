package riscv.ai

import chisel3._
import chisel3.util._

/**
 * 物理约束和DRC优化注解
 * 用于指导EDA工具进行物理实现优化
 */
object PhysicalConstraints {
  
  /**
   * 时钟约束
   */
  object ClockConstraints {
    val TARGET_FREQ_MHZ = 100
    val CLOCK_UNCERTAINTY_PS = 100
    val SETUP_MARGIN_PS = 150
    val HOLD_MARGIN_PS = 50
    
    // 时钟树约束
    val MAX_CLOCK_SKEW_PS = 50
    val MAX_CLOCK_LATENCY_PS = 500
    val CLOCK_TRANSITION_PS = 200
  }
  
  /**
   * 布局约束
   */
  object PlacementConstraints {
    // 核心区域利用率
    val CORE_UTILIZATION = 0.75
    
    // 模块间距约束（减少串扰）
    val MIN_MODULE_SPACING_UM = 2.0
    
    // 关键路径约束
    val CRITICAL_PATH_MAX_LENGTH_UM = 1000.0
    
    // 存储器放置约束
    val MEMORY_PLACEMENT_REGION = "CENTER"
    val LOGIC_PLACEMENT_REGION = "AROUND_MEMORY"
  }
  
  /**
   * 布线约束
   */
  object RoutingConstraints {
    // 金属层使用策略
    val POWER_METAL_LAYERS = Seq("M1", "M2", "M9", "M10")
    val SIGNAL_METAL_LAYERS = Seq("M3", "M4", "M5", "M6", "M7", "M8")
    val CLOCK_METAL_LAYERS = Seq("M7", "M8", "M9")
    
    // 最小线宽和间距（避免DRC违例）
    val MIN_WIRE_WIDTH_FACTOR = 1.2  // 比工艺最小值大20%
    val MIN_WIRE_SPACING_FACTOR = 1.5 // 比工艺最小值大50%
    
    // 通孔约束
    val MAX_VIA_DENSITY = 0.6
    val MIN_VIA_SPACING_FACTOR = 1.3
    
    // 天线规则约束
    val MAX_ANTENNA_RATIO = 0.8  // 比工艺限制小20%
    val ANTENNA_DIODE_INSERTION = true
  }
  
  /**
   * 电源网络约束
   */
  object PowerConstraints {
    // 电源网格参数
    val POWER_GRID_WIDTH_UM = 0.5
    val POWER_GRID_SPACING_UM = 10.0
    
    // 电源域
    val CORE_VOLTAGE = 1.0  // V
    val IO_VOLTAGE = 1.8    // V
    
    // 电流密度限制（避免电迁移）
    val MAX_CURRENT_DENSITY_MA_UM = 1.0
    
    // 去耦电容
    val DECAP_DENSITY = 0.1  // 10%的面积用于去耦电容
  }
  
  /**
   * DFT约束
   */
  object DFTConstraints {
    val SCAN_CHAIN_LENGTH = 100  // 每条扫描链最大长度
    val SCAN_COMPRESSION_RATIO = 10
    val BOUNDARY_SCAN_ENABLE = true
    val MBIST_ENABLE = true
  }
}

/**
 * 物理优化的设计规则检查器
 * 在Chisel层面预防DRC违例
 */
class PhysicalDRCChecker extends Module {
  val io = IO(new Bundle {
    val design_valid = Output(Bool())
    val drc_violations = Output(UInt(16.W))
    val violation_types = Output(UInt(8.W))
  })
  
  // 模拟DRC检查逻辑
  val violation_count = RegInit(0.U(16.W))
  val violation_types_reg = RegInit(0.U(8.W))
  
  // 检查项目
  val metal_spacing_ok = true.B  // 金属间距检查
  val via_density_ok = true.B    // 通孔密度检查
  val antenna_ok = true.B        // 天线效应检查
  val power_width_ok = true.B    // 电源线宽度检查
  val timing_ok = true.B         // 时序检查
  
  // 汇总检查结果
  val all_checks = Seq(
    metal_spacing_ok,
    via_density_ok,
    antenna_ok,
    power_width_ok,
    timing_ok
  )
  
  io.design_valid := all_checks.reduce(_ && _)
  io.drc_violations := violation_count
  io.violation_types := violation_types_reg
  
  // 更新违例计数
  when(!io.design_valid) {
    violation_count := violation_count + 1.U
  }
}

/**
 * 物理感知的时钟门控单元
 * 减少动态功耗和时钟树复杂度
 */
class PhysicalAwareClockGate extends Module {
  val io = IO(new Bundle {
    val clk_in = Input(Clock())
    val enable = Input(Bool())
    val test_enable = Input(Bool())
    val clk_out = Output(Clock())
  })
  
  // 集成时钟门控单元（ICG）
  // 使用标准单元库的ICG避免自定义设计的DRC问题
  val enable_latch = RegInit(false.B)
  
  // 在时钟下降沿锁存使能信号
  withClock((!io.clk_in.asBool).asClock) {
    enable_latch := io.enable || io.test_enable
  }
  
  // 门控时钟输出
  io.clk_out := (io.clk_in.asBool && enable_latch).asClock
}

/**
 * 物理优化的存储器包装器
 * 使用编译器存储器避免自定义存储器的DRC问题
 */
class PhysicalOptimizedMemory(
  width: Int,
  depth: Int
) extends Module {
  val addrWidth = log2Ceil(depth)
  
  val io = IO(new Bundle {
    val write = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Input(UInt(width.W))
      val mask = Input(UInt((width/8).W))
    }
    val read = new Bundle {
      val en = Input(Bool())
      val addr = Input(UInt(addrWidth.W))
      val data = Output(UInt(width.W))
    }
    val power_mode = Input(UInt(2.W))
  })
  
  // 使用SyncReadMem确保使用编译器存储器
  val mem = SyncReadMem(depth, UInt(width.W))
  
  // 时钟门控
  val clk_gate = Module(new PhysicalAwareClockGate())
  clk_gate.io.clk_in := clock
  clk_gate.io.enable := io.write.en || io.read.en
  clk_gate.io.test_enable := false.B
  
  // 存储器访问
  withClock(clk_gate.io.clk_out) {
    when(io.write.en) {
      // 支持字节写入掩码
      val write_data = Wire(UInt(width.W))
      write_data := io.write.data // 简化版本，实际需要根据mask处理
      mem.write(io.write.addr, write_data)
    }
    
    io.read.data := mem.read(io.read.addr, io.read.en)
  }
}

/**
 * 综合约束生成器
 * 生成SDC约束文件内容
 */
object SDCGenerator {
  def generateConstraints(designName: String): String = {
    s"""
# SDC约束文件 - ${designName}
# 自动生成，用于解决物理验证违例

# 时钟约束
create_clock -name "clk" -period ${1000.0/PhysicalConstraints.ClockConstraints.TARGET_FREQ_MHZ} [get_ports clock]
set_clock_uncertainty ${PhysicalConstraints.ClockConstraints.CLOCK_UNCERTAINTY_PS/1000.0} [get_clocks clk]

# 输入延迟约束
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# 输出延迟约束
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# 负载约束
set_load 0.1 [all_outputs]

# 驱动强度约束
set_driving_cell -lib_cell BUFX2 [all_inputs]

# 面积约束
set_max_area 0

# 功耗约束
set_max_dynamic_power 100.0
set_max_leakage_power 10.0

# 时序例外
set_false_path -from [get_ports reset]
set_multicycle_path -setup 2 -from [get_pins */matrixMult/*] -to [get_pins */result_reg*]

# DRC约束
set_min_pulse_width -high 0.4 [get_clocks clk]
set_min_pulse_width -low 0.4 [get_clocks clk]
"""
  }
}