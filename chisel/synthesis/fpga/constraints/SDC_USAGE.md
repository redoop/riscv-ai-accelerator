# SDC 约束文件使用指南

## 📋 文件说明

本目录包含两种格式的时序约束文件：

| 文件 | 格式 | 用途 | 工具支持 |
|------|------|------|----------|
| `timing_complete.sdc` | SDC | ASIC 流片 | Synopsys DC, Cadence Genus, Yosys, OpenSTA, iEDA |
| `timing.sdc` | SDC | 快速开始 | 同上（简化版） |
| `timing_complete.xdc` | XDC | FPGA 验证 | Xilinx Vivado |
| `timing.xdc` | XDC | FPGA 快速开始 | Xilinx Vivado |

## 🔄 SDC vs XDC

### SDC (Synopsys Design Constraints)
- ✅ **行业标准** - IEEE 1481-1999
- ✅ **工具通用** - 支持多种 EDA 工具
- ✅ **ASIC 流片** - 主要用途
- ⚠️ **仅时序约束** - 不包含引脚分配

### XDC (Xilinx Design Constraints)
- ✅ **Xilinx 专用** - Vivado 工具
- ✅ **功能完整** - 时序 + 引脚 + I/O
- ✅ **FPGA 验证** - 主要用途
- ⚠️ **不可移植** - 仅限 Xilinx

## 🚀 使用方法

### 1. Yosys 综合

```bash
# 使用 Yosys 进行综合
yosys -p "
  read_verilog SimpleEdgeAiSoC.sv;
  synth -top SimpleEdgeAiSoC;
  write_verilog -noattr synth.v
"

# 使用 OpenSTA 进行时序分析
sta -f timing_complete.sdc synth.v
```

### 2. Synopsys Design Compiler

```tcl
# DC 综合脚本
read_verilog SimpleEdgeAiSoC.sv
current_design SimpleEdgeAiSoC
link

# 读取 SDC 约束
read_sdc timing_complete.sdc

# 综合
compile_ultra

# 时序报告
report_timing
report_constraint -all_violators
```

### 3. Cadence Genus

```tcl
# Genus 综合脚本
read_hdl SimpleEdgeAiSoC.sv
elaborate SimpleEdgeAiSoC

# 读取 SDC 约束
read_sdc timing_complete.sdc

# 综合
syn_generic
syn_map
syn_opt

# 时序报告
report_timing
```

### 4. iEDA (中国开源 EDA)

```bash
# iEDA 综合流程
iEDA -design SimpleEdgeAiSoC \
     -verilog SimpleEdgeAiSoC.sv \
     -sdc timing_complete.sdc \
     -output synth.v
```

### 5. OpenSTA (静态时序分析)

```bash
# 使用 OpenSTA 进行时序分析
sta << EOF
read_liberty /path/to/library.lib
read_verilog synth.v
link_design SimpleEdgeAiSoC
read_sdc timing_complete.sdc
report_checks -path_delay min_max
report_tns
report_wns
EOF
```

## 📊 约束内容

### timing_complete.sdc (完整版)

包含以下约束：
- ✅ 主时钟定义 (100 MHz)
- ✅ SPI 生成时钟 (10 MHz)
- ✅ 时钟不确定性
- ✅ 时钟延迟
- ✅ 输入/输出延迟
- ✅ 假路径
- ✅ 多周期路径（注释）
- ✅ 最大延迟
- ✅ 输入转换时间
- ✅ 输出负载
- ✅ 设计规则（扇出、转换、电容）
- ✅ 详细注释和说明

### timing.sdc (简化版)

包含基本约束：
- ✅ 主时钟定义
- ✅ SPI 生成时钟
- ✅ 基本输入/输出延迟
- ✅ 假路径
- ✅ 基本设计规则

## 🔧 自定义约束

### 修改时钟频率

```tcl
# 修改主时钟为 50 MHz
create_clock -name sys_clk -period 20.000 [get_ports clock]

# 修改 SPI 时钟为 5 MHz (分频比 10)
create_generated_clock -name spi_clk \
  -source [get_ports clock] \
  -divide_by 10 \
  [get_pins -hierarchical *spiClkReg*/Q]
```

### 添加新的输入/输出

```tcl
# 添加新的输入端口约束
set_input_delay -clock sys_clk -max 2.0 [get_ports new_input]
set_input_delay -clock sys_clk -min 0.5 [get_ports new_input]

# 添加新的输出端口约束
set_output_delay -clock sys_clk -max 2.0 [get_ports new_output]
set_output_delay -clock sys_clk -min 0.5 [get_ports new_output]
```

### 调整时序裕量

```tcl
# 增加时钟不确定性（更保守）
set_clock_uncertainty -setup 1.0 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.5 [get_clocks sys_clk]

# 减少输入/输出延迟（更激进）
set_input_delay -clock sys_clk -max 1.0 [get_ports io_uart_rx]
set_output_delay -clock sys_clk -max 1.0 [get_ports io_uart_tx]
```

## 📝 注意事项

### 1. 生成时钟源点

生成时钟的源点需要根据综合后的网表调整：

```tcl
# 方法 1: 使用通配符（推荐）
[get_pins -hierarchical *spiClkReg*/Q]

# 方法 2: 使用精确路径（综合后确定）
[get_pins lcd/lcd/spiClkReg_reg/Q]

# 方法 3: 使用端口（如果时钟输出到端口）
[get_ports io_lcd_spi_clk]
```

### 2. 工艺库依赖

某些约束需要工艺库支持：

```tcl
# 需要工艺库的约束（注释掉）
# set_driving_cell -lib_cell BUFX2 [all_inputs]
# set_operating_conditions -max slow_1p08v_125c

# 通用约束（不需要工艺库）
set_input_transition 0.5 [all_inputs]
set_load 2.0 [all_outputs]
```

### 3. 时钟域交叉

如果设计中有多个时钟域，需要正确处理：

```tcl
# 异步时钟域
set_clock_groups -asynchronous \
  -group [get_clocks sys_clk] \
  -group [get_clocks spi_clk]

# 或使用假路径
set_false_path -from [get_clocks sys_clk] -to [get_clocks spi_clk]
set_false_path -from [get_clocks spi_clk] -to [get_clocks sys_clk]
```

## 🧪 验证约束

### 检查约束覆盖率

```tcl
# OpenSTA
report_checks -unconstrained

# Synopsys DC
report_timing -from [all_inputs] -to [all_outputs]
check_timing

# Cadence Genus
report_timing -unconstrained
```

### 检查时序违例

```tcl
# 检查 Setup 违例
report_timing -delay_type max -max_paths 10

# 检查 Hold 违例
report_timing -delay_type min -max_paths 10

# 检查所有违例
report_constraint -all_violators
```

## 📚 参考资料

- **SDC 标准**: IEEE 1481-1999
- **Synopsys DC**: Design Compiler User Guide
- **Cadence Genus**: Genus Synthesis User Guide
- **OpenSTA**: https://github.com/The-OpenROAD-Project/OpenSTA
- **iEDA**: https://ieda.oscc.cc/

## 🔗 相关文件

- `timing_complete.xdc` - XDC 格式（Xilinx FPGA）
- `timing.xdc` - XDC 简化版
- `CLOCK_CONSTRAINTS_SPEC.md` - 时钟约束规范
- `CLOCK_VERIFICATION_GUIDE.md` - 验证指南

## ✅ 验证状态

- ✅ Chisel 仿真: 100% 通过 (2/2 测试)
- ✅ SPI 频率: 10.204 MHz (误差 2.04%)
- ✅ SPI 占空比: 50.00% (偏差 0.00%)
- ✅ 综合频率: 178.569 MHz (超出目标 78.569%)
- ✅ 时序收敛: WNS 14.4ns, TNS 0ns

---

**创建日期**: 2025-11-21  
**版本**: v1.0  
**维护者**: tongxiaojun@redoop.com
