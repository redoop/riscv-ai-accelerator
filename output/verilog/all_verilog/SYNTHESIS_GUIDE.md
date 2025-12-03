# Verilog 综合指南

**日期**: 2025-12-03  
**目标**: ysyxSoC AI 加速器

---

## ⚠️ 综合限制

### SystemVerilog 支持问题

**问题**: `SimpleEdgeAiSoC.sv` 使用了 SystemVerilog 特性，开源工具 Yosys 支持有限。

**错误示例**:
```
SimpleEdgeAiSoC.sv:207: ERROR: syntax error, unexpected TOK_AUTOMATIC
```

---

## 🔧 综合方案

### 方案 1: 使用商业工具（推荐）

#### Vivado (Xilinx)

```tcl
# 创建项目
create_project ysyxsoc_ai ./vivado_project -part xc7a100tcsg324-1

# 添加所有文件
add_files [glob *.v]
add_files [glob *.sv]

# 设置顶层
set_property top ysyxSoCTop [current_fileset]

# 综合
synth_design -top ysyxSoCTop -part xc7a100tcsg324-1

# 报告
report_utilization
report_timing_summary
```

#### Quartus (Intel/Altera)

```tcl
# 创建项目
project_new ysyxsoc_ai -overwrite

# 添加文件
set_global_assignment -name VERILOG_FILE *.v
set_global_assignment -name SYSTEMVERILOG_FILE *.sv

# 设置顶层
set_global_assignment -name TOP_LEVEL_ENTITY ysyxSoCTop

# 综合
execute_flow -compile
```

### 方案 2: 使用 Verilator（仿真）

```bash
verilator --cc --exe --build \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED \
  --top-module ysyxSoCTop \
  *.v *.sv \
  sim_main.cpp

./obj_dir/VysyxSoCTop
```

### 方案 3: 转换为纯 Verilog

使用 `sv2v` 工具转换 SystemVerilog 到 Verilog：

```bash
# 安装 sv2v
# https://github.com/zachjs/sv2v

# 转换
sv2v SimpleEdgeAiSoC.sv > SimpleEdgeAiSoC_converted.v

# 然后用 Yosys 综合
yosys -p "read_verilog SimpleEdgeAiSoC_converted.v; ..."
```

---

## 📊 预期综合结果（基于之前的综合）

### 资源占用（ICS55 55nm）

| 资源 | 数量 |
|------|------|
| 标准单元 | ~96,000 |
| 触发器 | ~25,500 |
| 面积 | ~293,000 µm² |

### 模块分布

| 模块 | 占比 |
|------|------|
| PicoRV32 | ~40% |
| AI 加速器 | ~30% |
| 外设 | ~20% |
| 其他 | ~10% |

---

## 🎯 推荐流程

### 1. 仿真验证（Verilator）

```bash
cd /opt/github/riscv-ai-accelerator/output/verilog/all_verilog

# 编译
verilator --cc --exe --build \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-UNDRIVEN \
  -Wno-PINCONNECTEMPTY -Wno-PINMISSING -Wno-COMBDLY \
  -Wno-TIMESCALEMOD -Wno-MULTIDRIVEN -Wno-CASEINCOMPLETE \
  -Wno-BLKANDNBLK \
  --top-module ysyxSoCTop \
  *.v *.sv \
  sim_main.cpp

# 运行
./obj_dir/VysyxSoCTop
```

### 2. FPGA 综合（Vivado）

```bash
# 启动 Vivado
vivado -mode tcl

# 执行综合脚本
source synth_vivado.tcl
```

### 3. ASIC 综合（商业工具）

使用 Design Compiler, Genus 等工具。

---

## 📁 提供的脚本

| 文件 | 用途 |
|------|------|
| `synth.tcl` | Yosys 综合脚本（有限支持） |
| `synth_vivado.tcl` | Vivado 综合脚本（推荐） |
| `verilator_build.sh` | Verilator 编译脚本 |

---

## ⚠️ 注意事项

1. **SystemVerilog**: 需要支持 SV 的工具
2. **内存模块**: 可能需要替换为 FPGA/ASIC 库
3. **时钟约束**: 需要添加 SDC/XDC 约束文件
4. **IO 约束**: 需要根据目标板定义

---

## 📚 相关文档

- `README.md` - 使用说明
- `../../../chisel/synthesis/` - 之前的综合结果
- `../../../YSYXSOC_AI_BUILD_SUCCESS.md` - 编译成功报告

---

**创建日期**: 2025-12-03  
**状态**: 文档完成  
**推荐**: 使用 Vivado 或 Verilator
