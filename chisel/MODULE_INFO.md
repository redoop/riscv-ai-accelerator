# RISC-V AI 加速器模块信息

## RiscvAiChip (顶层芯片)

### 基本信息
- **Top Module Name**: `RiscvAiChip`
- **Clock Name**: `clock`
- **Reset Name**: `reset`
- **文件**: `generated/RiscvAiChip.sv`
- **文件大小**: 112K (3,704 行)

### 端口定义

```systemverilog
module RiscvAiChip(
  input         clock,              // 时钟信号
                reset,              // 复位信号（高电平有效）
  
  // 内存接口
  output        io_mem_valid,       // 内存请求有效
  input         io_mem_ready,       // 内存准备好
  output [31:0] io_mem_addr,        // 内存地址
                io_mem_wdata,       // 写数据
  output [3:0]  io_mem_wstrb,       // 写字节使能
  input  [31:0] io_mem_rdata,       // 读数据
  
  // 中断接口
  input  [31:0] io_irq,             // 中断请求
  
  // 状态输出
  output        io_trap,            // 陷阱信号
                io_busy,            // AI 加速器忙
  
  // 性能计数器
  output [31:0] io_perf_counters_0, // 性能计数器 0
                io_perf_counters_1, // 性能计数器 1
                io_perf_counters_2, // 性能计数器 2
                io_perf_counters_3  // 性能计数器 3
);
```

### 时钟约束

```tcl
# 创建时钟
create_clock -period 10.0 -name clock [get_ports clock]

# 时钟不确定性
set_clock_uncertainty 0.5 [get_clocks clock]

# 输入延迟
set_input_delay -clock clock 2.0 [all_inputs]

# 输出延迟
set_output_delay -clock clock 2.0 [all_outputs]
```

### 复位约束

```tcl
# 复位是同步高电平有效
set_false_path -from [get_ports reset]
```

---

## RiscvAiSystem (完整系统)

### 基本信息
- **Top Module Name**: `RiscvAiSystem`
- **Clock Name**: `clock`
- **Reset Name**: `reset`
- **文件**: `generated/RiscvAiSystem.sv`
- **文件大小**: 111K (3,675 行)

### 端口定义

```systemverilog
module RiscvAiSystem(
  input         clock,              // 时钟信号
                reset,              // 复位信号（高电平有效）
  
  // 内存接口
  output        io_mem_valid,       // 内存请求有效
                io_mem_instr,       // 指令获取标志
  input         io_mem_ready,       // 内存准备好
  output [31:0] io_mem_addr,        // 内存地址
                io_mem_wdata,       // 写数据
  output [3:0]  io_mem_wstrb,       // 写字节使能
  input  [31:0] io_mem_rdata,       // 读数据
  
  // 中断接口
  input  [31:0] io_irq,             // 中断请求
  output [31:0] io_eoi,             // 中断结束
  
  // 状态输出
  output        io_trap,            // 陷阱信号
                io_ai_busy,         // AI 加速器忙
                io_ai_done,         // AI 加速器完成
  
  // 性能计数器
  output [31:0] io_perf_counters_0, // 性能计数器 0
                io_perf_counters_1, // 性能计数器 1
                io_perf_counters_2, // 性能计数器 2
                io_perf_counters_3, // 性能计数器 3
  
  // Trace 接口
  output        io_trace_valid,     // Trace 有效
  output [35:0] io_trace_data       // Trace 数据
);
```

---

## CompactScaleAiChip (AI 加速器)

### 基本信息
- **Top Module Name**: `CompactScaleAiChip`
- **Clock Name**: `clock`
- **Reset Name**: `reset`
- **文件**: `generated/CompactScaleAiChip.sv`
- **文件大小**: 15K (515 行)

### 端口定义

```systemverilog
module CompactScaleAiChip(
  input         clock,              // 时钟信号
                reset,              // 复位信号（高电平有效）
  
  // AXI-Lite 写地址通道
  input  [9:0]  io_axi_awaddr,      // 写地址
  input         io_axi_awvalid,     // 写地址有效
  output        io_axi_awready,     // 写地址准备好
  
  // AXI-Lite 写数据通道
  input  [31:0] io_axi_wdata,       // 写数据
  input         io_axi_wvalid,      // 写数据有效
  output        io_axi_wready,      // 写数据准备好
  
  // AXI-Lite 写响应通道
  output        io_axi_bvalid,      // 写响应有效
  input         io_axi_bready,      // 写响应准备好
  
  // AXI-Lite 读地址通道
  input  [9:0]  io_axi_araddr,      // 读地址
  input         io_axi_arvalid,     // 读地址有效
  output        io_axi_arready,     // 读地址准备好
  
  // AXI-Lite 读数据通道
  output [31:0] io_axi_rdata,       // 读数据
  output        io_axi_rvalid,      // 读数据有效
  input         io_axi_rready,      // 读数据准备好
  
  // 状态输出
  output        io_status_busy,     // 忙标志
                io_status_done,     // 完成标志
  
  // 性能计数器
  output [31:0] io_perf_counters_0, // 性能计数器 0
                io_perf_counters_1, // 性能计数器 1
                io_perf_counters_2, // 性能计数器 2
                io_perf_counters_3  // 性能计数器 3
);
```

---

## 模块层次关系

```
RiscvAiChip (顶层)
  └── RiscvAiSystem (系统集成)
       ├── PicoRV32BlackBox (CPU)
       │    └── picorv32 (Verilog 模块)
       │         - clock: clk
       │         - reset: resetn (低电平有效)
       └── CompactScaleAiChip (AI 加速器)
            ├── MatrixMultiplier
            │    └── MacUnit
            └── AXI-Lite 接口
```

---

## 时钟域信息

### 主时钟域
- **时钟名称**: `clock`
- **目标频率**: 100 MHz (10 ns 周期)
- **时钟源**: 外部输入
- **时钟分布**: 全局时钟网络

### 复位策略
- **复位类型**: 同步复位
- **复位极性**: 高电平有效（Chisel 生成）
- **复位持续时间**: 至少 10 个时钟周期

**注意**: PicoRV32 内部使用低电平有效复位 (`resetn`)，在 `RiscvAiSystem` 中自动转换：
```systemverilog
cpu.io.resetn := !reset.asBool
```

---

## 综合约束示例

### Vivado (Xilinx FPGA)

```tcl
# 时钟约束
create_clock -period 10.000 -name clock [get_ports clock]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clock]

# 输入输出延迟
set_input_delay -clock clock -min 1.0 [all_inputs]
set_input_delay -clock clock -max 3.0 [all_inputs]
set_output_delay -clock clock -min 1.0 [all_outputs]
set_output_delay -clock clock -max 3.0 [all_outputs]

# 复位约束
set_false_path -from [get_ports reset]

# 时钟不确定性
set_clock_uncertainty 0.5 [get_clocks clock]
```

### Design Compiler (ASIC)

```tcl
# 时钟定义
create_clock -name clock -period 10.0 [get_ports clock]

# 时钟不确定性
set_clock_uncertainty 0.5 [get_clocks clock]

# 时钟转换时间
set_clock_transition 0.1 [get_clocks clock]

# 输入延迟
set_input_delay -clock clock -max 2.0 [remove_from_collection [all_inputs] [get_ports clock]]
set_input_delay -clock clock -min 0.5 [remove_from_collection [all_inputs] [get_ports clock]]

# 输出延迟
set_output_delay -clock clock -max 2.0 [all_outputs]
set_output_delay -clock clock -min 0.5 [all_outputs]

# 复位路径
set_false_path -from [get_ports reset]

# 负载约束
set_load 0.1 [all_outputs]

# 驱动约束
set_driving_cell -lib_cell BUFX2 [all_inputs]
```

---

## Verilog 实例化示例

### 实例化 RiscvAiChip

```systemverilog
RiscvAiChip u_riscv_ai_chip (
    // 时钟和复位
    .clock              (sys_clk),
    .reset              (sys_rst),
    
    // 内存接口
    .io_mem_valid       (mem_valid),
    .io_mem_ready       (mem_ready),
    .io_mem_addr        (mem_addr),
    .io_mem_wdata       (mem_wdata),
    .io_mem_wstrb       (mem_wstrb),
    .io_mem_rdata       (mem_rdata),
    
    // 中断
    .io_irq             (irq_signals),
    
    // 状态
    .io_trap            (cpu_trap),
    .io_busy            (ai_busy),
    
    // 性能计数器
    .io_perf_counters_0 (perf_cnt_0),
    .io_perf_counters_1 (perf_cnt_1),
    .io_perf_counters_2 (perf_cnt_2),
    .io_perf_counters_3 (perf_cnt_3)
);
```

### 实例化 CompactScaleAiChip

```systemverilog
CompactScaleAiChip u_ai_accel (
    // 时钟和复位
    .clock              (sys_clk),
    .reset              (sys_rst),
    
    // AXI-Lite 写通道
    .io_axi_awaddr      (axi_awaddr),
    .io_axi_awvalid     (axi_awvalid),
    .io_axi_awready     (axi_awready),
    .io_axi_wdata       (axi_wdata),
    .io_axi_wvalid      (axi_wvalid),
    .io_axi_wready      (axi_wready),
    .io_axi_bvalid      (axi_bvalid),
    .io_axi_bready      (axi_bready),
    
    // AXI-Lite 读通道
    .io_axi_araddr      (axi_araddr),
    .io_axi_arvalid     (axi_arvalid),
    .io_axi_arready     (axi_arready),
    .io_axi_rdata       (axi_rdata),
    .io_axi_rvalid      (axi_rvalid),
    .io_axi_rready      (axi_rready),
    
    // 状态
    .io_status_busy     (ai_busy),
    .io_status_done     (ai_done),
    
    // 性能计数器
    .io_perf_counters_0 (perf_cnt_0),
    .io_perf_counters_1 (perf_cnt_1),
    .io_perf_counters_2 (perf_cnt_2),
    .io_perf_counters_3 (perf_cnt_3)
);
```

---

## 快速参考

| 模块 | Top Name | Clock | Reset | 文件大小 |
|------|----------|-------|-------|---------|
| **RiscvAiChip** | `RiscvAiChip` | `clock` | `reset` (高有效) | 112K |
| **RiscvAiSystem** | `RiscvAiSystem` | `clock` | `reset` (高有效) | 111K |
| **CompactScaleAiChip** | `CompactScaleAiChip` | `clock` | `reset` (高有效) | 15K |

---

## 相关文档

- [GENERATED_FILES.md](GENERATED_FILES.md) - 生成文件总结
- [TEST_SUCCESS_SUMMARY.md](TEST_SUCCESS_SUMMARY.md) - 测试总结
- [generated/constraints/design_constraints.sdc](generated/constraints/design_constraints.sdc) - SDC 约束文件

---

**更新日期**: 2024年11月14日
