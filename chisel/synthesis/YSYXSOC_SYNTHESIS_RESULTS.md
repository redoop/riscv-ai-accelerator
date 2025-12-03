# ysyxSoC 综合结果

## 概述

成功使用 ICS55 PDK 对 `ysyx_26000001` 模块进行逻辑综合。

## 综合配置

- **顶层模块**: `ysyx_26000001`
- **RTL 源文件**:
  - `SimpleEdgeAiSoC.sv` (通过 slang 转换为 Verilog)
  - `ysyx_26000001_simple.v` (简化的 ysyxSoC wrapper)
- **PDK**: ICS55 (55nm)
- **标准单元库**: ics55_LLSC_H7CL (Low Leakage, Typical-Typical, 1.2V, 25°C)
- **综合工具**: Yosys 0.58+138 with slang plugin

## 综合统计

### 核心模块 (ip1_SimpleEdgeAiSoC)

| 指标 | 数值 |
|------|------|
| **总单元数** | 103,480 |
| **触发器 (DFFQX1H7L)** | 25,870 |
| **组合逻辑门** | 77,610 |
| **芯片面积** | 300,724.76 µm² |
| **时序单元面积占比** | 52.99% |
| **线网数** | 79,001 |
| **线网位数** | 110,116 |

### 主要单元类型分布

| 单元类型 | 数量 | 面积 (µm²) | 说明 |
|---------|------|-----------|------|
| DFFQX1H7L | 25,870 | 159,359.2 | D 触发器 |
| OAI21X0P5H7L | 14,455 | 20,237.0 | OR-AND-INVERT |
| NAND2X0P5H7L | 13,852 | 15,500.0 | 2输入 NAND |
| MUX2X0P5H7L | 10,427 | 29,195.6 | 2选1多路复用器 |
| OAI211X0P7H7L | 5,407 | 9,083.76 | OR-AND-INVERT |
| BUFX0P5H7L | 4,384 | 4,910.08 | 缓冲器 |
| AOI22X0P5H7L | 4,153 | 8,139.88 | AND-OR-INVERT |
| AOI21X0P5H7L | 3,638 | 5,093.2 | AND-OR-INVERT |
| NOR2X0P5H7L | 2,870 | 3,214.4 | 2输入 NOR |
| NAND2BX0P5H7L | 2,075 | 2,905.0 | 2输入 NAND (1反相) |
| MUX4X0P5H7L | 1,931 | 12,400.0 | 4选1多路复用器 |
| AO222X0P5H7L | 1,384 | 4,650.24 | AND-OR |

## 设计层次

```
ysyx_26000001 (顶层)
└── ip1_SimpleEdgeAiSoC (SoC 核心)
    ├── PicoRV32 CPU
    ├── CompactAccel (矩阵加速器)
    ├── BitNetAccel (二值神经网络加速器)
    ├── UART
    ├── LCD 控制器
    ├── GPIO
    ├── SPI Flash 控制器
    └── PSRAM 控制器
```

## 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `netlist/ysyx_26000001_syn.v` | 14 MB | 综合后的网表 |
| `netlist/synthesis_stats_ysyxsoc.txt` | 3.8 KB | 综合统计信息 |
| `netlist/synthesis_ysyxsoc.log` | - | 完整综合日志 |
| `netlist/SimpleEdgeAiSoC_converted.v` | - | SV 转 V 中间文件 |
| `netlist/ics55_LLSC_H7CL.v` | - | 标准单元 Verilog 模型 |

## 综合流程

### 步骤 1: SystemVerilog 转换
```bash
# 使用 slang 插件将 SystemVerilog 转换为 Verilog
plugin -i slang
read_slang SimpleEdgeAiSoC.sv
write_verilog SimpleEdgeAiSoC_converted.v
```

### 步骤 2: 逻辑综合
```bash
# 读取 Verilog 文件
read_verilog SimpleEdgeAiSoC_converted.v
read_verilog ysyx_26000001_simple.v

# 设置层次和综合
hierarchy -top ysyx_26000001
proc; opt; fsm; opt; memory; opt; techmap; opt

# 映射到 ICS55 标准单元
dfflibmap -liberty ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
abc -liberty ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib -D 10000

# 输出网表
write_verilog ysyx_26000001_syn.v
```

## 性能估算

基于 ICS55 PDK 典型工艺角 (TT, 1.2V, 25°C):

- **估算最大频率**: ~100-200 MHz (需要静态时序分析确认)
- **功耗**: 待 STA 和功耗分析
- **面积**: 0.30 mm²

## 使用方法

### 运行综合
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis
./run_ysyxsoc_synthesis.sh
```

### 查看统计
```bash
cat netlist/synthesis_stats_ysyxsoc.txt
```

### 查看网表
```bash
less netlist/ysyx_26000001_syn.v
```

## 后续步骤

1. **静态时序分析 (STA)**
   ```bash
   # 使用 OpenSTA
   sta -f netlist/timing_constraints.sdc netlist/ysyx_26000001_syn.v
   ```

2. **后综合仿真**
   ```bash
   # 需要实现 ysyxsoc 仿真支持
   python run_post_syn_sim.py --simulator iverilog --netlist ysyxsoc
   ```

3. **布局布线 (P&R)**
   - 使用 OpenROAD 或商业 P&R 工具
   - 需要 ICS55 PDK 的 LEF/DEF 文件

4. **功耗分析**
   - 使用 VCS 或其他工具进行功耗估算

## 注意事项

1. **简化的 Wrapper**: 当前使用 `ysyx_26000001_simple.v`，它只是简单地实例化 `ip1_SimpleEdgeAiSoC`，ysyxSoC 接口信号为空。完整实现需要连接实际的内存和总线接口。

2. **模块内联**: slang 将 SystemVerilog 的子模块内联到父模块中，因此最终网表中只有 `ip1_SimpleEdgeAiSoC` 和 `ysyx_26000001` 两个模块。

3. **时序约束**: 当前未使用 SDC 约束，建议添加时序约束以获得更好的综合结果。

4. **面积优化**: 可以通过调整综合策略和约束来进一步优化面积和性能。

## 文件清单

```
synthesis/
├── run_ysyxsoc_synthesis.sh      # 综合脚本
├── ysyx_26000001_simple.v        # 简化的 wrapper
├── filelist_ysyxsoc.f            # 文件列表
└── netlist/
    ├── ysyx_26000001_syn.v       # 综合网表
    ├── synthesis_stats_ysyxsoc.txt
    ├── synthesis_ysyxsoc.log
    ├── SimpleEdgeAiSoC_converted.v
    └── ics55_LLSC_H7CL.v
```

## 参考

- [ICS55 PDK](https://github.com/IDE-Platform/icsprout55-pdk)
- [Yosys 文档](https://yosyshq.readthedocs.io/)
- [ICS55 综合指南](ICS55_PDK_GUIDE.md)
