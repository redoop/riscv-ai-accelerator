# ysyxSoC Wrapper 综合说明

## 概述

本目录包含用于综合 ysyxSoC 包装器的脚本和文件列表。包装器将 SimpleEdgeAiSoC 的 AI 加速器集成到 ysyxSoC 框架中。

## 文件说明

### 综合脚本

- `run_ysyxsoc_synthesis.sh` - 完整 ysyxSoC 包装器综合脚本
- `run_ics55_synthesis.sh` - 原始 SimpleEdgeAiSoC 综合脚本（参考）

### 文件列表

- `filelist_ysyxsoc.f` - 完整 ysyxSoC 包装器文件列表
- `filelist_ysyxsoc_simple.f` - 简化版文件列表（仅 AI 加速器）
- `filelist_core.f` - 原始 SimpleEdgeAiSoC 文件列表（参考）

### 设计文件

- `ysyxSoc/all_verilog/ysyx_26000001_with_ai.v` - 完整 ysyxSoC 包装器
- `ysyx_26000001_simple.v` - 简化版包装器（测试用）

## 设计架构

### ysyx_26000001 顶层模块

```
ysyx_26000001
├── PicoRV32 CPU Core
├── AI Accelerators (from SimpleEdgeAiSoC)
│   ├── ip1_SimpleCompactAccel (矩阵加速器)
│   └── ip1_SimpleBitNetAccel (BitNet 加速器)
├── Storage Controllers
│   ├── ip1_SPIFlash (Flash 控制器)
│   └── ip1_PSRAM (PSRAM 控制器)
└── SimpleBus Interface
    ├── IFU (指令获取)
    └── LSU (数据加载/存储)
```

### 内存映射

```
0x0000_0000 - 0x0FFF_FFFF: PSRAM (ysyxSoC - 256MB)
0x0400_0000 - 0x047F_FFFF: PSRAM (SimpleEdgeAiSoC - 8MB)
0x1000_0000 - 0x1FFF_FFFF: SDRAM (ysyxSoC)
0x2000_0000 - 0x2000_0FFF: CompactAccel
0x2000_1000 - 0x2000_1FFF: BitNetAccel
0x2000_2000 - 0x2000_2FFF: Flash Controller
0x2000_3000 - 0x2000_3FFF: PSRAM Controller
0x3000_0000 - 0x3FFF_FFFF: Flash Memory (16MB)
```

## 使用方法

### 1. 准备工作

确保已生成 SimpleEdgeAiSoC RTL：

```bash
cd ../chisel
make
```

确保已安装 ICS55 PDK：

```bash
cd synthesis
python pdk/get_ics55_pdk.py
```

### 2. 运行综合

#### 方式 A: 完整 ysyxSoC 包装器

```bash
cd chisel/synthesis
./run_ysyxsoc_synthesis.sh
```

这将综合完整的 ysyxSoC 包装器，包括：
- PicoRV32 CPU
- AI 加速器（CompactAccel + BitNetAccel）
- 存储控制器（Flash + PSRAM）
- SimpleBus 接口

#### 方式 B: 简化版（仅 AI 加速器）

修改 `run_ysyxsoc_synthesis.sh` 中的 `FILELIST` 变量：

```bash
FILELIST="filelist_ysyxsoc_simple.f"
```

然后运行：

```bash
./run_ysyxsoc_synthesis.sh
```

### 3. 查看结果

综合完成后，输出文件位于 `netlist/` 目录：

```bash
# 查看网表
cat netlist/ysyx_26000001_ics55.v

# 查看综合统计
cat netlist/synthesis_stats_ysyxsoc.txt

# 查看综合日志
cat netlist/synthesis_ysyxsoc.log
```

### 4. 后续步骤

#### 静态时序分析

```bash
sta -f netlist/timing_constraints_ysyxsoc.sdc netlist/ysyx_26000001_ics55.v
```

#### 后综合仿真

```bash
python run_post_syn_sim.py --simulator iverilog --netlist ysyxsoc
```

## 综合配置

### PDK 配置

- **PDK**: ICS55 (icsprout55-pdk)
- **标准单元库**: ics55_LLSC_H7CL (Low Leakage Standard Cell)
- **工艺角**: Typical-Typical (tt), 1.2V, 25°C
- **Liberty 文件**: `ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib`

### Yosys 综合流程

1. **读取设计**: 使用 slang 插件读取 SystemVerilog
2. **层次化**: 设置顶层模块并检查层次结构
3. **综合**: proc → opt → fsm → opt → memory → opt → techmap → opt
4. **映射**: dfflibmap + abc (带时序约束)
5. **输出**: 生成网表和统计信息

### 时序约束

使用 `fpga/constraints/timing_complete.sdc` 中的约束：
- 时钟周期: 10ns (100MHz)
- 输入/输出延迟
- 时钟不确定性

## 故障排除

### 问题 1: 找不到 SimpleEdgeAiSoC.sv

**解决方案**:
```bash
cd ../chisel
make
```

### 问题 2: 找不到 ICS55 PDK

**解决方案**:
```bash
python pdk/get_ics55_pdk.py
```

### 问题 3: Yosys 综合失败

**检查**:
1. 查看日志: `cat netlist/synthesis_ysyxsoc.log`
2. 检查文件列表中的所有文件是否存在
3. 确认 slang 插件已安装

### 问题 4: 模块未定义错误

**原因**: SimpleEdgeAiSoC.sv 包含所有需要的模块定义

**解决方案**: 确保 SimpleEdgeAiSoC.sv 在文件列表中，且路径正确

## 设计特点

### 优势

1. **模块化**: AI 加速器作为独立模块，易于集成
2. **标准接口**: 使用 SimpleBus 协议，兼容 ysyxSoC
3. **内存映射**: 清晰的地址空间划分
4. **可扩展**: 易于添加新的加速器或外设

### 性能指标

- **时钟频率**: 目标 100MHz
- **AI 加速器**: 
  - CompactAccel: 8x8 矩阵乘法
  - BitNetAccel: 1-bit 权重神经网络加速
- **存储**: 
  - Flash: 16MB (SPI)
  - PSRAM: 8MB (Quad-SPI)

## 参考资料

- [ysyxSoC 文档](https://ysyx.oscc.cc/)
- [PicoRV32 文档](https://github.com/YosysHQ/picorv32)
- [ICS55 PDK 文档](pdk/icsprout55-pdk/README.md)
- [Yosys 文档](https://yosyshq.net/yosys/)

## 版本历史

- v1.0 (2024-12): 初始版本，支持 AI 加速器集成
