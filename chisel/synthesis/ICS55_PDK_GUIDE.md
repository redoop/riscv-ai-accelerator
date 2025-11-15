# ICS55 PDK 使用指南

本指南介绍如何使用 ICS55 PDK 进行逻辑综合和后端仿真。

## 📋 目录

- [PDK 简介](#pdk-简介)
- [安装 PDK](#安装-pdk)
- [逻辑综合](#逻辑综合)
- [后端仿真](#后端仿真)
- [故障排除](#故障排除)

## PDK 简介

**ICS55 PDK** 是一个开源的工艺设计套件（Process Design Kit），提供了标准单元库和相关的设计文件。

### 主要特性

- **工艺节点**: 55nm
- **标准单元库**: 包含基本逻辑门、触发器等
- **文件格式**: Liberty (.lib), Verilog (.v)
- **开源**: 可在 GitHub 上获取

### PDK 结构

```
icsprout55-pdk/
├── IP/
│   └── STD_cell/
│       └── ics55_LLSC_H7C_V1p10C100/
│           ├── ics55_LLSC_H7CL/        # Low 标准单元库
│           │   ├── liberty/            # Liberty 时序库
│           │   │   ├── *_typ_tt_1p2_25_nldm.lib  # 典型角度
│           │   │   ├── *_ff_*.lib      # 快速角度
│           │   │   └── *_ss_*.lib      # 慢速角度
│           │   └── verilog/            # Verilog 行为模型
│           │       └── ics55_LLSC_H7CL.v
│           ├── ics55_LLSC_H7CH/        # High 标准单元库
│           └── ics55_LLSC_H7CR/        # Regular 标准单元库
├── prtech/                             # 工艺技术文件
└── README.md
```

**标准单元库说明:**
- **H7CL (Low)**: 低功耗单元，驱动能力较弱
- **H7CH (High)**: 高性能单元，驱动能力强
- **H7CR (Regular)**: 常规单元，平衡性能和功耗

## 安装 PDK

### 方法 1: 使用提供的脚本（推荐）

```bash
cd chisel/synthesis/pdk
python get_ics55_pdk.py
```

脚本会自动从 GitHub 克隆 PDK 仓库：
```python
#!/bin/python

import os

os.system('git clone --recursive git@github.com:IDE-Platform/icsprout55-pdk.git')
```

### 方法 2: 手动克隆

```bash
cd chisel/synthesis/pdk
git clone --recursive git@github.com:IDE-Platform/icsprout55-pdk.git
```

### 验证安装

检查 PDK 文件是否存在：

```bash
ls -la pdk/icsprout55-pdk/lib/ics55_stdcell_typ.lib
ls -la pdk/icsprout55-pdk/verilog/ics55_stdcell.v
```

## 逻辑综合

### 前提条件

1. **RTL 设计**: 确保已生成 Chisel RTL
   ```bash
   cd chisel
   make verilog
   ```

2. **Yosys**: 确保已安装 Yosys 综合工具
   ```bash
   /opt/tools/oss-cad/oss-cad-suite/bin/yosys --version
   ```

3. **ICS55 PDK**: 确保已安装 PDK（见上节）

### 运行综合

使用提供的综合脚本：

```bash
cd chisel/synthesis
./run_ics55_synthesis.sh
```

### 综合流程

脚本会执行以下步骤：

1. **检查文件**: 验证 RTL 和 PDK 文件存在
2. **读取 RTL**: 使用 Slang 插件读取 SystemVerilog
3. **综合**: 执行逻辑综合
4. **映射**: 映射到 ICS55 标准单元
5. **优化**: ABC 优化
6. **输出**: 生成网表文件

### 输出文件

综合成功后会生成：

```
netlist/
├── SimpleEdgeAiSoC_ics55.v      # 综合网表
├── ics55_stdcell.v              # 标准单元模型（复制）
├── synthesis_ics55.log          # 综合日志
└── synthesis_stats_ics55.txt    # 统计信息
```

### 查看综合结果

```bash
# 查看网表
cat netlist/SimpleEdgeAiSoC_ics55.v

# 查看统计
cat netlist/synthesis_stats_ics55.txt

# 查看日志
cat netlist/synthesis_ics55.log
```

### 综合统计示例

```
=== SimpleEdgeAiSoC ===

   Number of wires:               XXXX
   Number of wire bits:           XXXX
   Number of public wires:        XXXX
   Number of public wire bits:    XXXX
   Number of memories:            X
   Number of memory bits:         XXXX
   Number of processes:           X
   Number of cells:               XXXX
     ICS55_AND2X1                 XXX
     ICS55_NAND2X1                XXX
     ICS55_OR2X1                  XXX
     ICS55_NOR2X1                 XXX
     ICS55_DFFX1                  XXX
     ...
```

## 后端仿真

### 使用 Icarus Verilog 仿真

综合完成后，可以使用 Icarus Verilog 进行后端仿真：

```bash
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

### 仿真流程

1. **编译**: 编译网表和标准单元库
2. **仿真**: 运行测试平台
3. **验证**: 检查功能正确性

### 查看仿真结果

仿真输出会显示：
- 测试进度
- 功能验证结果
- 错误信息（如果有）

### 波形查看

如果生成了波形文件：

```bash
# 使用 GTKWave
gtkwave waves/post_syn.vcd

# 或在仿真脚本中添加波形生成
python run_post_syn_sim.py --simulator iverilog --netlist ics55 --wave
```

## 与其他 PDK 对比

### IHP SG13G2 vs ICS55

| 特性 | IHP SG13G2 | ICS55 |
|------|------------|-------|
| 工艺节点 | 130nm | 55nm |
| 开源 | ✓ | ✓ |
| 标准单元 | 完整 | 完整 |
| 文档 | 详细 | 基本 |
| 社区支持 | 活跃 | 发展中 |

### 综合命令对比

```bash
# IHP PDK
./run_ihp_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ihp

# ICS55 PDK
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# 通用综合
./run_generic_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist generic
```

## 故障排除

### 问题 1: PDK 未找到

**错误信息:**
```
错误: 未找到 ICS55 PDK
请运行: python pdk/get_ics55_pdk.py
```

**解决方法:**
```bash
cd chisel/synthesis/pdk
python get_ics55_pdk.py
```

### 问题 2: Liberty 文件不存在

**错误信息:**
```
错误: 未找到 ICS55 PDK Liberty 文件
```

**解决方法:**
1. 检查 PDK 目录结构：
   ```bash
   ls -la pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/
   ```
2. 确认 Liberty 文件存在：
   ```bash
   ls -la pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
   ```
3. 如果路径不同，修改 `run_ics55_synthesis.sh` 中的 `LIBERTY_FILE` 变量

### 问题 3: Verilog 模型不存在

**错误信息:**
```
错误: 未找到 ICS55 PDK Verilog 模型
```

**解决方法:**
1. 检查 PDK 目录结构：
   ```bash
   ls -la pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/
   ```
2. 确认 Verilog 文件存在：
   ```bash
   ls -la pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v
   ```
3. 如果路径不同，修改 `run_ics55_synthesis.sh` 中的 `VERILOG_MODEL` 变量

### 问题 4: RTL 文件未生成

**错误信息:**
```
错误: 未找到 RTL 文件: ../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

**解决方法:**
```bash
cd chisel
make verilog
```

### 问题 5: Yosys 未安装

**错误信息:**
```
/opt/tools/oss-cad/oss-cad-suite/bin/yosys: No such file or directory
```

**解决方法:**
1. 安装 OSS CAD Suite
2. 或修改脚本中的 `YOSYS_BIN` 路径指向你的 Yosys 安装位置

### 问题 6: 综合失败

**解决方法:**
1. 查看综合日志：
   ```bash
   cat netlist/synthesis_ics55.log
   ```
2. 检查 RTL 语法
3. 确认 PDK 文件完整性
4. 尝试使用通用综合验证 RTL：
   ```bash
   ./run_generic_synthesis.sh
   ```

### 问题 7: 仿真失败

**解决方法:**
1. 确认网表文件存在：
   ```bash
   ls -la netlist/SimpleEdgeAiSoC_ics55.v
   ```
2. 确认标准单元库存在：
   ```bash
   ls -la netlist/ics55_stdcell.v
   ```
3. 检查测试平台：
   ```bash
   ls -la testbench/post_syn_tb.sv
   ```

## 高级用法

### 自定义综合参数

编辑 `run_ics55_synthesis.sh`，修改 Yosys 脚本：

```bash
# 添加更多优化
abc -liberty $LIBERTY_FILE -D 1000

# 保留层次结构
hierarchy -top SimpleEdgeAiSoC -keep

# 输出更详细的统计
stat -liberty $LIBERTY_FILE -width
```

### 多角度综合

为不同的工艺角度（corner）综合：

```bash
# 典型角度（typical, tt, 1.2V, 25°C）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"

# 快速角度（fast, ff, 1.32V, -40°C）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib"

# 慢速角度（slow, ss, 1.08V, 125°C）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib"
```

### 使用不同的标准单元库

```bash
# Low 库（低功耗）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"

# High 库（高性能）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CH/liberty/ics55_LLSC_H7CH_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CH/verilog/ics55_LLSC_H7CH.v"

# Regular 库（平衡）
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/verilog/ics55_LLSC_H7CR.v"
```

### 时序约束

添加时序约束文件（SDC）：

```tcl
# constraints.sdc
create_clock -period 10 [get_ports clock]
set_input_delay -clock clock 2 [all_inputs]
set_output_delay -clock clock 2 [all_outputs]
```

在综合脚本中使用：
```bash
read_sdc constraints.sdc
```

## 参考资料

### ICS55 PDK
- GitHub 仓库: https://github.com/IDE-Platform/icsprout55-pdk
- 文档: 查看 PDK 仓库中的 doc/ 目录

### Yosys
- 官方网站: https://yosyshq.net/yosys/
- 文档: https://yosyshq.readthedocs.io/

### Icarus Verilog
- 官方网站: http://iverilog.icarus.com/
- 文档: http://iverilog.wikia.com/

## 快速参考

### 完整流程

```bash
# 1. 安装 PDK
cd chisel/synthesis/pdk
python get_ics55_pdk.py

# 2. 生成 RTL
cd ../..
make verilog

# 3. 逻辑综合
cd synthesis
./run_ics55_synthesis.sh

# 4. 后端仿真
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# 5. 查看结果
cat netlist/synthesis_stats_ics55.txt
```

### 常用命令

```bash
# 查看 PDK 信息
ls -la pdk/icsprout55-pdk/

# 查看网表
cat netlist/SimpleEdgeAiSoC_ics55.v

# 查看综合统计
cat netlist/synthesis_stats_ics55.txt

# 查看综合日志
cat netlist/synthesis_ics55.log

# 重新综合
./run_ics55_synthesis.sh

# 重新仿真
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

---

**注意**: 本指南假设使用标准的 ICS55 PDK 目录结构。如果你的 PDK 结构不同，请相应调整脚本中的路径。
