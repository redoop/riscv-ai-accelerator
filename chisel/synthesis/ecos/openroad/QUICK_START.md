# OpenROAD P&R 快速开始

## 当前状态

OpenROAD P&R 流程已配置，但遇到 floorplan 步骤的问题。

## 快速命令

```bash
# 进入目录
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad

# 查看状态
cat STATUS.md

# 查看日志
tail -100 logs/pnr_run.log

# 运行 P&R（当前会失败）
./run.sh all
```

## 问题诊断

### 错误信息
```
[ERROR IFP-0011] use -site to add placement rows.
```

### 原因
1. LEF 文件中 SITE 定义不完整
2. 网表包含 IO PAD，但 LEF 未正确加载
3. 68 个 latch 单元 + 86 个 IO PAD 单元缺失

## 解决方案

### 方案 1: 使用核心网表（推荐）

提取不含 IO PAD 的核心模块：

```bash
# 1. 生成核心模块网表
cd /opt/github/riscv-ai-accelerator/chisel
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 2. 使用 Yosys 综合核心模块
cd synthesis/ecos
# 修改综合脚本，只综合 SimpleEdgeAiSoC 模块

# 3. 运行 P&R
cd openroad
# 修改 config.tcl:
#   set DESIGN_NAME "SimpleEdgeAiSoC"
#   set VERILOG_FILE "path/to/core_netlist.v"
./run.sh all
```

### 方案 2: 修复当前配置

添加 IO LEF 和处理 latch：

```tcl
# 在 run_pnr.tcl 中添加:
read_lef "$IO_PATH/lef/ICSIOA_N55_3P3_1P6M1TM.lef"

# 处理 latch 单元:
# 选项 1: 在综合时禁用 latch
# 选项 2: 添加 latch 的 LEF 定义
```

### 方案 3: 使用简化流程

降低复杂度，先验证基本流程：

```tcl
# 修改 config.tcl:
set CLOCK_PERIOD 40.0  # 25MHz
set CORE_UTILIZATION 0.5  # 50% 利用率

# 简化 CTS 配置:
set CTS_BUF_LIST "BUFX4H7L"  # 只用一种缓冲器
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.tcl` | 主配置文件 |
| `run_pnr.tcl` | 完整 P&R 流程脚本 |
| `run.sh` | 运行脚本 |
| `STATUS.md` | 详细状态报告 |
| `logs/` | 运行日志目录 |
| `results/` | 输出结果目录 |

## 预期输出

成功运行后会生成：

```
results/
├── final.def          # DEF 布局文件
├── final.v            # 布线后网表
├── 1_floorplan.def    # Floorplan 结果
├── 2_placement.def    # Placement 结果
├── 3_cts.def          # CTS 结果
└── 4_routing.def      # Routing 结果
```

## 时序目标

| 参数 | 目标值 |
|------|--------|
| 频率 | 25 MHz |
| 周期 | 40 ns |
| Setup Slack | > 0 ns |
| Hold Slack | > 0 ns |
| Clock Skew | < 1 ns |

## 帮助

查看详细文档：
- 状态报告: `cat STATUS.md`
- OpenROAD 文档: https://openroad.readthedocs.io/
- 项目 README: `/opt/github/riscv-ai-accelerator/README.md`

---

**创建时间**: 2025-12-03 11:14
