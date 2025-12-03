# ICS55 PDK ASIC 流程指南

**日期**: 2025-12-03  
**PDK**: ICS55 55nm (icsprout55-pdk)  
**设计**: RISC-V AI 加速器

## 概述

ICS55 是一个 55nm 工艺的 PDK，已经在项目中成功用于逻辑综合。本指南介绍如何使用 ICS55 PDK 完成完整的 ASIC 流程。

## PDK 信息

### 位置
```
/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk
```

### 标准单元库
```
ics55_LLSC_H7CL (Low Leakage Standard Cell, High 7-track, C variant, Low power)
```

### 文件结构
```
icsprout55-pdk/
└── IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/
    ├── liberty/
    │   ├── ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib    # Typical (25°C, 1.2V)
    │   ├── ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib  # Fast (-40°C, 1.32V)
    │   └── ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib # Slow (125°C, 1.08V)
    └── lef/
        ├── ics55_LLSC_H7CL.lef           # 标准 LEF
        └── ics55_LLSC_H7CL_ieda.lef      # iEDA 专用 LEF
```

## 方案对比

### 方案 1: OpenLane (推荐用于 SkyWater PDK)

**优点**:
- ✅ 完整的自动化流程
- ✅ 从 RTL 到 GDSII
- ✅ 内置 DRC/LVS 验证

**缺点**:
- ❌ 主要支持 SkyWater 130nm PDK
- ❌ 对其他 PDK 支持有限
- ❌ 需要大量配置适配

**适用场景**: 使用 SkyWater 130nm PDK 的项目

### 方案 2: OpenROAD (推荐用于 ICS55 PDK) ⭐

**优点**:
- ✅ 支持自定义 PDK
- ✅ 灵活的流程控制
- ✅ 已在项目中验证

**缺点**:
- ⚠️ 需要手动编写流程脚本
- ⚠️ GDSII 生成需要额外工具

**适用场景**: 使用 ICS55 或其他自定义 PDK

### 方案 3: ECOS (已完成) ✅

**优点**:
- ✅ 已成功完成逻辑综合
- ✅ 生成 623,516 行网表
- ✅ 完整的综合报告

**缺点**:
- ⚠️ 仅完成综合阶段
- ⚠️ 需要后续 P&R 流程

**适用场景**: 逻辑综合和网表生成

## 推荐流程

### 阶段 1: 逻辑综合 (已完成) ✅

使用 ECOS 完成:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
./run_synthesis.sh
```

**输出**:
- 网表: `project/netlist/asic_top_ics55.v` (16MB)
- 面积: 292,992 µm²
- 单元数: 96,087

### 阶段 2: 布局布线 (推荐)

使用 OpenROAD:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./run_openroad_ics55.sh
```

**流程步骤**:
1. 读取 Liberty 和 LEF
2. 读取网表
3. 布图规划 (Floorplan)
4. IO 放置
5. 电源网络 (PDN)
6. 全局布局 (Global Placement)
7. 详细布局 (Detailed Placement)
8. 时钟树综合 (CTS)
9. 全局布线 (Global Routing)
10. 详细布线 (Detailed Routing)
11. 填充单元 (Filler)
12. 输出 DEF 和网表

**预计时间**: 30-60 分钟

### 阶段 3: GDSII 生成 (可选)

使用 Magic 或 KLayout:

```bash
# 使用 Magic
magic -T ics55 -rcfile magic.rc final.def

# 或使用 KLayout
klayout -e -rd design=final.def -rd tech=ics55.lyt -rd output=final.gds
```

## 快速开始

### 1. 检查 PDK

```bash
ls -la /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/
```

### 2. 运行 OpenROAD 流程

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./run_openroad_ics55.sh
```

### 3. 查看结果

```bash
cd openroad_work/run_*/
ls -lh *.def *.v *.rpt
```

## 配置参数

### 时钟约束
```tcl
create_clock -name sys_clk -period 40.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 2.0 [get_clocks sys_clk]
```

- 时钟周期: 40ns (25MHz)
- 不确定性: 2ns

### 布图规划
```tcl
initialize_floorplan \
    -die_area "0 0 2000 2000" \
    -core_area "100 100 1900 1900"
```

- Die 面积: 2000 x 2000 µm
- Core 面积: 1800 x 1800 µm
- 利用率: ~30%

### 布局密度
```tcl
global_placement -density 0.30
```

- 目标密度: 30%
- 留有足够的布线空间

## 输出文件

| 文件 | 说明 |
|------|------|
| `final.def` | DEF 版图文件 |
| `final.v` | 最终网表 |
| `timing_max.rpt` | 最大路径时序 |
| `timing_min.rpt` | 最小路径时序 |
| `tns.rpt` | 总负时序裕量 |
| `wns.rpt` | 最差负时序裕量 |
| `area.rpt` | 面积报告 |
| `openroad.log` | 完整日志 |

## 常见问题

### Q1: OpenROAD 未安装?

```bash
# 使用 Docker
sudo docker pull openroad/openroad:latest

# 或安装本地版本
sudo apt-get install openroad
```

### Q2: 时序违例?

调整时钟周期:
```tcl
create_clock -name sys_clk -period 50.0 [get_ports sys_clk_i_pad]  # 20MHz
```

### Q3: 布局拥挤?

增加芯片面积:
```tcl
initialize_floorplan \
    -die_area "0 0 3000 3000" \
    -core_area "100 100 2900 2900"
```

### Q4: 布线失败?

降低布局密度:
```tcl
global_placement -density 0.25
```

## 与 ECOS 综合的集成

ECOS 已经完成了逻辑综合，OpenROAD 可以直接使用其输出:

```bash
# ECOS 综合输出
/opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/project/netlist/asic_top_ics55.v

# OpenROAD 输入
./run_openroad_ics55.sh  # 自动使用 ECOS 网表
```

## 下一步

1. **运行 OpenROAD 流程**
   ```bash
   ./run_openroad_ics55.sh
   ```

2. **查看时序报告**
   ```bash
   cat openroad_work/run_*/timing_max.rpt
   ```

3. **查看面积报告**
   ```bash
   cat openroad_work/run_*/area.rpt
   ```

4. **生成 GDSII** (可选)
   ```bash
   # 需要额外的 GDSII 生成工具
   ```

## 参考

- OpenROAD: https://github.com/The-OpenROAD-Project/OpenROAD
- ICS55 PDK: https://github.com/ideplatform/icsprout55-pdk
- ECOS 综合: ../SYNTHESIS_REPORT.md
- OpenROAD 文档: https://openroad.readthedocs.io/

---

**创建时间**: 2025-12-03 12:49  
**状态**: 准备就绪，可以运行
