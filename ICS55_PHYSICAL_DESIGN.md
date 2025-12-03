# ICS55 物理设计实施方案

**日期**: 2025-12-03  
**版本**: v0.4.1  
**工艺**: ICS55 55nm  
**PDK**: icsprout55-pdk (完整)  
**状态**: ✅ 准备就绪

---

## 📦 ICS55 PDK 资源

### 可用文件

| 类型 | 路径 | 状态 |
|------|------|------|
| **Liberty 文件** | `pdk/icsprout55-pdk/.../liberty/` | ✅ 7 个文件 (543 MB) |
| **LEF 文件** | `pdk/icsprout55-pdk/.../lef/` | ✅ 3 个文件 |
| **Verilog 模型** | `pdk/icsprout55-pdk/.../verilog/` | ✅ 可用 |
| **CDL 网表** | `pdk/icsprout55-pdk/.../cdl/` | ✅ 可用 |
| **文档** | `pdk/icsprout55-pdk/.../doc/` | ✅ 可用 |

### Liberty 文件详情

| 文件 | 工艺角 | 电压 | 温度 | 用途 |
|------|--------|------|------|------|
| `ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib` | TT | 1.2V | 25°C | **典型** (STA) |
| `ics55_LLSC_H7CL_ss_rcworst_1p08_125_nldm.lib` | SS | 1.08V | 125°C | 最坏 (Setup) |
| `ics55_LLSC_H7CL_ff_rcbest_1p32_m40_nldm.lib` | FF | 1.32V | -40°C | 最好 (Hold) |
| `ics55_LLSC_H7CL_ss_cworst_1p08_m40_nldm.lib` | SS | 1.08V | -40°C | 最坏 C |
| `ics55_LLSC_H7CL_ss_rcworst_1p2_m40_nldm.lib` | SS | 1.2V | -40°C | 最坏 RC |
| `ics55_LLSC_H7CL_ff_cbest_1p32_125_nldm.lib` | FF | 1.32V | 125°C | 最好 C |
| `ics55_LLSC_H7CL_ff_rcbest_1p08_125_nldm.lib` | FF | 1.08V | 125°C | 最好 RC |

**推荐使用**:
- **Setup 分析**: `ss_rcworst_1p08_125` (最坏情况)
- **Hold 分析**: `ff_rcbest_1p32_m40` (最好情况)
- **典型分析**: `typ_tt_1p2_25` (标称条件)

### LEF 文件

| 文件 | 用途 |
|------|------|
| `ics55_LLSC_H7CL.lef` | 标准单元 LEF |
| `ics55_LLSC_H7CL_ieda.lef` | iEDA 工具专用 |
| `ics55_LLSC_H7CL_ant.lef` | 天线效应规则 |

---

## 🔧 物理设计流程

### 完整流程

```
网表 → STA → 布局规划 → 布局 → CTS → 布线 → 优化 → 验证 → GDSII
```

### 工具选择

| 阶段 | 工具 | 状态 |
|------|------|------|
| **STA** | OpenSTA | ✅ 可用 |
| **P&R** | OpenROAD | ✅ 可用 |
| **验证** | Magic/Netgen | ⚠️ 待确认 |

---

## 📊 设计参数

### 芯片规格

| 参数 | 数值 |
|------|------|
| **工艺** | ICS55 55nm |
| **电源电压** | 1.2V (标称) |
| **工作温度** | -40°C ~ 125°C |
| **芯片面积** | 0.33 mm² (综合结果) |
| **目标面积** | 600 × 600 μm (0.36 mm²) |
| **核心面积** | 500 × 500 μm (0.25 mm²) |
| **IO Pads** | 61 个 |

### 时序约束

```tcl
# 主时钟 100 MHz
create_clock -name clk -period 10.0 [get_ports clock]

# SPI 时钟 10 MHz (10 分频)
create_generated_clock -name spi_clk \
    -source [get_ports clock] \
    -divide_by 10 \
    [get_pins lcd/spi_clk_reg/Q]

# 输入延迟 (20% 周期)
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# 输出延迟 (20% 周期)
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# 时钟不确定性 (5%)
set_clock_uncertainty 0.5 [all_clocks]

# 时钟转换时间
set_clock_transition 0.1 [all_clocks]

# 负载
set_load 0.05 [all_outputs]
```

### 物理约束

```tcl
# 芯片尺寸
set die_width 600.0
set die_height 600.0

# 核心区域 (留出 IO 环)
set core_margin_left 50.0
set core_margin_right 50.0
set core_margin_top 50.0
set core_margin_bottom 50.0

# 布局密度
set target_density 0.70

# 电源网格
set power_net VDD
set ground_net VSS
set power_stripe_width 2.0
set power_stripe_pitch 20.0
set power_stripe_offset 10.0
```

---

## 🚀 实施步骤

### Step 1: 静态时序分析 (STA)

**目标**: 验证时序收敛

**脚本**: `run_sta_ics55.tcl`

```tcl
# 读取 Liberty 文件
read_liberty pdk/icsprout55-pdk/.../liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog synthesis/netlist/SimpleEdgeAiSoC_ics55.v

# 链接设计
link_design ip1_SimpleEdgeAiSoC

# 读取 SDC 约束
read_sdc synthesis/sdc/timing.sdc

# 报告时序
report_checks -path_delay min_max -format full_clock_expanded
report_tns
report_wns
report_worst_slack

# 保存报告
report_checks > reports/timing_report.txt
```

**执行**:
```bash
cd chisel/synthesis/sta
sta run_sta_ics55.tcl
```

**验收标准**:
- WNS (Worst Negative Slack) > 0
- TNS (Total Negative Slack) = 0
- Setup slack > 1ns
- Hold slack > 0.2ns

### Step 2: 布局规划 (Floorplanning)

**目标**: 规划芯片布局

**OpenROAD 脚本**: `run_floorplan.tcl`

```tcl
# 读取 LEF
read_lef pdk/icsprout55-pdk/.../lef/ics55_LLSC_H7CL.lef

# 读取 Liberty
read_liberty pdk/icsprout55-pdk/.../liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog synthesis/netlist/SimpleEdgeAiSoC_ics55.v
link_design ip1_SimpleEdgeAiSoC

# 初始化布局规划
initialize_floorplan \
    -die_area "0 0 600 600" \
    -core_area "50 50 550 550" \
    -site unit

# 放置 IO Pads (61 个)
# 北侧: 电源、时钟、复位
place_pin -pin_name clock -layer metal3 -location "300 0" -force_to_die_boundary
place_pin -pin_name reset -layer metal3 -location "320 0" -force_to_die_boundary

# 东侧: UART, LCD
place_pin -pin_name io_uart_tx -layer metal4 -location "600 200" -force_to_die_boundary
place_pin -pin_name io_uart_rx -layer metal4 -location "600 220" -force_to_die_boundary

# 南侧: GPIO
# ... (GPIO 16 个引脚)

# 西侧: Flash, PSRAM
# ... (Flash 4 个, PSRAM 10 个引脚)

# 电源规划
add_global_connection -net VDD -pin_pattern {^VDD$} -power
add_global_connection -net VSS -pin_pattern {^VSS$} -ground

# 电源网格
pdngen

# 保存
write_def floorplan.def
```

**执行**:
```bash
cd chisel/synthesis/physical_design
openroad -exit run_floorplan.tcl
```

### Step 3: 布局 (Placement)

**目标**: 放置标准单元

```tcl
# 读取布局规划
read_def floorplan.def

# 全局布局
global_placement -density $target_density

# 详细布局
detailed_placement

# 优化
optimize_placement

# 保存
write_def placement.def
```

### Step 4: 时钟树综合 (CTS)

**目标**: 生成时钟树

```tcl
# 读取布局
read_def placement.def

# 时钟树综合
clock_tree_synthesis \
    -root_buf BUFX4 \
    -buf_list "BUFX2 BUFX4 BUFX8" \
    -wire_unit 20

# 优化时钟偏斜
repair_clock_nets

# 保存
write_def cts.def
```

### Step 5: 布线 (Routing)

**目标**: 完成信号布线

```tcl
# 读取 CTS 结果
read_def cts.def

# 全局布线
global_route

# 详细布线
detailed_route

# 修复天线效应
repair_antennas

# 保存
write_def routed.def
write_verilog routed.v
```

### Step 6: 优化

**目标**: 时序和功耗优化

```tcl
# 读取布线结果
read_def routed.def

# 时序优化
repair_timing -setup -hold

# 功耗优化
repair_design -max_wire_length 500

# 保存
write_def optimized.def
```

### Step 7: 物理验证

**目标**: DRC/LVS 检查

```bash
# DRC (需要 Magic)
magic -dnull -noconsole << EOF
tech load ics55
gds read optimized.gds
drc check
drc catchup
drc count
quit
EOF

# LVS (需要 Netgen)
netgen -batch lvs \
    optimized.spice \
    synthesis/netlist/SimpleEdgeAiSoC_ics55.v \
    ics55_setup.tcl \
    lvs_report.txt
```

### Step 8: GDSII 生成

**目标**: 生成流片数据

```tcl
# 读取最终 DEF
read_def optimized.def

# 生成 GDSII
write_gds SimpleEdgeAiSoC_v0.4.1.gds

# 生成最终网表
write_verilog SimpleEdgeAiSoC_final.v

# 生成 SPICE
write_spice SimpleEdgeAiSoC.spice
```

---

## 📝 执行脚本

### 主控脚本

创建 `run_physical_design.sh`:

```bash
#!/bin/bash

set -e

echo "=== ICS55 Physical Design Flow ==="

# 设置环境变量
export PDK_ROOT=/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk
export PDK_PATH=$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL

# 创建输出目录
mkdir -p physical_design/{reports,results,logs}

# Step 1: STA
echo "Step 1: Static Timing Analysis..."
cd sta
sta run_sta_ics55.tcl | tee ../physical_design/logs/sta.log
cd ..

# Step 2: Floorplanning
echo "Step 2: Floorplanning..."
cd physical_design
openroad -exit run_floorplan.tcl | tee logs/floorplan.log

# Step 3: Placement
echo "Step 3: Placement..."
openroad -exit run_placement.tcl | tee logs/placement.log

# Step 4: CTS
echo "Step 4: Clock Tree Synthesis..."
openroad -exit run_cts.tcl | tee logs/cts.log

# Step 5: Routing
echo "Step 5: Routing..."
openroad -exit run_routing.tcl | tee logs/routing.log

# Step 6: Optimization
echo "Step 6: Optimization..."
openroad -exit run_optimization.tcl | tee logs/optimization.log

# Step 7: Verification
echo "Step 7: Physical Verification..."
./run_verification.sh | tee logs/verification.log

# Step 8: GDSII Generation
echo "Step 8: GDSII Generation..."
openroad -exit run_gdsii.tcl | tee logs/gdsii.log

echo ""
echo "=== Physical Design Complete ==="
echo "Results in physical_design/results/"
ls -lh physical_design/results/
```

---

## ✅ 验收标准

### 时序

| 指标 | 目标 | 说明 |
|------|------|------|
| Setup WNS | > 0 ns | 无建立时间违例 |
| Setup TNS | = 0 ns | 总负时序为零 |
| Hold WNS | > 0 ns | 无保持时间违例 |
| Hold TNS | = 0 ns | 总负时序为零 |
| Clock Skew | < 100 ps | 时钟偏斜 |

### 物理

| 指标 | 目标 | 说明 |
|------|------|------|
| DRC Violations | 0 | 无设计规则违例 |
| LVS Errors | 0 | 版图与网表一致 |
| Antenna Violations | 0 | 无天线效应 |
| Routing Completion | 100% | 完全布线 |
| Core Utilization | 60-80% | 核心利用率 |

### 功耗

| 指标 | 目标 | 说明 |
|------|------|------|
| Static Power | < 1 mW | 静态功耗 |
| Dynamic Power | < 100 mW @ 100MHz | 动态功耗 |
| Total Power | < 100 mW | 总功耗 |

---

## 📚 交付物

### 最终文件

| 文件 | 说明 |
|------|------|
| `SimpleEdgeAiSoC_v0.4.1.gds` | GDSII 版图文件 |
| `SimpleEdgeAiSoC_final.v` | 最终网表 |
| `SimpleEdgeAiSoC.spice` | SPICE 网表 |
| `timing_report.txt` | 时序报告 |
| `drc_report.txt` | DRC 报告 |
| `lvs_report.txt` | LVS 报告 |
| `power_report.txt` | 功耗报告 |

### 文档

| 文档 | 说明 |
|------|------|
| `PHYSICAL_DESIGN_REPORT.md` | 物理设计报告 |
| `TIMING_ANALYSIS.md` | 时序分析报告 |
| `VERIFICATION_REPORT.md` | 验证报告 |
| `TAPEOUT_CHECKLIST.md` | 流片检查清单 |

---

## 🎯 下一步

### 立即执行

1. **创建 STA 脚本**
   ```bash
   cd chisel/synthesis/sta
   vim run_sta_ics55.tcl
   ```

2. **运行 STA**
   ```bash
   sta run_sta_ics55.tcl
   ```

3. **分析时序报告**
   - 检查 WNS/TNS
   - 识别关键路径
   - 评估时序裕量

### 后续步骤

4. 创建布局规划脚本
5. 执行完整 P&R 流程
6. 物理验证
7. 生成 GDSII

---

**创建日期**: 2025-12-03  
**PDK**: ICS55 55nm (完整)  
**状态**: ✅ 准备就绪，可立即开始
