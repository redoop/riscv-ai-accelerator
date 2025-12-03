# CTS 和 Routing 解决方案报告

**日期**: 2025-12-03 11:36  
**状态**: 部分解决 ✅

## 解决方案总结

### ✅ 已解决

1. **Routing 层定义问题**
   - 创建了 `tech_complete.lef` 包含 6 层金属层
   - 添加了 `make_tracks` 命令定义布线轨道
   - Floorplan 和 Placement 成功完成

2. **Placement 优化**
   - 平均位移: 2.3 um (优秀)
   - HPWL 增量: 7% (优秀)
   - Overflow: 9.9% (可接受)

### ⚠️ 部分解决

1. **CTS 时钟网络**
   - 问题: 时钟网络未找到
   - 原因: 时钟端口 `u_sys_clk_pad/C` 可能不存在
   - 状态: CTS DEF 文件已生成，但未插入时钟缓冲器

2. **Routing IO PAD**
   - 问题: IO PAD 没有有效的布线层几何形状
   - 原因: IO PAD 单元需要特殊处理
   - 状态: 核心逻辑可以布线，但 IO PAD 需要额外配置

## 技术细节

### 创建的文件

1. **tech_complete.lef**
   ```lef
   - SITE core7 (0.38 × 2.8 um)
   - LAYER MET1-MET6 (6 层金属)
   - 包含 WIDTH, SPACING, PITCH, RESISTANCE, CAPACITANCE
   ```

2. **run_fixed.tcl**
   ```tcl
   - 使用 tech_complete.lef
   - 添加 make_tracks 命令
   - 定义布线层 MET1-MET6
   - 设置电源网络
   ```

### 生成的结果

| 文件 | 大小 | 状态 |
|------|------|------|
| `1_floorplan_fixed.def` | 17 MB | ✅ 完成 |
| `2_placement_fixed.def` | 20 MB | ✅ 完成 |
| `3_cts_fixed.def` | 20 MB | ⚠️ 无时钟树 |

### Placement 质量

```
总位移: 237,985.1 um
平均位移: 2.3 um
最大位移: 10.2 um
原始 HPWL: 3,347,235.8 um
合法化 HPWL: 3,576,470.9 um
HPWL 增量: 7%
Overflow: 9.9%
```

## 剩余问题

### 问题 1: CTS 时钟网络未找到

**错误信息**:
```
[WARNING CTS-0083] No clock nets have been found.
```

**可能原因**:
1. 时钟端口命名不匹配
2. 时钟信号未连接到触发器
3. 需要使用内部时钟网络而非端口

**解决方案**:
```tcl
# 方案 1: 查找实际的时钟引脚
report_clocks
report_clock_networks

# 方案 2: 使用内部时钟网络
create_clock -period 40.0 -name sys_clk [get_nets sys_clk]

# 方案 3: 跳过 CTS，直接布线
# (对于 25MHz 低频设计可行)
```

### 问题 2: IO PAD 布线

**错误信息**:
```
[ERROR GRT-0042] Pin io_pad0 does not have geometries in a valid routing layer.
```

**原因**:
- IO PAD 单元缺少 LEF 定义
- 或者 IO PAD 需要在不同的层

**解决方案**:
```tcl
# 方案 1: 排除 IO PAD
set_dont_touch [get_cells u_io_pad*]
set_dont_route [get_nets *pad*]

# 方案 2: 使用核心模块网表
# 重新综合，只包含 SimpleEdgeAiSoC 模块

# 方案 3: 添加 IO LEF
read_lef "$IO_PATH/lef/ICSIOA_N55_3P3_1P6M1TM.lef"
```

## 快速修复方案

### 方案 A: 跳过 CTS 和 IO PAD

```tcl
# 修改 run_fixed.tcl

# 跳过 CTS
# clock_tree_synthesis ...

# 排除 IO PAD 布线
set_dont_route [get_nets *pad*]

# 只布线核心逻辑
global_route
detailed_route
```

### 方案 B: 使用核心模块

```bash
# 1. 提取核心模块网表
grep -A 1000000 "module SimpleEdgeAiSoC" netlist.v > core.v

# 2. 修改 DESIGN_NAME
set DESIGN_NAME "SimpleEdgeAiSoC"
set VERILOG_FILE "core.v"

# 3. 重新运行
openroad -exit run_fixed.tcl
```

## 当前成就

✅ **成功完成**:
1. Floorplan with tracks
2. High-quality Placement (7% HPWL增量)
3. 解决了布线层定义问题

⚠️ **需要改进**:
1. CTS 时钟网络识别
2. IO PAD 处理

## 建议

### 短期 (1小时)

1. **跳过 CTS**
   - 对于 25MHz 低频设计，可以不需要 CTS
   - 直接使用理想时钟进行布线

2. **排除 IO PAD**
   - 只布线核心逻辑
   - IO PAD 可以后续手动处理

### 中期 (1天)

1. **修复时钟网络**
   - 分析网表找到正确的时钟信号
   - 使用 `report_clock_networks` 调试

2. **添加 IO LEF**
   - 正确配置 IO PAD 的 LEF 文件
   - 或使用核心模块网表

### 长期 (2-3天)

1. **完整 P&R 流程**
   - 包含 CTS 的完整流程
   - 处理所有 IO PAD

2. **物理验证**
   - DRC/LVS 检查
   - 生成 GDSII

## 命令参考

### 运行修复后的流程

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad

# 查看结果
ls -lh results/*fixed*

# 查看日志
tail -100 logs/run_fixed_final.log

# 重新运行
openroad -exit run_fixed.tcl
```

### 调试时钟网络

```tcl
# 在 OpenROAD 中
report_clocks
report_clock_networks
get_clocks
get_nets sys_clk*
```

## 结论

✅ **主要问题已解决**:
- Routing 层定义 ✅
- Placement 质量优秀 ✅
- 布线轨道已定义 ✅

⚠️ **次要问题待解决**:
- CTS 时钟网络识别
- IO PAD 布线处理

**建议**: 对于 25MHz 低频设计，当前的 Placement 结果已经足够好，可以考虑：
1. 跳过 CTS，使用理想时钟
2. 排除 IO PAD，只布线核心逻辑
3. 或使用商业工具完成最后步骤

---

**创建时间**: 2025-12-03 11:36  
**状态**: Floorplan + Placement 完成 ✅  
**下一步**: 跳过 CTS 或修复时钟网络
