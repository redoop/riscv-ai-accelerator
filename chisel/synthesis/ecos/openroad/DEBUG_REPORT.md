# 时钟网络和 IO PAD 调试报告

**日期**: 2025-12-03 11:47  
**状态**: 部分成功 ✅

## 调试结果总结

### ✅ 成功解决

1. **时钟网络识别**
   - 找到内部时钟网络: `sys_clk`
   - 连接到 25,556 个触发器
   - 时钟源: DFF 单元的 CK 引脚

2. **Placement 完成**
   - 平均位移: 2.3 um
   - HPWL 增量: 7%
   - Overflow: 9.9%

### ❌ 未解决

1. **CTS 时钟树**
   - 问题: `create_clock` 命令无法正确指定时钟源
   - 尝试: `[get_nets sys_clk]`, `[get_pins ...]`
   - 结果: CTS 仍然找不到时钟网络

2. **IO PAD Routing**
   - 问题: IO PAD 没有有效的布线层几何形状
   - 原因: IO PAD 单元需要特殊的 LEF 定义
   - 影响: 无法完成 global_route

## 技术细节

### 时钟网络分析

```
=== 时钟端口 ===
  sys_clk_i_pad (输入端口)
  sys_clk_o_pad (输出端口)

=== 时钟网络 ===
  sys_clk (pins: 25556) ← 主时钟网络
  sys_clk_i_pad (pins: 0)
  sys_clk_o_pad (pins: 0)

=== DFF 单元 ===
  数量: 25,556
  示例: _154729_
  时钟引脚: CK
  连接网络: sys_clk
```

### CTS 问题

**尝试的方法**:
1. `create_clock -period 40.0 sys_clk_i_pad` ✅ 可以创建
2. `create_clock -period 40.0 [get_nets sys_clk]` ❌ 类型错误
3. `create_clock -period 40.0 [get_pins u_sys_clk_pad/C]` ❌ 引脚不存在
4. `create_clock -period 40.0 [get_pins -of_objects [get_nets sys_clk] -filter "direction==out"]` ❌ CTS 仍找不到

**根本原因**:
- OpenROAD CTS 需要时钟源是一个明确的引脚或端口
- 内部网络 `sys_clk` 可能没有明确的驱动源
- 或者时钟网络的拓扑结构不符合 CTS 的要求

### IO PAD 问题

**错误信息**:
```
[ERROR GRT-0042] Pin io_pad0 does not have geometries in a valid routing layer.
```

**原因分析**:
1. IO PAD LEF 文件已加载，但可能不完整
2. IO PAD 单元的引脚定义在特殊层（如 RDL）
3. 这些层未在 tech LEF 中定义
4. 或者 IO PAD 需要在 floorplan 阶段特殊放置

## 当前成果

### 生成的文件

| 文件 | 大小 | 状态 | 说明 |
|------|------|------|------|
| `placement_simple.def` | 16 MB | ✅ | 高质量 placement |
| `1_final_floorplan.def` | 17 MB | ✅ | 带 tracks 的 floorplan |
| `2_final_placement.def` | 20 MB | ✅ | 完整 placement |
| `3_final_cts.def` | 20 MB | ⚠️ | 无时钟树 |

### Placement 质量

```
总位移: 237,985.1 um
平均位移: 2.3 um (优秀)
最大位移: 10.2 um
HPWL 增量: 7% (优秀)
Overflow: 9.9% (可接受)
```

## 解决方案建议

### 方案 1: 跳过 CTS (推荐)

对于 25MHz 低频设计，可以不需要 CTS:

```tcl
# 使用理想时钟
create_clock -period 40.0 sys_clk_i_pad
set_propagated_clock [all_clocks]  # 使用理想时钟传播
```

**优点**:
- 简化流程
- 25MHz 频率下时钟偏斜影响小
- 可以直接进行布线

**缺点**:
- 没有真实的时钟树
- 时序分析不够准确

### 方案 2: 使用核心模块网表

提取不含 IO PAD 的核心模块:

```bash
# 从网表中提取 SimpleEdgeAiSoC 模块
grep -A 1000000 "module SimpleEdgeAiSoC" asic_top_ics55.v > core.v

# 修改设计名称
set DESIGN_NAME "SimpleEdgeAiSoC"
set VERILOG_FILE "core.v"
```

**优点**:
- 避免 IO PAD 问题
- 可以完成完整的 P&R 流程
- 核心逻辑的布线可以完成

**缺点**:
- 需要重新处理 IO 连接
- 最终需要集成 IO PAD

### 方案 3: 手动 CTS

使用 OpenSTA 进行时钟树分析:

```tcl
# 读取 placement 结果
read_def placement_simple.def

# 使用 OpenSTA 分析时钟
report_clock_properties
report_clock_skew

# 手动插入时钟缓冲器
# (需要编写脚本)
```

### 方案 4: 使用商业工具

对于复杂的 IO PAD 和 CTS，建议使用商业 EDA 工具:
- Cadence Innovus
- Synopsys ICC2
- Mentor Calibre

## 结论

✅ **主要成就**:
1. 成功识别时钟网络 (sys_clk, 25556 个触发器)
2. 完成高质量 Placement (7% HPWL 增量)
3. 解决了布线层定义问题

⚠️ **剩余挑战**:
1. CTS 无法自动构建时钟树
2. IO PAD 阻止 routing 完成

💡 **建议**:
- 对于 25MHz 设计，使用理想时钟 + 核心模块网表
- 或使用商业工具完成最后步骤
- 当前的 Placement 结果已经足够用于面积和初步时序评估

---

**创建时间**: 2025-12-03 11:47  
**调试时间**: ~10 分钟  
**状态**: Placement 成功，CTS/Routing 需要进一步工作
