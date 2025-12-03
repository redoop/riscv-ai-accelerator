# v0.4.1 物理设计计划

**开始日期**: 2025-12-03  
**版本**: v0.4.1  
**工艺**: ICS55 55nm  
**状态**: 规划中

---

## 📋 物理设计流程

### 阶段概览

```
综合网表 → 静态时序分析 → 布局规划 → 布局 → 时钟树综合 → 布线 → 物理验证 → GDSII
```

### 详细阶段

1. **静态时序分析 (STA)** ⏱️ 30 分钟
   - 时序路径分析
   - 建立时间/保持时间检查
   - 时钟域交叉验证
   - 时序报告生成

2. **布局规划 (Floorplanning)** ⏱️ 1 小时
   - 芯片尺寸规划
   - IO Pad 放置
   - 电源网格规划
   - 宏单元放置

3. **布局 (Placement)** ⏱️ 2 小时
   - 全局布局
   - 详细布局
   - 布局优化
   - 拥塞分析

4. **时钟树综合 (CTS)** ⏱️ 1 小时
   - 时钟树生成
   - 时钟偏斜优化
   - 时钟缓冲器插入

5. **布线 (Routing)** ⏱️ 3 小时
   - 全局布线
   - 详细布线
   - 布线优化
   - 天线效应修复

6. **物理验证** ⏱️ 1 小时
   - DRC (设计规则检查)
   - LVS (版图与原理图一致性)
   - 天线效应检查
   - 电迁移分析

7. **GDSII 生成** ⏱️ 30 分钟
   - 版图数据生成
   - 最终检查
   - 流片数据准备

**总预计时间**: 9 小时

---

## 🎯 当前状态

### 已完成

- ✅ RTL 设计 (Chisel)
- ✅ 功能验证 (57/57 测试通过)
- ✅ 逻辑综合 (Yosys)
- ✅ 后综合仿真 (Icarus Verilog)
- ✅ IO Pad 优化 (97 → 61)

### 进行中

- 🔄 **静态时序分析 (STA)** ← 当前阶段

### 待完成

- ⏳ 布局规划 (Floorplanning)
- ⏳ 布局 (Placement)
- ⏳ 时钟树综合 (CTS)
- ⏳ 布线 (Routing)
- ⏳ 物理验证 (DRC/LVS)
- ⏳ GDSII 生成

---

## 🔧 工具链

### 可用工具

| 工具 | 版本 | 用途 | 状态 |
|------|------|------|------|
| **OpenROAD** | Latest | 完整 P&R 流程 | ✅ 已安装 |
| **Yosys** | 0.58+138 | 逻辑综合 | ✅ 已使用 |
| **OpenSTA** | Latest | 静态时序分析 | ✅ 已安装 |
| **Magic** | - | 版图编辑/DRC | ⚠️ 待确认 |
| **Netgen** | - | LVS 验证 | ⚠️ 待确认 |
| **KLayout** | - | 版图查看 | ⚠️ 待确认 |

### PDK 支持

| PDK | 工艺 | 状态 | 说明 |
|-----|------|------|------|
| **ICS55** | 55nm | ✅ 已使用 | 综合完成 |
| **SkyWater 130nm** | 130nm | ⚠️ 备选 | 开源 PDK |
| **IHP SG13G2** | 130nm | ⚠️ 备选 | 开源 PDK |

---

## 📊 设计约束

### 芯片规格

| 参数 | 目标值 | 当前值 | 状态 |
|------|--------|--------|------|
| **芯片面积** | < 1 mm² | 0.33 mm² | ✅ |
| **IO Pads** | ≤ 81 | 61 | ✅ |
| **时钟频率** | 100 MHz | 100 MHz | ✅ |
| **功耗** | < 100 mW | TBD | ⏳ |
| **电源电压** | 1.2V | 1.2V | ✅ |
| **工作温度** | -40~125°C | 25°C (典型) | ✅ |

### 时序约束

```tcl
# 主时钟
create_clock -name clk -period 10.0 [get_ports clock]

# SPI 时钟 (10 MHz)
create_generated_clock -name spi_clk -source [get_ports clock] \
    -divide_by 10 [get_pins lcd/spi_clk_reg/Q]

# 输入延迟
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# 输出延迟
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# 时钟不确定性
set_clock_uncertainty 0.5 [all_clocks]

# 时钟转换时间
set_clock_transition 0.1 [all_clocks]
```

### 物理约束

```tcl
# 芯片尺寸 (假设正方形)
set die_width 600
set die_height 600

# 核心区域 (留出 IO 环)
set core_margin 50
set core_width [expr $die_width - 2 * $core_margin]
set core_height [expr $die_height - 2 * $core_margin]

# 电源网格
set power_stripe_width 2.0
set power_stripe_spacing 20.0

# 布局密度
set target_density 0.7
```

---

## 🚀 执行计划

### Phase 1: 静态时序分析 (立即执行)

**目标**: 验证时序收敛，识别关键路径

**步骤**:
```bash
cd chisel/synthesis/sta
./run_sta.sh
```

**预期输出**:
- 时序报告 (setup/hold)
- 关键路径列表
- 时钟域交叉报告
- 时序裕量分析

**验收标准**:
- WNS (Worst Negative Slack) > 0
- TNS (Total Negative Slack) = 0
- 无时序违例

### Phase 2: 布局规划 (STA 通过后)

**目标**: 规划芯片布局，放置 IO Pad

**步骤**:
```bash
cd chisel/synthesis/physical_design
./run_floorplan.sh
```

**关键任务**:
1. 确定芯片尺寸
2. 放置 61 个 IO Pads
3. 规划电源网格
4. 放置宏单元 (RAM, ROM)

**验收标准**:
- IO Pad 位置合理
- 电源网格覆盖完整
- 核心利用率 60-80%

### Phase 3: 布局 (布局规划完成后)

**目标**: 放置标准单元

**步骤**:
```bash
./run_placement.sh
```

**关键任务**:
1. 全局布局
2. 详细布局
3. 拥塞分析
4. 布局优化

**验收标准**:
- 无拥塞区域
- 布局密度均匀
- 时序预估满足要求

### Phase 4: 时钟树综合 (布局完成后)

**目标**: 生成时钟树，优化时钟偏斜

**步骤**:
```bash
./run_cts.sh
```

**关键任务**:
1. 时钟树生成
2. 缓冲器插入
3. 时钟偏斜优化

**验收标准**:
- 时钟偏斜 < 100 ps
- 时钟延迟均衡
- 无时钟违例

### Phase 5: 布线 (CTS 完成后)

**目标**: 完成所有信号布线

**步骤**:
```bash
./run_routing.sh
```

**关键任务**:
1. 全局布线
2. 详细布线
3. 天线效应修复
4. 布线优化

**验收标准**:
- 100% 布线完成
- 无 DRC 违例
- 无天线效应

### Phase 6: 物理验证 (布线完成后)

**目标**: 验证版图正确性

**步骤**:
```bash
./run_verification.sh
```

**关键任务**:
1. DRC 检查
2. LVS 验证
3. 天线效应检查
4. 电迁移分析

**验收标准**:
- DRC 0 错误
- LVS 通过
- 无天线效应
- 电迁移安全

### Phase 7: GDSII 生成 (验证通过后)

**目标**: 生成流片数据

**步骤**:
```bash
./generate_gdsii.sh
```

**交付物**:
- GDSII 文件
- 网表文件
- 时序报告
- 验证报告

---

## 📈 里程碑

| 里程碑 | 预计完成 | 状态 |
|--------|---------|------|
| 静态时序分析 | Day 1 | 🔄 进行中 |
| 布局规划 | Day 1-2 | ⏳ 待开始 |
| 布局 | Day 2-3 | ⏳ 待开始 |
| 时钟树综合 | Day 3 | ⏳ 待开始 |
| 布线 | Day 3-4 | ⏳ 待开始 |
| 物理验证 | Day 4 | ⏳ 待开始 |
| GDSII 生成 | Day 4 | ⏳ 待开始 |

---

## ⚠️ 风险与挑战

### 技术风险

1. **时序收敛**
   - 风险: 100 MHz 可能难以满足
   - 缓解: 降低频率或优化关键路径

2. **布线拥塞**
   - 风险: 高密度区域布线困难
   - 缓解: 调整布局密度

3. **电源完整性**
   - 风险: IR drop 过大
   - 缓解: 优化电源网格

4. **IO Pad 限制**
   - 风险: 61 个 IO 可能不够
   - 缓解: 已优化，有 20 个裕量

### 工具限制

1. **PDK 支持**
   - ICS55 PDK 可能不完整
   - 备选: 使用 SkyWater 130nm

2. **OpenROAD 成熟度**
   - 开源工具可能有限制
   - 备选: 使用商业工具

---

## 📚 参考资料

### OpenROAD 文档

- [OpenROAD Flow Scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
- [OpenROAD User Guide](https://openroad.readthedocs.io/)
- [OpenSTA Manual](https://github.com/The-OpenROAD-Project/OpenSTA)

### PDK 文档

- [ICS55 PDK](https://github.com/idea-fasoc/OpenFASOC/tree/main/openfasoc/generators/pdk)
- [SkyWater 130nm PDK](https://github.com/google/skywater-pdk)
- [IHP SG13G2 PDK](https://github.com/IHP-GmbH/IHP-Open-PDK)

---

## ✅ 下一步行动

### 立即执行

1. **运行静态时序分析**
   ```bash
   cd chisel/synthesis/sta
   ./run_sta.sh
   ```

2. **分析时序报告**
   - 检查 WNS/TNS
   - 识别关键路径
   - 评估时序裕量

3. **准备布局规划**
   - 确定芯片尺寸
   - 规划 IO Pad 位置
   - 设计电源网格

### 后续任务

4. 执行布局规划
5. 执行布局
6. 执行时钟树综合
7. 执行布线
8. 执行物理验证
9. 生成 GDSII

---

**创建日期**: 2025-12-03  
**预计完成**: 2025-12-07 (4 天)  
**状态**: 规划完成，准备执行
