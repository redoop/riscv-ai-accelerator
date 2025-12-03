# OpenSTA 时序分析问题总结

## 问题描述

OpenSTA 报告 "No paths found"，无法分析任何时序路径，尽管逻辑综合成功完成。

## 根本原因

**IO PAD 模块是黑盒，阻止了时钟传播**

1. **时钟路径**：
   ```
   sys_clk_i_pad (输入端口) 
     → u_sys_clk_pad (P65_1233_PWE 黑盒) 
     → sys_clk (内部网络) 
     → 触发器 CK 引脚
   ```

2. **问题分析**：
   - `P65_1233_PWE` 和 `P65_1233_PBMUX` 在网表中是黑盒模块
   - 黑盒模块没有输出引脚定义
   - `sys_clk` 网络没有驱动源（driver pins = 0）
   - OpenSTA 无法从输入端口传播时钟到内部网络

3. **调试证据**：
   ```tcl
   # 检查 sys_clk 网络
   set net [get_nets sys_clk]
   set driver_pins [get_pins -of_objects $net -filter "direction==out"]
   # 结果：driver_pins = "" (空)
   ```

## 解决方案

### 方案 1：获取 PAD 的 Liberty 时序模型（推荐）⭐

**步骤**：
1. 联系 ICS55 PDK 供应商（IDE Platform）
2. 获取 IO PAD 的 `.lib` 文件：
   - `P65_1233_PWE.lib` (时钟输入 PAD)
   - `P65_1233_PBMUX.lib` (双向 PAD)
3. 在 STA 脚本中加载这些库：
   ```tcl
   read_liberty /path/to/P65_1233_PWE.lib
   read_liberty /path/to/P65_1233_PBMUX.lib
   ```

**优点**：
- ✅ 最准确的时序分析
- ✅ 包括 PAD 的实际延迟
- ✅ 符合标准 ASIC 设计流程

**缺点**：
- ❌ 需要从供应商获取文件
- ❌ 可能需要 NDA

### 方案 2：创建简化的 PAD 模型

**步骤**：
1. 创建功能模型（已完成）：`pad_models.v`
2. 创建简化的 Liberty 模型（估算延迟）
3. 在 STA 中使用这些模型

**优点**：
- ✅ 可以立即开始分析
- ✅ 不依赖供应商

**缺点**：
- ❌ 延迟值不准确
- ❌ 需要后续用真实模型验证

### 方案 3：使用虚拟时钟（临时方案）

**步骤**：
1. 使用提供的 `sta_virtual_clock.sdc`
2. 创建虚拟时钟，不关联到端口
3. 手动设置时钟延迟

**优点**：
- ✅ 可以评估内部逻辑时序
- ✅ 快速获得初步结果

**缺点**：
- ❌ 不包括 PAD 延迟
- ❌ 结果不完整
- ❌ 仅用于初步评估

## 当前状态

- ✅ 逻辑综合成功：0 错误，363 警告
- ✅ 网表生成：623,516 行
- ✅ 芯片面积：~0.3 mm²
- ❌ 静态时序分析：被 PAD 黑盒阻塞

## 下一步行动

### 短期（立即）
1. 使用方案 3（虚拟时钟）进行初步时序评估
2. 验证内部逻辑的时序约束

### 中期（1-2 周）
1. 联系 IDE Platform 获取 IO PAD 的 Liberty 文件
2. 或创建简化的 PAD Liberty 模型

### 长期（设计完成前）
1. 使用真实的 PAD 时序模型完成完整的 STA
2. 确保所有时序路径满足约束

## 相关文件

- 网表：`project/netlist/asic_top_ics55.v`
- SDC 约束：`sdc/sta_combined.sdc`, `sdc/sta_virtual_clock.sdc`
- STA 脚本：`run_sta_combined.tcl`
- 调试脚本：`debug_driver.tcl`, `debug_clock.tcl`
- PAD 模型：`pad_models.v` (功能模型，无时序)

## 参考信息

- 综合报告：`project/netlist/synthesis_stats.txt`
- 时序约束：`sdc/timing_complete.sdc`
- 主时钟：100MHz (10ns 周期)
- SPI 时钟：10MHz (100ns 周期)

## 联系信息

**PDK 供应商**：IDE Platform  
**PDK 版本**：ICS55 55nm  
**工艺角**：TT 1.2V 25°C

---

**创建日期**：2025-12-03  
**最后更新**：2025-12-03
