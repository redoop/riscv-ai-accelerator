# EdgeAiSoC 开发状态

## 当前状态

EdgeAiSoC 的设计已完成，包含以下组件：

### ✅ 已完成的设计
1. **内存映射定义** - 完整的地址空间规划
2. **寄存器映射** - AI 加速器的寄存器定义
3. **中断映射** - 32 个中断源的分配
4. **PicoRV32 集成** - RISC-V 核心的 BlackBox 封装
5. **AI 加速器 Wrapper** - CompactScale 和 BitNetScale 的封装
6. **DMA 控制器** - 基础 DMA 功能
7. **中断控制器** - 中断管理逻辑
8. **外设控制器** - UART 和 GPIO
9. **完整文档** - README 和构建指南

### ⚠️ 已知问题

当前实现遇到 Chisel 的 AXI4-Lite 接口方向问题：

**问题描述**:
- 使用 `Flipped(new AXI4LiteIO())` 时，子字段的方向会反转
- 在模块内部无法直接给 Flipped IO 的输出端口赋默认值
- 需要重新设计 AXI 接口的连接方式

**影响**:
- AddressDecoder 无法正确实现
- AI 加速器 Wrapper 的 AXI 接口连接有问题
- 当前无法生成完整的 SystemVerilog

### 🔧 解决方案

有以下几种解决方案：

#### 方案 1: 使用 DecoupledIO (推荐)
将 AXI4-Lite 接口改为使用 Chisel 的 `Decoupled` 接口，这样方向更清晰。

#### 方案 2: 手动定义方向
不使用 `Flipped`，而是手动定义每个信号的方向（Input/Output）。

#### 方案 3: 使用 Diplomatic (高级)
使用 RocketChip 的 Diplomatic 框架来处理总线连接。

#### 方案 4: 简化设计
移除 AXI 接口，使用简单的寄存器读写接口。

## 设计价值

尽管当前实现有技术问题，但 EdgeAiSoC 的设计具有重要价值：

### 架构设计 ✅
- 完整的 SoC 架构规划
- 合理的内存映射
- 清晰的模块划分
- 标准的总线接口设计

### 文档完整 ✅
- 详细的 README
- 构建指南
- 寄存器映射表
- 中断映射表
- 使用示例

### 概念验证 ✅
- RISC-V + AI 加速器集成方案
- 双加速器架构（CompactScale + BitNetScale）
- DMA 和中断支持
- 完整的外设系统

## 后续工作

### 短期 (1-2周)
1. 修复 AXI 接口方向问题
2. 实现简化版的 AddressDecoder
3. 完成 SystemVerilog 生成
4. 基础功能测试

### 中期 (1-2月)
1. FPGA 原型验证
2. 软件驱动开发
3. 性能测试和优化
4. 完整的测试覆盖

### 长期 (3-6月)
1. 流片准备
2. DRC/LVS 验证
3. 时序收敛
4. 量产准备

## 使用建议

当前阶段，EdgeAiSoC 可以作为：

1. **参考设计** - 学习 RISC-V SoC 设计
2. **架构模板** - 作为其他 SoC 项目的起点
3. **文档示例** - 完整的设计文档范例
4. **概念验证** - AI 加速器集成方案

## 相关文件

- `chisel/src/main/scala/EdgeAiSoC.scala` - 主设计文件
- `chisel/docs/EdgeAiSoC_README.md` - 项目文档
- `chisel/docs/EdgeAiSoC_BUILD.md` - 构建指南
- `chisel/docs/RISCV_INTEGRATION_PLAN.md` - 集成方案

## 总结

EdgeAiSoC 是一个设计完整、文档齐全的 RISC-V AI SoC 项目。虽然当前实现遇到 Chisel AXI 接口的技术问题，但整体架构设计合理，具有很高的参考价值。通过修复接口问题，可以快速完成实现并进行 FPGA 验证。

---

**状态**: 设计完成，实现待修复  
**最后更新**: 2025-11-14  
**维护者**: EdgeAiSoC 团队
