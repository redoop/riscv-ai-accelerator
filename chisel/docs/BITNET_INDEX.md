# BitNetScaleAiChip 文档索引

## 📁 文档清单

### 核心设计文件
- **`BitNetScaleAiChip.sv`** (generated/)
  - 生成的 SystemVerilog 代码 (~2,900 行)
  - 可直接用于 FPGA 综合或 ASIC 流片
  - 包含 6 种模块，12 个实例

- **`BitNetScaleDesign.scala`** (src/main/scala/)
  - Chisel 源代码
  - 包含 BitNetComputeUnit, BitNetMatrixMultiplier, BitNetScaleAiChip

### 分析文档

#### 1. BITNET_CHIP_ANALYSIS.md ⭐
**最详细的技术文档**
- 完整的模块分析
- 端口接口说明
- 地址映射方案
- 设计特点分析
- 资源估算
- 综合建议

**适合**: 深入理解设计细节、验证工程师、综合工程师

#### 2. BITNET_MODULE_HIERARCHY.txt 📊
**可视化层次结构**
- ASCII 艺术图展示模块层次
- 数据流图
- 时序图
- 存储器规格详解

**适合**: 快速理解架构、新团队成员、文档展示

#### 3. BITNET_SUMMARY.txt 📋
**成果总结**
- 模块组成清单
- 核心特性列表
- 技术规格表
- 接口定义
- 性能指标
- 使用流程

**适合**: 项目汇报、快速概览、管理层展示

#### 4. BITNET_QUICK_REFERENCE.md 🚀
**开发者快速参考**
- 寄存器映射表
- 内存映射表
- 权重编码表
- 典型操作序列 (C 代码示例)
- 性能计算公式
- 调试技巧
- 常见问题解答

**适合**: 软件开发、驱动编写、日常开发参考

#### 5. BITNET_INDEX.md (本文件) 📚
**文档导航**
- 文档清单
- 使用指南
- 快速链接

## 🎯 使用指南

### 场景 1: 我是硬件工程师，想理解设计
1. 先看 **BITNET_MODULE_HIERARCHY.txt** 了解整体架构
2. 再看 **BITNET_CHIP_ANALYSIS.md** 深入理解细节
3. 参考 **BitNetScaleAiChip.sv** 查看实际代码

### 场景 2: 我是软件工程师，要写驱动
1. 直接看 **BITNET_QUICK_REFERENCE.md**
2. 参考其中的 C 代码示例
3. 查看寄存器和内存映射表

### 场景 3: 我要做项目汇报
1. 使用 **BITNET_SUMMARY.txt** 作为演讲稿
2. 引用 **BITNET_MODULE_HIERARCHY.txt** 中的图表
3. 展示关键指标和优势

### 场景 4: 我要进行 FPGA 综合
1. 阅读 **BITNET_CHIP_ANALYSIS.md** 的"综合建议"部分
2. 使用 **BitNetScaleAiChip.sv** 作为顶层文件
3. 参考资源估算调整约束

### 场景 5: 我要进行功能验证
1. 查看 **BITNET_QUICK_REFERENCE.md** 的操作序列
2. 参考 **BITNET_CHIP_ANALYSIS.md** 的验证要点
3. 使用 **BITNET_MODULE_HIERARCHY.txt** 的时序图

## 📊 文档对比

| 文档 | 页数 | 详细度 | 技术深度 | 适合人群 |
|------|------|--------|---------|---------|
| CHIP_ANALYSIS | ⭐⭐⭐⭐⭐ | 最详细 | 深 | 硬件工程师 |
| MODULE_HIERARCHY | ⭐⭐⭐⭐ | 详细 | 中 | 架构师 |
| SUMMARY | ⭐⭐⭐ | 中等 | 中 | 项目经理 |
| QUICK_REFERENCE | ⭐⭐⭐⭐ | 详细 | 浅 | 软件工程师 |

## 🔗 快速链接

### 设计文件
- [Chisel 源码](src/main/scala/BitNetScaleDesign.scala)
- [生成的 Verilog](generated/BitNetScaleAiChip.sv)
- [主程序](src/main/scala/RiscvAiChipMain.scala)

### 文档
- [详细分析](BITNET_CHIP_ANALYSIS.md)
- [模块层次](BITNET_MODULE_HIERARCHY.txt)
- [成果总结](BITNET_SUMMARY.txt)
- [快速参考](BITNET_QUICK_REFERENCE.md)

### 工具脚本
- [生成脚本](generate_bitnet.sh)
- [综合修复](fix_synthesis.sh)
- [运行脚本](run.sh)

## 📈 关键数据速查

```
模块数量:     6 种 (12 实例)
代码行数:     ~2,900 行
逻辑资源:     ~5,000 LUTs
存储资源:     ~3 KB SRAM
DSP 资源:     0 (无乘法器)
功耗:         ~45 mW @ 100MHz
延迟:         4,096 周期/矩阵
性能:         2 GOPS @ 100MHz
```

## 🎓 学习路径

### 初学者
1. BITNET_SUMMARY.txt (了解概况)
2. BITNET_MODULE_HIERARCHY.txt (理解架构)
3. BITNET_QUICK_REFERENCE.md (学习使用)

### 进阶者
1. BITNET_CHIP_ANALYSIS.md (深入细节)
2. BitNetScaleDesign.scala (学习 Chisel)
3. BitNetScaleAiChip.sv (分析生成代码)

### 专家
1. 直接阅读源码和生成代码
2. 参考文档进行优化和改进
3. 扩展设计添加新功能

## 💡 提示

- 所有文档都是从生成的 SystemVerilog 代码分析得出
- 文档之间相互补充，建议结合阅读
- 代码已验证可综合，可直接使用
- 如有疑问，优先查看 QUICK_REFERENCE

## 📞 技术支持

如需帮助，请参考:
1. 文档中的"常见问题"部分
2. Chisel 官方文档
3. CIRCT 工具文档

---

**最后更新**: 2024
**版本**: 1.0
**状态**: ✅ 完成并验证
