# RISC-V AI 加速器文档索引

## 📚 文档概览

本目录包含 RISC-V AI 加速器系统集成项目的完整文档。

---

## 🚀 快速开始

**新用户请从这里开始**:
1. 阅读 [INTEGRATION_README.md](INTEGRATION_README.md) - 快速开始指南
2. 查看 [INTEGRATION.md](INTEGRATION.md) - 了解系统架构
3. 运行测试 - 参考 [TESTING.md](TESTING.md)

---

## 📖 文档列表

### 核心文档

#### 1. [INTEGRATION_README.md](INTEGRATION_README.md)
**快速开始指南**
- 项目概述
- 快速开始步骤
- 系统架构图
- 软件编程示例
- 常见问题

**适合**: 新用户、快速上手

---

#### 2. [INTEGRATION.md](INTEGRATION.md)
**集成架构详解**
- 系统组成
- 接口定义 (PCPI, AXI-Lite)
- 地址映射
- 文件结构
- 核心模块说明
- 使用方法
- 性能特性
- 扩展方向

**适合**: 深入了解系统架构

---

#### 3. [TESTING.md](TESTING.md)
**测试完整指南**
- 测试架构
- 测试环境设置
- 单元测试详解
- 集成测试详解
- 测试覆盖率
- 性能测试
- 调试技巧
- 持续集成

**适合**: 测试和验证

---

### 参考文档

#### 4. [TEST_SUMMARY.md](TEST_SUMMARY.md)
**测试总结**
- 完成工作总结
- 测试运行指南
- 架构概览
- 地址映射
- 关键特性
- 测试覆盖
- 性能指标
- 软件接口示例
- 已知限制
- 下一步工作

**适合**: 了解测试结果和项目状态

---

#### 5. [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)
**验证清单**
- 完成项目清单
- 验证项目
- 测试覆盖率目标
- 代码质量检查
- 性能验证
- 下一步工作
- 验证结论

**适合**: 项目验证和质量保证

---

#### 6. [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)
**集成完成报告**
- 项目状态
- 执行摘要
- 交付物清单
- 验证结论

**适合**: 项目总结和汇报

---

## 🗂️ 文档分类

### 按用途分类

#### 入门文档
- [INTEGRATION_README.md](INTEGRATION_README.md) - 快速开始

#### 技术文档
- [INTEGRATION.md](INTEGRATION.md) - 架构详解
- [TESTING.md](TESTING.md) - 测试指南

#### 参考文档
- [TEST_SUMMARY.md](TEST_SUMMARY.md) - 测试总结
- [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) - 验证清单
- [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - 完成报告

### 按读者分类

#### 开发者
1. [INTEGRATION_README.md](INTEGRATION_README.md) - 快速上手
2. [INTEGRATION.md](INTEGRATION.md) - 深入理解
3. [TESTING.md](TESTING.md) - 测试开发

#### 测试工程师
1. [TESTING.md](TESTING.md) - 测试指南
2. [TEST_SUMMARY.md](TEST_SUMMARY.md) - 测试结果
3. [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) - 验证清单

#### 项目经理
1. [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - 项目总结
2. [TEST_SUMMARY.md](TEST_SUMMARY.md) - 项目状态
3. [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) - 质量保证

---

## 📁 相关资源

### 源代码
- `../src/main/scala/` - Chisel 源码
- `../src/main/rtl/` - Verilog 源码
- `../src/test/scala/` - 测试代码

### 示例代码
- `../examples/matrix_multiply.c` - C 语言示例

### 工具脚本
- `../run_integration_tests.sh` - 完整测试脚本
- `../quick_test.sh` - 快速测试脚本

---

## 🔍 快速查找

### 我想了解...

**系统架构**
→ [INTEGRATION.md](INTEGRATION.md) 第2节

**如何编程**
→ [INTEGRATION_README.md](INTEGRATION_README.md) 第7节  
→ [INTEGRATION.md](INTEGRATION.md) 第6节

**如何测试**
→ [TESTING.md](TESTING.md) 第2节  
→ [TEST_SUMMARY.md](TEST_SUMMARY.md) 第2节

**接口定义**
→ [INTEGRATION.md](INTEGRATION.md) 第3节

**地址映射**
→ [INTEGRATION.md](INTEGRATION.md) 第3.1节  
→ [INTEGRATION_README.md](INTEGRATION_README.md) 第4节

**性能指标**
→ [TEST_SUMMARY.md](TEST_SUMMARY.md) 第7节  
→ [INTEGRATION.md](INTEGRATION.md) 第7节

**测试结果**
→ [TEST_SUMMARY.md](TEST_SUMMARY.md) 第6节  
→ [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) 第3节

---

## 📝 文档更新记录

| 日期 | 文档 | 更新内容 |
|------|------|---------|
| 2024-11-14 | 所有文档 | 初始版本完成 |

---

## 🤝 贡献指南

如需更新文档:
1. 保持文档格式一致
2. 更新相关交叉引用
3. 更新本索引文件
4. 更新版本记录

---

## 📧 联系方式

如有文档问题或建议:
- 提交 Issue
- 发起 Pull Request

---

**最后更新**: 2024年11月14日  
**文档版本**: 1.0
