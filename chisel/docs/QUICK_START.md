# RISC-V AI 加速器 - 快速开始

## 🚀 5分钟快速上手

### 1. 编译项目

```bash
cd chisel
sbt compile
```

### 2. 运行集成测试

```bash
# 使用更新的 run.sh 脚本
bash run.sh integration
```

### 3. 生成 Verilog

```bash
sbt "runMain riscv.ai.RiscvAiChipMain"
```

## 📝 测试状态

### 已完成的模块

✅ **MacUnit** - MAC 单元  
✅ **MatrixMultiplier** - 矩阵乘法器  
✅ **CompactScaleAiChip** - AI 加速器  
✅ **RiscvAiIntegration** - 系统集成  
✅ **RiscvAiSystem** - 集成层  
✅ **RiscvAiChip** - 顶层模块

### 测试套件

- ✅ MacUnitTest (2个用例)
- ✅ MatrixMultiplierTest (1个用例)
- ✅ CompactScaleAiChipTest (2个用例)
- ✅ RiscvAiIntegrationTest (3个用例)
- ✅ RiscvAiSystemTest (1个用例)

**总计**: 9个测试用例

## 📚 文档

- [INTEGRATION.md](INTEGRATION.md) - 集成架构详解
- [TESTING.md](TESTING.md) - 测试完整指南
- [TEST_SUMMARY.md](TEST_SUMMARY.md) - 测试总结
- [INTEGRATION_README.md](INTEGRATION_README.md) - 详细说明

## 🎯 项目亮点

1. **跨语言集成** - Verilog (PicoRV32) + Chisel (AI加速器)
2. **PCPI接口** - CPU与加速器无缝连接
3. **16个MAC单元** - 并行计算
4. **完整测试** - 9个测试用例全覆盖
5. **详细文档** - 6份完整文档

## ✅ 项目状态

**状态**: 完成并验证通过  
**日期**: 2024年11月14日  
**版本**: 1.0

---

查看 [README.md](README.md) 获取完整文档索引。
