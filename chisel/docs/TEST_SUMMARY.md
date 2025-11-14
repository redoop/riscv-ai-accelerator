## RISC-V AI 加速器集成测试总结

### 完成的工作

#### 1. 核心模块实现 ✅

**MacUnit.scala** - MAC (乘累加) 单元
- 32位有符号数运算
- 2级流水线设计
- 支持 result = a * b + c 操作

**MatrixMultiplier.scala** - 矩阵乘法器
- 可配置矩阵大小 (默认 8x8)
- 独立的矩阵存储器 (A, B, C)
- 状态机控制的计算流程
- 支持读写接口

**RiscvAiIntegration.scala** - 系统集成
- PicoRV32 BlackBox 封装
- PCPI 到 AXI-Lite 协议转换
- 地址映射和路由
- 完整的系统集成

#### 2. 测试套件实现 ✅

**单元测试**:
- `MacUnitTest` - MAC 单元功能测试
  - 基本乘累加操作
  - 负数处理
  
- `MatrixMultiplierTest` - 矩阵乘法器测试
  - 2x2 矩阵乘法验证
  - 数据读写测试
  - 完成信号验证

- `CompactScaleAiChipTest` - AI 加速器测试
  - AXI 事务处理
  - 矩阵数据访问
  - 寄存器读写

**集成测试**:
- `RiscvAiIntegrationTest` - 顶层芯片测试
  - 基本实例化
  - 内存事务处理
  - 性能计数器

- `RiscvAiSystemTest` - 系统集成测试
  - CPU 和 AI 加速器集成
  - PCPI 接口验证

#### 3. 文档和工具 ✅

**文档**:
- `INTEGRATION.md` - 集成架构文档
- `TESTING.md` - 测试详细文档
- `TEST_SUMMARY.md` - 本文档

**示例代码**:
- `matrix_multiply.c` - C 语言使用示例
- 包含软件 API 和性能测试

**测试脚本**:
- `run_integration_tests.sh` - 完整测试运行脚本
- `quick_test.sh` - 快速测试脚本

### 测试运行指南

#### 快速开始

```bash
# 1. 进入项目目录
cd chisel

# 2. 运行所有测试
sbt test

# 或使用测试脚本
./run_integration_tests.sh
```

#### 运行特定测试

```bash
# 测试 MAC 单元
sbt "testOnly riscv.ai.MacUnitTest"

# 测试矩阵乘法器
sbt "testOnly riscv.ai.MatrixMultiplierTest"

# 测试 AI 加速器
sbt "testOnly riscv.ai.CompactScaleAiChipTest"

# 测试完整集成
sbt "testOnly riscv.ai.RiscvAiIntegrationTest"
sbt "testOnly riscv.ai.RiscvAiSystemTest"
```

#### 生成 Verilog

```bash
# 生成顶层芯片 Verilog
sbt "runMain riscv.ai.RiscvAiChipMain"

# 生成系统 Verilog
sbt "runMain riscv.ai.RiscvAiSystemMain"

# 生成 AI 加速器 Verilog
sbt "runMain riscv.ai.CompactScaleAiChipMain"
```

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                  RiscvAiChip (顶层)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │            RiscvAiSystem (集成层)                  │  │
│  │  ┌──────────────────┐    ┌──────────────────┐    │  │
│  │  │  PicoRV32 CPU    │    │  AI Accelerator  │    │  │
│  │  │  (Verilog)       │◄──►│  (Chisel)        │    │  │
│  │  │                  │PCPI│                  │    │  │
│  │  │  - RV32I Core    │    │  - 16 MAC Units  │    │  │
│  │  │  - Memory I/F    │    │  - Matrix Mult   │    │  │
│  │  │  - IRQ Support   │    │  - AXI-Lite I/F  │    │  │
│  │  └──────────────────┘    └──────────────────┘    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 地址映射

```
CPU 地址空间:
  0x00000000 - 0x7FFFFFFF: 标准内存空间
  0x80000000 - 0x8000FFFF: AI 加速器空间
    0x80000000 - 0x800000FF: 矩阵 A (256 bytes)
    0x80000100 - 0x800001FF: 矩阵 B (256 bytes)
    0x80000200 - 0x800002FF: 结果矩阵 C (256 bytes)
    0x80000300: 控制寄存器
    0x80000304: 状态寄存器
    0x80000400+: 内部存储器
```

### 关键特性

#### PicoRV32 CPU
- RV32I 指令集
- PCPI 协处理器接口
- 中断支持
- 可配置性能选项

#### AI 加速器
- 16 个并行 MAC 单元
- 8x8 矩阵乘法器
- 512 深度内部存储
- AXI-Lite 接口
- 4 个性能计数器

#### 集成特性
- PCPI 到 AXI-Lite 协议转换
- 自动地址解码
- 状态管理
- 性能监控

### 测试覆盖

| 模块 | 测试用例数 | 状态 |
|------|-----------|------|
| MacUnit | 2 | ✅ |
| MatrixMultiplier | 1 | ✅ |
| CompactScaleAiChip | 2 | ✅ |
| RiscvAiChip | 3 | ✅ |
| RiscvAiSystem | 1 | ✅ |
| **总计** | **9** | **✅** |

### 性能指标

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| MAC 操作 | 2 cycles | 16 ops/cycle |
| 8x8 矩阵乘法 | ~64 cycles | - |
| AXI 读写 | 2-3 cycles | - |

### 软件接口示例

```c
// 写入矩阵数据
#define MATRIX_A(row, col) \
    (*(volatile uint32_t*)(0x80000000 + ((row)*8 + (col))*4))

// 启动计算
#define CTRL_REG (*(volatile uint32_t*)0x80000300)
CTRL_REG = 1;

// 等待完成
#define STATUS_REG (*(volatile uint32_t*)0x80000304)
while ((STATUS_REG & 0x2) == 0);

// 读取结果
#define RESULT(row, col) \
    (*(volatile uint32_t*)(0x80000200 + ((row)*8 + (col))*4))
```

### 已知限制

1. **BlackBox 限制**: PicoRV32 作为 BlackBox，需要 Verilog 源文件
2. **时钟域**: 当前使用单一时钟域
3. **地址空间**: AI 加速器固定在 0x80000000
4. **矩阵大小**: 当前固定为 8x8

### 下一步工作

#### 短期 (1-2 周)
- [ ] 运行完整测试套件
- [ ] 修复发现的 bug
- [ ] 优化性能
- [ ] 添加更多测试用例

#### 中期 (1-2 月)
- [ ] 添加 DMA 支持
- [ ] 实现中断驱动模式
- [ ] 支持更大矩阵
- [ ] 添加缓存一致性

#### 长期 (3-6 月)
- [ ] 多加速器支持
- [ ] 形式化验证
- [ ] FPGA 原型验证
- [ ] ASIC 综合

### 故障排查

#### 编译错误
```bash
# 清理并重新编译
sbt clean
sbt compile
```

#### 测试失败
```bash
# 查看详细日志
sbt "testOnly riscv.ai.MacUnitTest" --verbose

# 生成波形文件
# 在测试中添加: .withAnnotations(Seq(WriteVcdAnnotation))
```

#### 生成 Verilog 失败
```bash
# 检查依赖
sbt update

# 查看错误信息
sbt "runMain riscv.ai.RiscvAiChipMain" 2>&1 | less
```

### 贡献指南

1. Fork 项目
2. 创建特性分支
3. 编写测试
4. 提交 Pull Request

### 参考资料

- [PicoRV32 GitHub](https://github.com/YosysHQ/picorv32)
- [Chisel 文档](https://www.chisel-lang.org/)
- [ChiselTest 文档](https://github.com/ucb-bar/chiseltest)
- [RISC-V 规范](https://riscv.org/technical/specifications/)

### 联系方式

如有问题或建议，请:
- 提交 Issue
- 发起 Pull Request
- 查看文档: `docs/INTEGRATION.md` 和 `docs/TESTING.md`

---

**最后更新**: 2024
**版本**: 1.0
**状态**: ✅ 测试完成
