# RISC-V AI 加速器系统 - 完整集成

## 🎯 项目概述

本项目成功将 **PicoRV32 RISC-V 处理器** (Verilog) 与 **自定义 AI 加速器** (Chisel) 通过 PCPI 接口集成，形成完整的 RISC-V AI 加速器系统。

## ✨ 主要特性

### 处理器 (PicoRV32)
- ✅ RV32I 指令集
- ✅ PCPI 协处理器接口
- ✅ 中断支持
- ✅ 可配置性能选项

### AI 加速器
- ✅ 16 个并行 MAC 单元
- ✅ 8x8 矩阵乘法器
- ✅ 512 深度内部存储
- ✅ AXI-Lite 接口
- ✅ 性能监控

### 系统集成
- ✅ PCPI ↔ AXI-Lite 协议转换
- ✅ 自动地址解码
- ✅ 完整的测试套件
- ✅ 详细的文档

## 📁 项目结构

```
chisel/
├── src/
│   ├── main/
│   │   ├── scala/
│   │   │   ├── RiscvAiIntegration.scala    # 🔥 主集成文件
│   │   │   ├── CompactScaleDesign.scala    # AI 加速器
│   │   │   ├── MacUnit.scala               # MAC 单元
│   │   │   ├── MatrixMultiplier.scala      # 矩阵乘法器
│   │   │   └── RiscvAiChipMain.scala       # Verilog 生成
│   │   └── rtl/
│   │       └── picorv32.v                  # PicoRV32 源码
│   └── test/
│       └── scala/
│           └── RiscvAiIntegrationTest.scala # 完整测试套件
├── docs/
│   ├── INTEGRATION.md                       # 集成文档
│   └── TESTING.md                           # 测试文档
├── examples/
│   └── matrix_multiply.c                    # C 语言示例
├── run_integration_tests.sh                 # 测试脚本
├── quick_test.sh                            # 快速测试
└── TEST_SUMMARY.md                          # 测试总结
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保已安装 sbt 和 Java
java -version  # 需要 Java 11+
sbt --version  # 需要 sbt 1.5+
```

### 2. 编译项目

```bash
cd chisel
sbt compile
```

### 3. 运行测试

```bash
# 运行所有测试
sbt test

# 或使用测试脚本
./run_integration_tests.sh

# 快速测试
./quick_test.sh
```

### 4. 生成 Verilog

```bash
# 生成完整系统
sbt "runMain riscv.ai.RiscvAiChipMain"

# 输出: generated/RiscvAiChip.sv
```

## 📊 测试结果

| 测试模块 | 测试用例 | 状态 |
|---------|---------|------|
| MacUnit | 2 | ✅ PASS |
| MatrixMultiplier | 1 | ✅ PASS |
| CompactScaleAiChip | 2 | ✅ PASS |
| RiscvAiChip | 3 | ✅ PASS |
| RiscvAiSystem | 1 | ✅ PASS |
| **总计** | **9** | **✅ ALL PASS** |

## 🏗️ 系统架构

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

## 🗺️ 地址映射

```
CPU 地址空间:
  0x00000000 - 0x7FFFFFFF: 标准内存空间
  0x80000000 - 0x8000FFFF: AI 加速器空间
    ├─ 0x80000000 - 0x800000FF: 矩阵 A (256 bytes)
    ├─ 0x80000100 - 0x800001FF: 矩阵 B (256 bytes)
    ├─ 0x80000200 - 0x800002FF: 结果矩阵 C (256 bytes)
    ├─ 0x80000300: 控制寄存器
    ├─ 0x80000304: 状态寄存器
    └─ 0x80000400+: 内部存储器
```

## 💻 软件编程示例

### C 语言接口

```c
#include <stdint.h>

// AI 加速器基地址
#define AI_BASE 0x80000000

// 写入矩阵 A
void write_matrix_a(int row, int col, int value) {
    volatile int *addr = (int*)(AI_BASE + (row * 8 + col) * 4);
    *addr = value;
}

// 启动矩阵乘法
void start_matmul() {
    volatile int *ctrl = (int*)(AI_BASE + 0x300);
    *ctrl = 1;
}

// 等待完成
void wait_done() {
    volatile int *status = (int*)(AI_BASE + 0x304);
    while ((*status & 0x2) == 0);
}

// 读取结果
int read_result(int row, int col) {
    volatile int *addr = (int*)(AI_BASE + 0x200 + (row * 8 + col) * 4);
    return *addr;
}

// 主函数
int main() {
    // 1. 初始化矩阵
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            write_matrix_a(i, j, i + j);
        }
    }
    
    // 2. 启动计算
    start_matmul();
    
    // 3. 等待完成
    wait_done();
    
    // 4. 读取结果
    int result = read_result(0, 0);
    
    return 0;
}
```

## 📈 性能指标

| 操作 | 延迟 | 吞吐量 |
|------|------|--------|
| MAC 操作 | 2 cycles | 16 ops/cycle |
| 8x8 矩阵乘法 | ~64 cycles | - |
| AXI 读写 | 2-3 cycles | - |
| 峰值性能 | - | 16 GOPS @ 1GHz |

## 📚 文档

- **[INTEGRATION.md](docs/INTEGRATION.md)** - 详细的集成架构文档
- **[TESTING.md](docs/TESTING.md)** - 完整的测试文档
- **[TEST_SUMMARY.md](TEST_SUMMARY.md)** - 测试总结
- **[matrix_multiply.c](examples/matrix_multiply.c)** - C 语言示例

## 🔧 开发指南

### 添加新的测试

```scala
class MyNewTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MyModule"
  
  it should "do something" in {
    test(new MyModule) { dut =>
      // 测试代码
    }
  }
}
```

### 修改 AI 加速器配置

```scala
val aiAccel = Module(new CompactScaleAiChip(
  dataWidth = 32,      // 数据位宽
  matrixSize = 8,      // 矩阵大小
  numMacUnits = 16,    // MAC 单元数
  memoryDepth = 512    // 存储深度
))
```

### 生成波形文件

```scala
test(new MacUnit(32)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // 测试代码
}
```

查看波形:
```bash
gtkwave test_run_dir/*/MacUnit.vcd
```

## 🐛 故障排查

### 编译错误

```bash
# 清理并重新编译
sbt clean
sbt compile
```

### 测试失败

```bash
# 查看详细日志
sbt "testOnly riscv.ai.MacUnitTest" --verbose

# 查看测试输出
cat test_run_dir/*/test.log
```

### 生成 Verilog 失败

```bash
# 检查依赖
sbt update

# 查看完整错误
sbt "runMain riscv.ai.RiscvAiChipMain" 2>&1 | tee verilog_gen.log
```

## 🎓 学习资源

### Chisel 相关
- [Chisel 官方文档](https://www.chisel-lang.org/)
- [ChiselTest 文档](https://github.com/ucb-bar/chiseltest)
- [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp)

### RISC-V 相关
- [RISC-V 规范](https://riscv.org/technical/specifications/)
- [PicoRV32 GitHub](https://github.com/YosysHQ/picorv32)
- [RISC-V 软件工具链](https://github.com/riscv/riscv-gnu-toolchain)

### 硬件设计
- [AXI 协议规范](https://developer.arm.com/documentation/ihi0022/latest/)
- [数字设计最佳实践](https://zipcpu.com/)

## 🤝 贡献

欢迎贡献！请遵循以下步骤:

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

- PicoRV32: ISC License
- 本项目: 根据项目许可证

## 🙏 致谢

- PicoRV32 项目
- Chisel/FIRRTL 团队
- RISC-V 基金会

## 📧 联系方式

如有问题或建议:
- 提交 Issue
- 发起 Pull Request
- 查看文档

---

**状态**: ✅ 集成完成并测试通过  
**版本**: 1.0  
**最后更新**: 2024
