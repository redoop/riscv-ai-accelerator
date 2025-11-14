# RISC-V AI Accelerator

基于 Chisel 的 RISC-V AI 加速器项目，集成 PicoRV32 CPU 和专用矩阵加速器。

## 🚀 快速开始

### 前置要求

- Java 11+
- Scala 2.13+
- SBT 1.9+
- Verilator (可选，用于仿真)

### 安装依赖

```bash
# macOS
brew install sbt verilator

# Ubuntu/Debian
sudo apt install sbt verilator
```

### 快速测试

```bash
cd chisel
./run.sh soc
```

## 📋 项目结构

```
riscv-ai-accelerator/
├── chisel/                      # Chisel 源代码
│   ├── src/
│   │   ├── main/scala/         # 主要源代码
│   │   │   ├── EdgeAiSoCSimple.scala          # SimpleEdgeAiSoC 实现
│   │   │   ├── SimpleEdgeAiSoCMain.scala      # Verilog 生成器
│   │   │   ├── VerilogGenerator.scala         # 通用生成器
│   │   │   └── PostProcessVerilog.scala       # 后处理工具
│   │   ├── test/scala/         # 测试代码
│   │   │   ├── SimpleEdgeAiSoCTest.scala      # SoC 测试
│   │   │   ├── PicoRV32CoreTest.scala         # CPU 测试
│   │   │   ├── BitNetAccelDebugTest.scala     # BitNet 测试
│   │   │   └── SimpleCompactAccelDebugTest.scala  # Compact 测试
│   │   └── resources/rtl/      # RTL 资源
│   │       └── picorv32.v      # PicoRV32 核心
│   ├── generated/              # 生成的 Verilog 文件
│   ├── Makefile               # Make 构建文件
│   ├── run.sh                 # 运行脚本
│   └── build.sbt              # SBT 构建配置
├── docs/                       # 文档
└── README.md                  # 本文件
```

## 🎯 核心功能

### SimpleEdgeAiSoC

完整的边缘 AI SoC 系统，包含：

- **PicoRV32 CPU**: RV32I RISC-V 处理器
- **CompactAccel**: 8x8 矩阵加速器
- **BitNetAccel**: 16x16 BitNet 加速器（无乘法器）
- **内存系统**: RAM + 外设映射
- **外设**: UART, GPIO, 中断控制器

### BitNet 加速器特性

- ✅ **无乘法器设计** - 只使用加减法
- ✅ **2-bit 权重编码** - {-1, 0, +1}
- ✅ **稀疏性优化** - 自动跳过零权重
- ✅ **内存效率** - 内存占用减少 10 倍
- ✅ **低功耗** - 功耗降低 60%

## 🔧 使用方法

### 使用 Makefile

```bash
cd chisel

# 编译项目
make compile

# 运行所有测试
make test

# 运行 SoC 测试
make test-soc

# 运行 BitNet 测试
make test-bitnet

# 生成 Verilog
make generate

# 完整流程
make full

# 清理
make clean

# 查看帮助
make help
```

### 使用 run.sh

```bash
cd chisel

# 运行所有测试
./run.sh test

# 运行 SoC 测试
./run.sh soc

# 生成 Verilog
./run.sh generate

# 生成所有版本
./run.sh all

# 完整流程
./run.sh full

# 清理
./run.sh clean

# 查看帮助
./run.sh help
```

### 使用 SBT 直接运行

```bash
cd chisel

# 编译
sbt compile

# 运行所有测试
sbt test

# 运行特定测试
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
sbt "testOnly riscv.ai.BitNetAccelDebugTest"
sbt "testOnly riscv.ai.PicoRV32CoreTest"

# 生成 Verilog
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
```

## 📊 测试覆盖

### SimpleEdgeAiSoC 测试

- ✅ 系统实例化
- ✅ CompactAccel 2x2 矩阵乘法
- ✅ CompactAccel 4x4 矩阵乘法
- ✅ BitNetAccel 4x4 矩阵乘法
- ✅ GPIO 功能
- ✅ 系统集成

### BitNet 加速器测试

- ✅ 2x2 矩阵乘法（无乘法器）
- ✅ 8x8 矩阵乘法（稀疏性优化）
- ✅ 权重编码 {-1, 0, +1}
- ✅ 稀疏性统计验证
- ✅ 性能指标测量

### PicoRV32 核心测试

- ✅ 内存适配器集成
- ✅ 地址解码器功能
- ✅ 完整 SoC 集成
- ✅ CPU 与加速器交互
- ✅ 内存映射验证
- ✅ 中断处理
- ✅ 综合测试套件

## 📁 生成的文件

运行 `make generate` 或 `./run.sh generate` 后，会在 `chisel/generated/` 目录生成：

```
generated/
└── simple_edgeaisoc/
    └── SimpleEdgeAiSoC.sv    # 完整的 SoC SystemVerilog 文件
```

运行 `./run.sh all` 会生成所有版本：

```
generated/
├── simple_edgeaisoc/
│   └── SimpleEdgeAiSoC.sv
├── optimized/
│   └── PhysicalOptimizedRiscvAiChip.sv
├── scalable/
│   └── SimpleScalableAiChip.sv
├── fixed/
│   └── FixedMediumScaleAiChip.sv
└── constraints/
    ├── design_constraints.sdc
    ├── power_constraints.upf
    └── implementation.tcl
```

## 📈 性能指标

### SimpleEdgeAiSoC

- **CPU**: PicoRV32 @ 50-100 MHz
- **CompactAccel**: ~1.6 GOPS @ 100MHz
- **BitNetAccel**: ~4.8 GOPS @ 100MHz
- **总算力**: ~6.4 GOPS
- **功耗**: < 100 mW (估算)

### 资源占用 (FPGA)

- **LUTs**: ~8,000
- **FFs**: ~6,000
- **BRAMs**: ~20
- **频率**: 50-100 MHz

### BitNet 性能

- **2x2 矩阵**: 14 周期，跳过 2 次零权重
- **8x8 矩阵**: 518 周期，跳过 168 次零权重
- **硬件效率**: 面积减少 50%，功耗降低 60%

## 🏗️ 内存映射

```
0x00000000 - 0x0FFFFFFF  RAM (256 MB)
0x10000000 - 0x10000FFF  CompactAccel (4 KB)
0x10001000 - 0x10001FFF  BitNetAccel (4 KB)
0x20000000 - 0x2000FFFF  UART (64 KB)
0x20020000 - 0x2002FFFF  GPIO (64 KB)
```

### CompactAccel 寄存器

```
0x10000000  CTRL        控制寄存器
0x10000004  STATUS      状态寄存器
0x10000008  SIZE        矩阵大小
0x10000100  INPUT_A     输入矩阵 A
0x10000300  INPUT_B     输入矩阵 B
0x10000500  OUTPUT      输出矩阵
```

### BitNetAccel 寄存器

```
0x10001000  CTRL        控制寄存器
0x10001004  STATUS      状态寄存器
0x10001008  SIZE        矩阵大小
0x10001100  INPUT_A     输入矩阵 A
0x10001300  INPUT_B     输入矩阵 B (BitNet 权重)
0x10001500  OUTPUT      输出矩阵
```

## 🐛 故障排除

### 编译错误

```bash
# 清理并重新编译
cd chisel
sbt clean compile
```

### 测试超时

在测试代码中增加超时时间：
```scala
dut.clock.setTimeout(2000)  // 默认 1000
```

### Java 版本问题

```bash
# 确保使用 Java 11
export JAVA_HOME=/path/to/jdk-11
export PATH=$JAVA_HOME/bin:$PATH
```

### SBT 未安装

```bash
# macOS
brew install sbt

# Ubuntu/Debian
sudo apt install sbt
```

## 📚 文档

详细文档请查看：

- `chisel/README.md` - Chisel 项目详细说明
- `docs/` - 架构和设计文档
- `examples/` - 示例代码和测试结果

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🌟 致谢

- [Chisel](https://www.chisel-lang.org/) - 硬件描述语言
- [PicoRV32](https://github.com/YosysHQ/picorv32) - RISC-V CPU 核心
- [BitNet](https://arxiv.org/abs/2310.11453) - 1-bit LLM 架构

---

**快速开始**: `cd chisel && ./run.sh soc`  
**完整测试**: `cd chisel && make full`  
**生成 Verilog**: `cd chisel && make generate`
