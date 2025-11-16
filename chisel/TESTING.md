# RISC-V AI SoC 测试指南

本文档介绍如何运行和使用各种测试。

## 快速开始

```bash
cd chisel

# 运行所有测试
sbt test

# 或使用便捷脚本
./test.sh all
```

## 测试类型

### 1. 硬件单元测试（ChiselTest）

使用 ChiselTest 框架进行硬件模块的单元测试。

#### 运行所有测试
```bash
sbt test
```

#### 运行特定模块测试

**UART 控制器测试**
```bash
./test.sh uart
# 或
sbt "testOnly riscv.ai.peripherals.RealUARTTest"
```

测试内容：
- 初始化和配置
- 波特率设置
- TX/RX 使能
- 字节发送和接收
- FIFO 操作
- 中断生成

**TFT LCD 控制器测试**
```bash
./test.sh lcd
# 或
sbt "testOnly riscv.ai.peripherals.TFTLCDTest"
```

测试内容：
- 初始化和复位
- 背光控制
- SPI 命令和数据发送
- 显示窗口配置
- 帧缓冲写入

**AI 加速器测试**
```bash
./test.sh ai
# 或
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
```

测试内容：
- CompactAccel 2x2 和 4x4 矩阵乘法
- BitNetAccel 4x4 矩阵乘法
- GPIO 功能测试
- 完整 SoC 集成测试

**PicoRV32 CPU 测试**
```bash
./test.sh cpu
# 或
sbt "testOnly riscv.ai.PicoRV32CoreTest"
```

测试内容：
- RV32I 指令执行
- 内存访问
- 中断处理
- SoC 集成

#### 运行外设测试
```bash
./test.sh peripherals
# 或
sbt "testOnly riscv.ai.peripherals.*"
```

#### 快速测试（跳过长时间测试）
```bash
./test.sh quick
```

### 2. 软件上传模拟器

模拟程序上传到硬件的过程，无需真实硬件。

```bash
cd chisel/software

# 模拟上传 hello_lcd 程序
./tools/test_upload.sh hello_lcd

# 模拟上传 AI 演示程序
./tools/test_upload.sh ai_demo

# 模拟上传性能测试
./tools/test_upload.sh benchmark

# 模拟上传系统监控
./tools/test_upload.sh system_monitor

# 模拟上传 bootloader
./tools/test_upload.sh bootloader
```

输出示例：
```
=== RISC-V AI SoC Program Upload Simulator ===

📦 Program: hello_lcd
📊 Size: 3708 bytes

🔌 Connecting to device...
✅ Connected (simulated)

📤 Uploading program...
Progress: 100% [====================]
✅ Upload complete!

🚀 Running program...

=== Hello LCD Output ===
UART initialized at 115200 bps
LCD initialized
Displaying: Hello RISC-V!
Animation running...
Heartbeat: . . . . .

✅ Program is running on device (simulated)
```

### 3. Verilog 生成和仿真

生成 Verilog 代码用于综合或仿真。

```bash
cd chisel

# 生成 Verilog 代码
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 生成的文件在 generated/simple_edgeaisoc/ 目录
ls generated/simple_edgeaisoc/
```

生成的文件：
- `SimpleEdgeAiSoC.sv` - 顶层模块
- `*.sv` - 各个子模块

可以使用 Verilator、Icarus Verilog 或其他工具进行仿真。

## 测试脚本使用

`test.sh` 脚本提供了便捷的测试接口：

```bash
# 查看所有可用选项
./test.sh list

# 运行所有测试
./test.sh all

# 运行特定测试
./test.sh uart          # UART 测试
./test.sh lcd           # LCD 测试
./test.sh ai            # AI 加速器测试
./test.sh soc           # SoC 测试
./test.sh cpu           # CPU 测试
./test.sh peripherals   # 所有外设测试
./test.sh quick         # 快速测试
```

## 测试配置

测试配置在 `build.sbt` 中：

```scala
// Test configuration
Test / testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oD"),
Test / parallelExecution := false,
Test / logBuffered := false
```

配置说明：
- `-oD` - 显示详细的测试持续时间
- `parallelExecution := false` - 串行执行测试（避免资源冲突）
- `logBuffered := false` - 实时显示测试输出

## 查看测试结果

### 波形文件

ChiselTest 会生成 VCD 波形文件，位于 `test_run_dir/` 目录：

```bash
# 查看生成的波形文件
ls test_run_dir/

# 使用 GTKWave 查看波形
gtkwave test_run_dir/RealUART_should_transmit_a_byte/RealUART.vcd
```

### 测试报告

测试报告位于 `target/test-reports/` 目录：

```bash
# 查看测试报告
ls target/test-reports/
```

## 持续集成

可以在 CI/CD 中运行测试：

```yaml
# .github/workflows/test.yml 示例
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: coursier/setup-action@v1
        with:
          jvm: adopt:11
      - name: Run tests
        run: |
          cd chisel
          sbt test
```

## 调试技巧

### 1. 启用波形输出

在测试中添加 `WriteVcdAnnotation`：

```scala
test(new MyModule()).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // 测试代码
}
```

### 2. 增加超时时间

```scala
test(new MyModule()) { dut =>
  dut.clock.setTimeout(0)  // 禁用超时
  // 或
  dut.clock.setTimeout(10000)  // 设置更长的超时
}
```

### 3. 打印调试信息

```scala
println(s"Status: ${dut.io.status.peek().litValue}")
```

### 4. 单步调试

```scala
for (i <- 0 until 100) {
  println(s"Cycle $i")
  dut.clock.step(1)
  println(s"  Output: ${dut.io.output.peek().litValue}")
}
```

## 常见问题

### 测试超时

如果测试超时，可以：
1. 增加超时时间：`dut.clock.setTimeout(10000)`
2. 禁用超时：`dut.clock.setTimeout(0)`
3. 检查是否有死锁或无限循环

### 波形文件太大

可以：
1. 减少测试周期数
2. 只在需要时启用波形输出
3. 使用 `test_run_dir/` 清理脚本

### 测试失败

1. 查看详细错误信息
2. 检查波形文件
3. 添加调试打印
4. 单步执行测试

## 性能测试

运行性能基准测试：

```bash
cd chisel/software
make benchmark
./tools/test_upload.sh benchmark
```

输出示例：
```
=== Benchmark Output ===
Performance Benchmark
Testing UART...
Testing LCD...
Testing Graphics...
Testing AI...

=== Results ===
UART: 11520 B/s
LCD: 625K px/s
Graphics: 15 FPS
AI: 6 GOPS
```

## 总结

测试层次：
1. **单元测试** - ChiselTest 测试各个模块
2. **集成测试** - 测试模块间的交互
3. **系统测试** - 完整 SoC 测试
4. **软件测试** - 上传模拟器测试

推荐工作流：
1. 开发新功能时，先写单元测试
2. 使用 `./test.sh quick` 快速验证
3. 提交前运行 `sbt test` 完整测试
4. 使用上传模拟器测试软件集成

---

**更多信息：**
- [开发计划](docs/DEV_PLAN_V0.2.md)
- [硬件测试](HARDWARE_TEST.md)
- [软件工具](software/tools/README.md)
