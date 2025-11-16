# 测试集成总结

## 完成的工作

已成功将所有硬件模块测试集成到 `sbt test` 命令中，并创建了便捷的测试工具。

### 1. 更新 build.sbt

添加了测试配置：
```scala
// Test configuration
Test / testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oD"),
Test / parallelExecution := false,
Test / logBuffered := false
```

**配置说明：**
- `-oD` - 显示详细的测试持续时间
- `parallelExecution := false` - 串行执行测试，避免资源冲突
- `logBuffered := false` - 实时显示测试输出

### 2. 创建测试脚本 (test.sh)

创建了便捷的测试脚本 `chisel/test.sh`，支持以下命令：

```bash
./test.sh all          # 运行所有测试
./test.sh uart         # UART 控制器测试
./test.sh lcd          # TFT LCD 控制器测试
./test.sh ai           # AI 加速器测试
./test.sh soc          # 完整 SoC 测试
./test.sh cpu          # PicoRV32 CPU 测试
./test.sh peripherals  # 所有外设测试
./test.sh quick        # 快速测试
./test.sh list         # 显示帮助信息
```

### 3. 创建测试文档 (TESTING.md)

创建了完整的测试指南文档，包含：
- 快速开始
- 测试类型说明
- 各模块测试详情
- 调试技巧
- 常见问题解答

### 4. 更新 README.md

在主 README 中添加了测试文档的链接。

## 现有测试模块

项目中已包含以下测试：

### 外设测试
1. **RealUARTTest** - UART 控制器测试
   - 初始化和配置
   - 波特率设置
   - TX/RX 使能
   - 字节发送和接收
   - FIFO 操作
   - 中断生成

2. **TFTLCDTest** - TFT LCD 控制器测试
   - 初始化和复位
   - 背光控制
   - SPI 命令和数据发送
   - 显示窗口配置
   - 帧缓冲写入

### SoC 测试
3. **SimpleEdgeAiSoCTest** - 完整 SoC 测试
   - CompactAccel 2x2 矩阵乘法
   - CompactAccel 4x4 矩阵乘法
   - BitNetAccel 4x4 矩阵乘法
   - GPIO 功能测试
   - 综合系统测试

### CPU 测试
4. **PicoRV32CoreTest** - CPU 测试
   - RV32I 指令执行
   - 内存访问
   - 中断处理
   - SoC 集成

### 加速器测试
5. **BitNetAccelDebugTest** - BitNet 加速器调试测试
6. **SimpleCompactAccelDebugTest** - Compact 加速器调试测试

## 使用方法

### 运行所有测试
```bash
cd chisel
sbt test
# 或
./test.sh all
```

### 运行特定测试
```bash
# 使用脚本（推荐）
./test.sh uart
./test.sh lcd
./test.sh ai

# 或直接使用 sbt
sbt "testOnly riscv.ai.peripherals.RealUARTTest"
sbt "testOnly riscv.ai.peripherals.TFTLCDTest"
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
```

### 快速测试
```bash
./test.sh quick
```

## 软件模拟测试

除了硬件测试，还可以使用软件模拟器测试上传流程：

```bash
cd chisel/software

# 模拟上传各种程序
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh ai_demo
./tools/test_upload.sh benchmark
./tools/test_upload.sh system_monitor
./tools/test_upload.sh bootloader
```

## 测试输出

### 成功示例
```
=== RISC-V AI SoC 测试套件 ===

运行 UART 测试...
[info] RealUARTTest:
[info] RealUART
[info] - should initialize correctly
[info] - should configure baud rate
[info] - should enable TX and RX
[info] - should transmit a byte
[info] - should fill TX FIFO
[info] - should generate TX interrupt
[info] - should generate RX interrupt
[info] Run completed in 5 seconds.
[info] Total number of tests run: 7
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 7, failed 0, canceled 0, ignored 1, pending 0
[info] All tests passed.

✓ 测试完成
```

### 波形文件

测试会生成 VCD 波形文件，位于 `test_run_dir/` 目录：
```bash
ls test_run_dir/
gtkwave test_run_dir/RealUART_should_transmit_a_byte/RealUART.vcd
```

## 文档结构

```
chisel/
├── TESTING.md              # 测试指南（新增）
├── TEST_SUMMARY.md         # 本文件（新增）
├── test.sh                 # 测试脚本（新增）
├── build.sbt               # 更新了测试配置
├── README.md               # 更新了文档链接
├── src/test/scala/         # 测试源代码
│   ├── RealUARTTest.scala
│   ├── TFTLCDTest.scala
│   ├── SimpleEdgeAiSoCTest.scala
│   ├── PicoRV32CoreTest.scala
│   ├── BitNetAccelDebugTest.scala
│   └── SimpleCompactAccelDebugTest.scala
└── software/tools/
    └── test_upload.sh      # 软件上传模拟器
```

## 下一步

测试系统已经完整集成，可以：

1. **开发新功能时**：先写测试，然后实现功能
2. **提交代码前**：运行 `sbt test` 确保所有测试通过
3. **快速验证**：使用 `./test.sh quick` 进行快速测试
4. **调试问题**：查看 VCD 波形文件和测试输出

## 总结

✅ 所有测试已集成到 `sbt test`  
✅ 创建了便捷的测试脚本 `test.sh`  
✅ 编写了完整的测试文档 `TESTING.md`  
✅ 更新了主 README 文档  
✅ 支持硬件测试和软件模拟测试  

现在可以通过简单的命令运行所有测试：
```bash
cd chisel
./test.sh all
```

或运行特定测试：
```bash
./test.sh uart
./test.sh lcd
./test.sh ai
```
