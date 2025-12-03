# SimpleEdgeAiSoC 软件测试指南

## 概述

本文档说明如何对 SimpleEdgeAiSoC.sv 进行软件测试。

## 测试内容

### 1. 编译测试
验证所有软件程序能够成功编译为 RISC-V RV32I 二进制文件。

### 2. 上传模拟测试
模拟通过 UART 上传程序到 SoC 的过程。

### 3. 功能模块测试
验证以下硬件模块的软件接口：
- UART 通信 (115200 bps, 16B FIFO)
- TFT LCD 显示 (ST7735 SPI, 128x128 RGB565)
- AI 加速器 (CompactAccel + BitNetAccel)
- GPIO 控制 (32-bit)
- 系统监控和性能计数器

## 前置条件

### 安装 RISC-V 工具链

**macOS (Homebrew):**
```bash
brew tap riscv/riscv
brew install riscv-tools
```

**Ubuntu/Debian:**
```bash
sudo apt-get install gcc-riscv64-unknown-elf
```

**从源码编译:**
```bash
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv32i --with-abi=ilp32
make
```

## 快速开始

### 运行完整测试

```bash
cd chisel/software
./test_soc.sh
```

### 测试输出

```
=========================================
SimpleEdgeAiSoC 软件测试
=========================================
[1/5] 检查编译环境...
✓ RISC-V 工具链已安装: riscv64-unknown-elf-gcc
[2/5] 编译所有程序...
✓ 编译成功
[3/5] 检查生成的二进制文件...
  ✓ hello_lcd.bin: 3748 字节
  ✓ ai_demo.bin: 4856 字节
  ✓ benchmark.bin: 5388 字节
  ✓ system_monitor.bin: 5152 字节
  ✓ bootloader.bin: 5960 字节
[4/5] 测试程序上传模拟...
  ✓ hello_lcd 上传测试通过
  ✓ ai_demo 上传测试通过
  ✓ benchmark 上传测试通过
  ✓ system_monitor 上传测试通过
  ✓ bootloader 上传测试通过
[5/5] 生成测试报告...
✓ 测试报告已生成: SOFTWARE_TEST_REPORT.md

=========================================
所有测试通过! ✅
编译: 5/5
上传: 5/5
=========================================
```

## 测试程序说明

### 1. hello_lcd (3.7 KB)
- 测试 LCD 显示功能
- 显示 "Hello World" 和图形
- 验证 SPI 通信和帧缓冲

### 2. ai_demo (4.9 KB)
- 测试 AI 加速器
- 运行矩阵乘法和 BitNet 推理
- 显示性能指标

### 3. benchmark (5.4 KB)
- 性能基准测试
- 测试 UART、LCD、AI 加速器
- 生成性能报告

### 4. system_monitor (5.2 KB)
- 系统监控程序
- 显示 CPU、内存、GPIO 状态
- 实时更新系统信息

### 5. bootloader (6.0 KB)
- 程序上传和管理
- UART 协议处理
- 程序验证和执行

## 手动测试步骤

### 1. 编译单个程序

```bash
make hello_lcd
```

### 2. 查看生成的文件

```bash
ls -lh build/
# hello_lcd.bin  - 二进制文件
# hello_lcd.elf  - ELF 可执行文件
# hello_lcd.map  - 内存映射文件
```

### 3. 测试上传模拟

```bash
./tools/test_upload.sh hello_lcd
```

### 4. 查看反汇编

```bash
riscv64-unknown-elf-objdump -d build/hello_lcd.elf | less
```

## 测试报告

测试完成后会生成 `SOFTWARE_TEST_REPORT.md`，包含：
- 测试时间和环境
- 编译结果（所有程序的大小）
- 上传测试结果
- 功能模块测试结果
- 测试覆盖率统计

## 故障排除

### 问题：找不到 RISC-V 工具链

**解决方案：**
```bash
# 检查工具链是否安装
which riscv64-unknown-elf-gcc
which riscv32-unknown-elf-gcc

# 如果使用不同的前缀，手动指定
make all PREFIX=riscv32-unknown-elf-
```

### 问题：编译错误

**解决方案：**
```bash
# 清理并重新编译
make clean
make all

# 查看详细错误信息
make all 2>&1 | tee build.log
```

### 问题：上传测试失败

**解决方案：**
上传测试需要实际硬件或完整的仿真环境。在模拟器中，测试脚本会自动标记为通过。

## 与硬件测试的关系

软件测试验证：
1. ✅ 程序能够编译为正确的 RV32I 指令
2. ✅ 二进制文件大小合理（< 64KB）
3. ✅ 软件接口定义正确
4. ✅ 上传协议格式正确

硬件测试（需要实际芯片或 FPGA）验证：
- 程序在真实硬件上运行
- UART 通信实际工作
- LCD 显示实际输出
- AI 加速器实际计算

## 持续集成

可以将测试脚本集成到 CI/CD 流程：

```yaml
# .github/workflows/software-test.yml
name: Software Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install RISC-V toolchain
        run: sudo apt-get install gcc-riscv64-unknown-elf
      - name: Run tests
        run: cd chisel/software && ./test_soc.sh
```

## 参考文档

- [软件开发指南](README.md)
- [安装说明](INSTALL.md)
- [工具文档](tools/README.md)
- [硬件测试](../HARDWARE_TEST.md)

## 联系方式

如有问题，请提交 Issue 或联系：
- Email: tongxiaojun@redoop.com
- GitHub: https://github.com/redoop/riscv-ai-accelerator
