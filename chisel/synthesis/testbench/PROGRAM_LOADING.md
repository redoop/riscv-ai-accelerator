# 后综合仿真程序加载说明

## 问题

后综合网表是门级网表，无法直接访问内部内存实例来加载程序。

## 已编译的测试程序

- **源文件**: `test.c`, `start.S`
- **二进制**: `test.elf`, `test.bin`, `test.hex`
- **GPIO 地址**: 0x20020000
- **功能**: 循环输出 0xAAAAAAAA 和 0x55555555 到 GPIO

## 解决方案

### 方案 1: RTL 仿真（推荐）

使用 Chisel/Verilator 进行 RTL 仿真，可以直接加载程序：

```bash
cd /opt/github/riscv-ai-accelerator/chisel
sbt "testOnly *SimpleEdgeAiSoCTest"
```

### 方案 2: 添加 Bootloader

修改设计添加一个小的 ROM bootloader，通过 UART 加载程序到 RAM。

### 方案 3: 预初始化内存

在综合时使用 `$readmemh` 初始化内存，但这需要修改 RTL。

### 方案 4: 仅验证硬件（当前方案）

后综合仿真主要验证：
- ✅ 综合工具正确性
- ✅ 时序约束
- ✅ 信号连接
- ✅ 门级行为

软件功能测试应在 RTL 阶段完成。

## 当前仿真状态

- ✅ 网表编译成功
- ✅ 信号连接正确
- ✅ 时钟和复位工作
- ✅ GPIO/IRQ 信号可观察
- ⚠ CPU 执行未初始化内存（TRAP 正常）

## 查看波形

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis
./view_wave.sh
```

关键信号：
- `clock`, `reset`
- `io_gpio_out[31:0]`
- `io_trap`
- `io_compact_irq`, `io_bitnet_irq`
