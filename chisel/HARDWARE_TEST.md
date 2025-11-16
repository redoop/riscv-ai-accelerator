# Hardware Testing Guide

## 测试结果

### 串口设备检测 ✅
- 设备: `/dev/tty.usbserial-0001`
- 状态: 已检测到
- 权限: 可读写 (crw-rw-rw-)

### 上传工具测试 ✅
- Python 工具: 正常工作
- pyserial: 已安装（v3.5）
- 串口通信: 正常

### 上传测试结果
```
Connected to /dev/tty.usbserial-0001 at 115200 bps
Warning: No response from bootloader
Uploading 3708 bytes...
Progress: 100.0%
Upload failed!
```

**分析：**
- ✅ 串口连接成功
- ✅ 数据发送成功
- ⚠️ 没有收到 bootloader 响应

**可能原因：**
1. 设备上还没有烧录 bootloader
2. 设备是其他类型的串口设备
3. 波特率不匹配
4. 设备正在运行其他程序

## 下一步操作

### 如果这是 RISC-V AI SoC 硬件

1. **首先烧录 bootloader**
   ```bash
   # 使用 JTAG 或其他方式烧录 bootloader.bin
   # 或者通过 FPGA 配置加载
   ```

2. **验证 bootloader 运行**
   ```bash
   # 连接串口终端
   screen /dev/tty.usbserial-0001 115200
   
   # 应该看到：
   # RISC-V AI Bootloader v0.2
   # Ready for commands...
   ```

3. **测试 bootloader 命令**
   ```bash
   # 发送 'P' (Ping)
   # 应该收到 'K' 响应
   
   # 发送 'I' (Info)
   # 应该收到系统信息
   ```

4. **上传程序**
   ```bash
   make run PROG=hello_lcd PORT=/dev/tty.usbserial-0001
   ```

### 如果这是其他设备

可以使用模拟器测试上传流程：
```bash
cd chisel/software
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh benchmark
```

## 硬件连接检查清单

- [ ] USB-UART 连接正确
- [ ] 电源供电正常（3.3V）
- [ ] TFT LCD 连接正确
- [ ] SPI 信号连接正确
- [ ] 地线连接良好
- [ ] Bootloader 已烧录
- [ ] 设备正常启动

## 调试步骤

### 1. 检查串口通信
```bash
# 使用 screen 或 minicom
screen /dev/tty.usbserial-0001 115200

# 或使用 Python
python3 -m serial.tools.miniterm /dev/tty.usbserial-0001 115200
```

### 2. 发送测试命令
在串口终端中输入：
- `P` - Ping（应该返回 'K'）
- `I` - Info（应该返回系统信息）
- `L` - LCD Test（应该显示彩色条纹）

### 3. 检查硬件信号
使用逻辑分析仪或示波器检查：
- UART TX/RX 信号
- SPI CLK/MOSI/CS 信号
- LCD 背光信号
- 复位信号

## 仿真测试（无需硬件）

### 使用 Verilator 仿真
```bash
cd chisel
# 生成 Verilog
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 使用 Verilator 仿真（需要额外配置）
# verilator --cc generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

### 使用 ChiselTest
```bash
# 运行所有测试
sbt test

# 运行特定测试
sbt "testOnly riscv.ai.peripherals.RealUARTTest"
sbt "testOnly riscv.ai.peripherals.TFTLCDTest"
```

### 使用上传模拟器
```bash
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh ai_demo
./tools/test_upload.sh benchmark
./tools/test_upload.sh system_monitor
./tools/test_upload.sh bootloader
```

## 总结

当前状态：
- ✅ 软件工具正常工作
- ✅ 串口设备已检测到
- ✅ 可以发送数据
- ⚠️ 需要在设备上运行 bootloader

建议：
1. 使用模拟器测试上传流程
2. 在 FPGA 上实现硬件
3. 烧录 bootloader
4. 然后进行实际硬件测试

---

**测试日期:** 2025-11-16  
**设备:** /dev/tty.usbserial-0001  
**状态:** 设备检测成功，等待 bootloader
