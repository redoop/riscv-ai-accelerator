# Linux环境下RISC-V AI加速器硬件检查示例

## 预期的硬件检查输出（Linux系统 + RISC-V AI硬件）

当在Linux系统上连接了RISC-V AI加速器硬件时，硬件检查的预期输出如下：

```bash
$ make -f Makefile.pytorch_test check-hardware
```

### 预期输出示例

```
RISC-V AI加速器硬件检查工具
检查硬件连接、驱动程序和系统兼容性
============================================================

============================================================
RISC-V AI加速器硬件检查报告
============================================================
=== 系统信息 ===
Os: Linux
Os Version: #1 SMP PREEMPT_DYNAMIC Ubuntu 22.04.3 LTS
Architecture: x86_64
Python Version: 3.10.12
Kernel: 6.2.0-39-generic

=== RISC-V AI加速器兼容性 ===
✓ Linux系统 - 支持RISC-V AI加速器

=== 设备文件检查 ===
✓ 找到设备: /dev/ai_accel
  权限: 666
  所有者: 0:0
✓ 找到设备: /dev/ai_accel0
  权限: 666
  所有者: 0:0
✓ 找到设备: /dev/ai_accel1
  权限: 666
  所有者: 0:0
✗ 未找到设备: /dev/riscv_ai
✓ 找到设备: /dev/tpu0
  权限: 666
  所有者: 0:0
✓ 找到设备: /dev/tpu1
  权限: 666
  所有者: 0:0

=== 内核模块检查 ===
✓ 找到模块: ai_accel              16384  0
✓ 找到模块: tpu_driver            32768  2
✓ 找到模块: vpu_driver            24576  1

=== PCIe设备检查 ===
✓ 可能的AI设备: 01:00.0 Processing accelerators: RISC-V AI Accelerator Corp Device 1234
✓ 可能的AI设备: 02:00.0 Processing accelerators: RISC-V AI Accelerator Corp Device 1235

=== 软件依赖检查 ===
✓ torch (PyTorch深度学习框架): 2.0.1
✓ numpy (数值计算库): 1.23.5
✓ pybind11 (Python C++绑定库): 3.0.1

=== PyTorch后端检查 ===
✓ PyTorch版本: 2.0.1
✓ CUDA可用: True
✓ CUDA设备数: 1
  设备 0: NVIDIA GeForce RTX 4090
✓ RISC-V AI后端: 可用
✓ 后端状态: 可用
✓ AI设备数: 2

=== 检查总结 ===
✅ 硬件状态: RISC-V AI加速器已正确安装和配置
✅ 建议: 可以运行完整的AI加速器测试

=== 下一步建议 ===
1. 运行完整测试:
   make -f Makefile.pytorch_test test-comprehensive
2. 检查性能基准:
   make -f Makefile.pytorch_test benchmark

📄 详细报告已保存到: test_results/hardware_check_report.json
```

### 预期的JSON报告内容

```json
{
  "timestamp": "2025-08-30T19:04:32.296075",
  "system_info": {
    "os": "Linux",
    "os_version": "#1 SMP PREEMPT_DYNAMIC Ubuntu 22.04.3 LTS",
    "architecture": "x86_64",
    "python_version": "3.10.12",
    "kernel": "6.2.0-39-generic"
  },
  "compatible": true,
  "devices_found": [
    "/dev/ai_accel",
    "/dev/ai_accel0", 
    "/dev/ai_accel1",
    "/dev/tpu0",
    "/dev/tpu1"
  ],
  "kernel_modules": [
    "ai_accel",
    "tpu_driver", 
    "vpu_driver"
  ],
  "pcie_devices": [
    "01:00.0 Processing accelerators: RISC-V AI Accelerator Corp Device 1234",
    "02:00.0 Processing accelerators: RISC-V AI Accelerator Corp Device 1235"
  ],
  "dependencies": {
    "torch": "2.0.1",
    "numpy": "1.23.5",
    "pybind11": "3.0.1"
  }
}
```

## 硬件安装和配置步骤

### 1. 硬件连接
```bash
# 检查PCIe插槽连接
lspci | grep -i accelerator

# 检查电源连接
dmesg | grep -i "ai accelerator"
```

### 2. 驱动安装
```bash
# 编译和安装驱动
cd software/drivers
make all
sudo make install

# 加载内核模块
sudo modprobe ai_accel
sudo modprobe tpu_driver
sudo modprobe vpu_driver
```

### 3. 设备文件权限
```bash
# 设置设备文件权限
sudo chmod 666 /dev/ai_accel*
sudo chmod 666 /dev/tpu*

# 或者添加用户到设备组
sudo usermod -a -G ai_accel $USER
```

### 4. 验证安装
```bash
# 运行硬件检查
make -f Makefile.pytorch_test check-hardware

# 运行基本功能测试
make -f Makefile.pytorch_test test-simple
```

## 故障排除

### 常见问题和解决方案

#### 1. 设备文件不存在
```bash
# 检查内核模块是否加载
lsmod | grep ai_accel

# 手动加载模块
sudo modprobe ai_accel

# 检查dmesg日志
dmesg | tail -20
```

#### 2. 权限被拒绝
```bash
# 检查设备文件权限
ls -l /dev/ai_accel*

# 修改权限
sudo chmod 666 /dev/ai_accel*

# 或添加用户到组
sudo usermod -a -G ai_accel $USER
newgrp ai_accel
```

#### 3. PCIe设备未识别
```bash
# 检查PCIe设备
lspci -v | grep -A 10 -i accelerator

# 重新扫描PCIe总线
echo 1 | sudo tee /sys/bus/pci/rescan

# 检查BIOS设置
# 确保PCIe插槽已启用
```

#### 4. 驱动编译失败
```bash
# 安装内核头文件
sudo apt-get install linux-headers-$(uname -r)

# 安装构建工具
sudo apt-get install build-essential

# 重新编译驱动
cd software/drivers
make clean
make all
```

## 性能验证

### 预期的性能测试结果

当硬件正常工作时，运行完整测试应该看到类似的加速比：

```
=== 基本操作测试 ===
矩阵乘法测试:
  64x64: CPU=0.0012s, AI=0.0002s, 加速比=6.00x, 准确性=✓
  128x128: CPU=0.0089s, AI=0.0011s, 加速比=8.09x, 准确性=✓
  256x256: CPU=0.0712s, AI=0.0071s, 加速比=10.03x, 准确性=✓
  512x512: CPU=0.5678s, AI=0.0456s, 加速比=12.45x, 准确性=✓

卷积测试:
  3x32x32->16: CPU=0.0045s, AI=0.0008s, 加速比=5.63x, 准确性=✓
  32x64x64->64: CPU=0.0234s, AI=0.0029s, 加速比=8.07x, 准确性=✓
  64x128x128->128: CPU=0.1945s, AI=0.0156s, 加速比=12.47x, 准确性=✓

=== 神经网络模型测试 ===
AI推理时间: 0.1234s (100次)
加速比: 7.25x
推理吞吐量: 810.37 inferences/sec

=== 量化测试 ===
INT8量化: 加速比=15.23x, 准确性损失=0.000123
INT16量化: 加速比=8.45x, 准确性损失=0.000045

=== 并发执行测试 ===
串行时间: 0.2345s
并发时间: 0.0678s
并发加速比: 3.46x (使用2个TPU)
```

## 总结

这个硬件检查工具提供了：

1. **全面的系统兼容性检查** - 自动检测操作系统和架构
2. **详细的硬件状态报告** - 设备文件、内核模块、PCIe设备
3. **软件依赖验证** - PyTorch和相关库的安装状态
4. **明确的故障排除指导** - 针对不同问题的解决方案
5. **性能验证建议** - 下一步测试和优化建议

在Linux系统上正确安装RISC-V AI加速器后，用户可以期待看到显著的AI工作负载加速效果。