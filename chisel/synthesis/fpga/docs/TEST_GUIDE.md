# FPGA 测试指南

本文档说明如何在 AWS F1 实例上测试 RISC-V AI 加速器 FPGA 镜像。

## 测试环境

### 硬件
- AWS F1 实例（f1.2xlarge 或更高）
- 已加载的 AFI 镜像

### 软件
- AWS FPGA SDK
- 测试程序（C/Python）
- 调试工具（ILA、VIO）

## 测试流程

### 1. 加载 AFI

```bash
# 检查 FPGA 状态
sudo fpga-describe-local-image -S 0

# 加载 AFI
sudo fpga-load-local-image -S 0 -I <your-afi-id>

# 验证加载
sudo fpga-describe-local-image -S 0 -H
# 应显示：Status: loaded, AFI ID: <your-afi-id>
```

### 2. 基础功能测试

#### 2.1 处理器启动测试

```bash
cd chisel/synthesis/fpga/scripts
./test_processor_boot.sh
```

**预期输出：**
```
Testing processor boot...
Reset asserted
Reset released
Processor started
Boot sequence: OK
PASS: Processor boot test
```

#### 2.2 UART 通信测试

```bash
./test_uart.sh
```

**测试内容：**
- 发送字符串 "Hello FPGA"
- 接收回显
- 验证波特率（115200）

**预期输出：**
```
Sending: Hello FPGA
Received: Hello FPGA
PASS: UART communication test
```

#### 2.3 GPIO 测试

```bash
./test_gpio.sh
```

**测试内容：**
- 写入 GPIO 输出寄存器
- 读取 GPIO 输入寄存器
- 测试方向控制

**预期输出：**
```
Writing GPIO: 0xA5A5A5A5
Reading GPIO: 0xA5A5A5A5
PASS: GPIO test
```

### 3. 加速器功能测试

#### 3.1 CompactAccel 测试

```bash
./test_compact_accel.sh
```

**测试用例：**

| 矩阵大小 | 输入 A | 输入 B | 预期输出 |
|---------|--------|--------|----------|
| 2x2 | [[1,2],[3,4]] | [[5,6],[7,8]] | [[19,22],[43,50]] |
| 4x4 | 随机 | 随机 | 软件验证 |
| 8x8 | 随机 | 随机 | 软件验证 |

**预期输出：**
```
Test 1: 2x2 matrix multiplication
  Result: PASS
Test 2: 4x4 matrix multiplication
  Result: PASS
Test 3: 8x8 matrix multiplication
  Result: PASS
PASS: CompactAccel test
```

#### 3.2 BitNetAccel 测试

```bash
./test_bitnet_accel.sh
```

**测试用例：**

| 矩阵大小 | 输入 A (1-bit) | 输入 B (1-bit) | 预期输出 |
|---------|---------------|---------------|----------|
| 2x2 | [[1,0],[1,1]] | [[1,1],[0,1]] | [[1,2],[1,2]] |
| 8x8 | 随机 | 随机 | 软件验证 |

**预期输出：**
```
Test 1: 2x2 BitNet multiplication
  Result: PASS
Test 2: 8x8 BitNet multiplication
  Result: PASS
PASS: BitNetAccel test
```

### 4. 性能测试

#### 4.1 GOPS 测量

```bash
./benchmark_gops.sh
```

**测试方法：**
1. 连续执行 1000 次 8x8 矩阵乘法
2. 测量总时间
3. 计算 GOPS = (操作数 × 次数) / 时间

**预期输出：**
```
CompactAccel Performance:
  Operations: 1024 (8x8x2x8)
  Iterations: 1000
  Time: 160 ms
  GOPS: 6.4
  Target: 6.4 GOPS
  Status: PASS

BitNetAccel Performance:
  Operations: 1024
  Iterations: 1000
  Time: 160 ms
  GOPS: 6.4
  Target: 6.4 GOPS
  Status: PASS
```

#### 4.2 延迟测量

```bash
./benchmark_latency.sh
```

**测试内容：**
- 单次 8x8 矩阵乘法延迟
- 启动开销
- 数据传输时间

**预期输出：**
```
Latency Measurement:
  Matrix size: 8x8
  Computation cycles: 64
  Startup overhead: 10 cycles
  Total latency: 74 cycles (740 ns @ 100MHz)
  Target: <100 cycles
  Status: PASS
```

### 5. 压力测试

#### 5.1 连续运行测试

```bash
./stress_test.sh --duration 3600  # 运行 1 小时
```

**测试内容：**
- 连续执行加速器操作
- 监控错误率
- 检查稳定性

#### 5.2 随机测试

```bash
./random_test.sh --iterations 10000
```

**测试内容：**
- 随机矩阵输入
- 随机操作序列
- 边界条件测试

### 6. 调试

#### 6.1 使用 ILA 查看波形

```bash
# 连接到 Vivado Hardware Manager
vivado -mode tcl
connect_hw_server -url localhost:3121
open_hw_target

# 触发 ILA
run_hw_ila hw_ila_1

# 查看波形
display_hw_ila_data [get_hw_ila_data hw_ila_1]
```

#### 6.2 读取调试寄存器

```bash
# 读取处理器状态
./read_debug_regs.sh

# 输出示例：
# PC: 0x00001000
# State: RUNNING
# Last instruction: 0x00000013 (nop)
```

## 测试报告

### 生成测试报告

```bash
./generate_test_report.sh
```

报告包含：
- 测试用例列表
- 通过/失败状态
- 性能指标
- 错误日志
- 波形截图

### 报告示例

```
=== FPGA Test Report ===
Date: 2025-11-16
AFI ID: afi-xxxxxxxxx
Instance: f1.2xlarge

Functional Tests:
  ✓ Processor boot
  ✓ UART communication
  ✓ GPIO read/write
  ✓ CompactAccel 2x2
  ✓ CompactAccel 4x4
  ✓ CompactAccel 8x8
  ✓ BitNetAccel 2x2
  ✓ BitNetAccel 8x8
  ✓ Interrupt handling

Performance Tests:
  ✓ CompactAccel GOPS: 6.4 (target: 6.4)
  ✓ BitNetAccel GOPS: 6.4 (target: 6.4)
  ✓ Latency: 74 cycles (target: <100)

Stress Tests:
  ✓ 1-hour continuous run: 0 errors
  ✓ 10000 random tests: 0 errors

Overall: PASS (11/11 tests passed)
```

## 故障排查

### 问题 1：AFI 加载失败

**症状：**
```
Error: Failed to load AFI
```

**解决方案：**
1. 检查 AFI 状态：`aws ec2 describe-fpga-images --fpga-image-ids <afi-id>`
2. 确认 AFI 状态为 "available"
3. 检查实例权限
4. 重启 FPGA：`sudo fpga-clear-local-image -S 0`

### 问题 2：测试失败

**症状：**
```
FAIL: CompactAccel test
Expected: 19, Got: 0
```

**解决方案：**
1. 检查复位信号
2. 验证时钟频率
3. 使用 ILA 查看内部信号
4. 检查内存映射

### 问题 3：性能不达标

**症状：**
```
GOPS: 3.2 (target: 6.4)
```

**解决方案：**
1. 检查实际工作频率
2. 分析流水线停顿
3. 优化数据传输
4. 检查时序违例

## 下一步

测试通过后，可以：
1. 生成详细的性能分析报告
2. 进行功耗测量
3. 准备流片前的最终验证
4. 编写用户文档
