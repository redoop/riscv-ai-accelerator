# RISC-V AI加速器芯片PyTorch测试使用指南

## 概述

本测试套件为RISC-V AI加速器芯片提供了完整的PyTorch集成测试，包括：

- **CPU基准测试** - 提供性能对比基线
- **AI加速器功能测试** - 验证TPU/VPU功能
- **端到端性能评估** - 完整的AI工作负载测试
- **自动化测试流程** - 一键运行所有测试

## 快速开始

### 1. 基本测试（无需硬件）

```bash
# 运行CPU基准测试
make -f Makefile test-simple
```

这将运行CPU基准测试，输出类似：

```
=== CPU基准性能测试 ===
矩阵乘法测试:
  64x64: 0.000003s, 175.36 GFLOPS
  128x128: 0.000009s, 476.62 GFLOPS
  256x256: 0.000054s, 617.13 GFLOPS
  512x512: 0.000334s, 803.96 GFLOPS

=== 简单神经网络测试 ===
Batch size 1: 0.000838s, 1193.52 samples/sec
Batch size 4: 0.001607s, 2489.27 samples/sec
```

### 2. 检查系统环境

```bash
# 查看系统信息
make -f Makefile info

# 检查依赖
make -f Makefile check-deps

# 详细硬件检查（推荐）
make -f Makefile check-hardware

# 或者快速检查
make -f Makefile check-hardware-quick
```

### 3. 完整测试（需要RISC-V AI硬件）

```bash
# 运行综合测试
make -f Makefile test-comprehensive
```

## 测试结果解读

### 性能指标说明

1. **GFLOPS** - 每秒十亿次浮点运算，衡量计算性能
2. **samples/sec** - 每秒处理的样本数，衡量推理吞吐量
3. **加速比** - AI加速器相对于CPU的性能提升倍数
4. **准确性** - AI加速器结果与CPU结果的一致性

### 典型性能数据

基于测试结果，在Apple M1芯片上的CPU基准性能：

| 操作类型 | 性能指标 | 备注 |
|---------|---------|------|
| 矩阵乘法 (512x512) | 803.96 GFLOPS | 大矩阵性能更好 |
| 神经网络推理 (batch=4) | 2489.27 samples/sec | 批处理提升吞吐量 |
| ReLU激活 (1M元素) | 0.27ms | 向量化操作很快 |
| Sigmoid激活 (1M元素) | 1.00ms | 复杂函数较慢 |

### 内存访问模式分析

测试显示了内存访问模式对性能的影响：

```
连续内存访问:
  1024x1024: 行访问=0.014s, 列访问=0.011s, 比率=0.81
  8192x8192: 行访问=0.064s, 列访问=0.586s, 比率=9.18
```

- **行访问**：连续内存访问，缓存友好
- **列访问**：跳跃内存访问，缓存不友好
- **比率增长**：矩阵越大，内存访问模式影响越明显

## 与RISC-V AI加速器对比

当RISC-V AI后端可用时，测试将显示加速比：

```
=== 基本操作测试 ===
矩阵乘法测试:
  64x64: CPU=0.0012s, AI=0.0002s, 加速比=6.00x, 准确性=✓
  512x512: CPU=0.0712s, AI=0.0071s, 加速比=10.03x, 准确性=✓

=== 神经网络模型测试 ===
AI推理时间: 0.1234s (100次)
加速比: 7.25x
推理吞吐量: 810.37 inferences/sec
```

### 预期加速效果

基于RISC-V AI加速器的设计规格：

| 操作类型 | 预期加速比 | 说明 |
|---------|-----------|------|
| 矩阵乘法 | 5-15x | TPU专门优化 |
| 卷积操作 | 3-10x | 空间局部性好 |
| 激活函数 | 2-8x | VPU向量化 |
| 完整神经网络 | 3-12x | 综合优化效果 |

## 故障排除

### 常见问题

1. **PyTorch未安装**
   ```bash
   make -f Makefile install-deps
   ```

2. **权限问题**
   ```bash
   # 添加用户到ai_accel组
   sudo usermod -a -G ai_accel $USER
   # 重新登录或重启
   ```

3. **设备文件不存在**
   ```bash
   # 检查硬件连接
   lspci | grep -i ai
   # 检查内核模块
   lsmod | grep ai_accel
   ```

4. **性能异常**
   - 检查CPU频率调节器设置
   - 确保系统负载较低
   - 关闭不必要的后台程序

### 调试模式

启用详细输出：

```bash
# 设置调试环境变量
export RISCV_AI_DEBUG=1
export PYTORCH_DEBUG=1

# 运行测试
python3 scripts/pytorch_chip_test.py --output debug_results.json
```

## 自定义测试

### 添加新的测试用例

1. **修改scripts/simple_chip_test.py**添加CPU基准测试
2. **修改scripts/pytorch_chip_test.py**添加AI加速器测试

示例：添加新的矩阵操作测试

```python
def test_custom_matrix_op():
    """自定义矩阵操作测试"""
    print("测试自定义矩阵操作...")
    
    # 准备测试数据
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    
    # CPU基准
    start_time = time.time()
    result_cpu = torch.matmul(a.T, b)  # 转置矩阵乘法
    cpu_time = time.time() - start_time
    
    # AI加速（如果可用）
    if BACKEND_AVAILABLE:
        start_time = time.time()
        result_ai = riscv_ai_backend.matmul_transpose(a, b)
        ai_time = time.time() - start_time
        
        speedup = cpu_time / ai_time
        accuracy = torch.allclose(result_cpu, result_ai, rtol=1e-4)
        
        print(f"转置矩阵乘法: CPU={cpu_time:.6f}s, AI={ai_time:.6f}s, "
              f"加速比={speedup:.2f}x, 准确性={'✓' if accuracy else '✗'}")
```

### 测试特定模型

```python
# 加载预训练模型
import torchvision.models as models
model = models.resnet18(pretrained=True)

# 运行测试
if BACKEND_AVAILABLE:
    model_id = runtime.load_model_from_torch(model, "resnet18", optimize=True)
    stats = runtime.benchmark_model(model_id, (1, 3, 224, 224))
    print(f"ResNet18吞吐量: {stats['throughput']:.2f} fps")
```

## 性能优化建议

### 1. 数据类型选择

- **FP32**: 最高精度，适合训练和高精度推理
- **FP16**: 平衡精度和性能，推荐用于推理
- **INT8**: 最高性能，适合量化推理

### 2. 批处理大小

- 小批处理：低延迟，适合实时应用
- 大批处理：高吞吐量，适合批量处理

### 3. 内存优化

- 使用连续内存布局
- 避免频繁的内存分配
- 预分配缓冲区

### 4. 模型优化

- 启用操作融合
- 使用量化技术
- 优化网络结构

## 结果分析工具

### 生成性能报告

```bash
# 运行完整测试并生成报告
make -f Makefile test-comprehensive
make -f Makefile benchmark
```

### 可视化结果

```python
import json
import matplotlib.pyplot as plt

# 加载测试结果
with open('test_results/comprehensive_results.json', 'r') as f:
    results = json.load(f)

# 绘制加速比图表
operations = []
speedups = []

for op, data in results['basic_operations'].items():
    if data.get('speedup'):
        operations.append(op)
        speedups.append(data['speedup'])

plt.figure(figsize=(10, 6))
plt.bar(operations, speedups)
plt.title('RISC-V AI加速器性能提升')
plt.ylabel('加速比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('speedup_chart.png')
```

## 总结

这个测试套件提供了：

1. **完整的性能基准** - CPU和AI加速器对比
2. **自动化测试流程** - 一键运行所有测试
3. **详细的结果分析** - 多维度性能评估
4. **易于扩展** - 支持添加自定义测试

通过这些测试，你可以：

- 验证RISC-V AI加速器的功能正确性
- 评估实际工作负载的性能提升
- 识别性能瓶颈和优化机会
- 为应用部署提供性能参考

开始使用：

```bash
# 1. 运行基本测试
make -f Makefile test-simple

# 2. 检查硬件（如果有）
make -f Makefile check-hardware

# 3. 运行完整测试（如果硬件可用）
make -f Makefile test-comprehensive
```