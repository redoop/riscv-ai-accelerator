# RISC-V AI加速器芯片PyTorch测试程序

本目录包含用于测试RISC-V AI加速器芯片的PyTorch测试程序，提供全面的性能评估和功能验证。

## 文件说明

### 测试程序
- `scripts/pytorch_chip_test.py` - 综合测试程序，包含完整的AI加速器功能测试
- `scripts/simple_chip_test.py` - 简化测试程序，提供CPU基准性能数据
- `Makefile` - 构建和测试自动化脚本（已集成PyTorch测试功能）

### 测试内容

#### 1. 基本操作测试
- **矩阵乘法**: 测试不同尺寸的矩阵乘法性能
- **2D卷积**: 测试卷积操作的加速效果
- **激活函数**: ReLU、Sigmoid、Tanh等激活函数性能
- **池化操作**: MaxPool2D、AvgPool2D性能测试

#### 2. 神经网络测试
- **简单CNN**: 卷积神经网络端到端性能
- **ResNet块**: 残差网络模块性能
- **Transformer注意力**: 自注意力机制性能

#### 3. 高级功能测试
- **模型优化**: 自动模型优化和操作融合
- **量化**: INT8/INT16量化性能和精度
- **内存性能**: 内存分配和数据传输带宽
- **并发执行**: 多TPU并发处理能力

## 快速开始

### 1. 环境准备

```bash
# 检查依赖
make check-deps

# 安装依赖（如果需要）
make install-deps
```

### 2. 运行简单测试

```bash
# 运行CPU基准测试
make test-simple
```

这将运行CPU基准测试，提供性能对比基线。

### 3. 检查硬件状态

```bash
# 检查RISC-V AI加速器硬件连接
make -f Makefile.pytorch_test check-hardware
```

### 4. 运行综合测试

```bash
# 构建AI后端并运行完整测试
make -f Makefile.pytorch_test test-comprehensive
```

### 5. 生成性能报告

```bash
# 生成详细性能基准报告
make -f Makefile.pytorch_test benchmark
```

## 详细使用说明

### 命令行选项

#### pytorch_chip_test.py
```bash
python3 scripts/pytorch_chip_test.py [选项]

选项:
  --output, -o FILE     测试结果输出文件 (默认: test_results.json)
  --no-profiling        禁用性能分析
  --quick              快速测试模式
```

#### 示例
```bash
# 运行完整测试并保存结果
python3 scripts/pytorch_chip_test.py --output my_results.json

# 快速测试模式
python3 scripts/pytorch_chip_test.py --quick

# 禁用性能分析的测试
python3 scripts/pytorch_chip_test.py --no-profiling
```

### Makefile目标

```bash
# 显示所有可用目标
make -f Makefile.pytorch_test help

# 主要目标:
make -f Makefile.pytorch_test all              # 默认：检查依赖并运行简单测试
make -f Makefile.pytorch_test test-simple      # CPU基准测试
make -f Makefile.pytorch_test test-comprehensive # 完整AI加速器测试
make -f Makefile.pytorch_test test-quick       # 快速测试
make -f Makefile.pytorch_test benchmark        # 性能基准报告
make -f Makefile.pytorch_test clean            # 清理构建文件
```

## 测试结果解读

### 性能指标

1. **加速比 (Speedup)**: AI加速器相对于CPU的性能提升倍数
2. **吞吐量 (Throughput)**: 每秒处理的推理次数
3. **延迟 (Latency)**: 单次推理的平均时间
4. **准确性 (Accuracy)**: AI加速器结果与CPU结果的一致性

### 典型输出示例

```
=== 基本操作测试 ===
矩阵乘法测试:
  64x64: CPU=0.0012s, AI=0.0002s, 加速比=6.00x, 准确性=✓
  128x128: CPU=0.0089s, AI=0.0011s, 加速比=8.09x, 准确性=✓
  256x256: CPU=0.0712s, AI=0.0071s, 加速比=10.03x, 准确性=✓

卷积测试:
  3x32x32->16: CPU=0.0045s, AI=0.0008s, 加速比=5.63x, 准确性=✓
  32x64x64->64: CPU=0.0234s, AI=0.0029s, 加速比=8.07x, 准确性=✓

=== 神经网络模型测试 ===
AI推理时间: 0.1234s (100次)
加速比: 7.25x
推理吞吐量: 810.37 inferences/sec
```

### 结果文件格式

测试结果保存为JSON格式，包含详细的性能数据：

```json
{
  "basic_operations": {
    "matmul_64x64": {
      "cpu_time": 0.0012,
      "ai_time": 0.0002,
      "speedup": 6.0,
      "accuracy": true
    }
  },
  "neural_networks": {
    "simple_net": {
      "speedup": 7.25,
      "benchmark_stats": {
        "throughput": 810.37,
        "mean_time": 0.001234
      }
    }
  },
  "device_info": {
    "tpu_count": 2,
    "vpu_count": 2,
    "backend_available": true
  }
}
```

## 故障排除

### 常见问题

1. **RISC-V AI后端不可用**
   ```
   ⚠ RISC-V AI后端不可用: No module named 'riscv_ai_backend'
   ```
   - 检查软件框架是否正确构建
   - 运行 `make -f Makefile.pytorch_test build-backend`

2. **设备文件不存在**
   ```
   ⚠ 未找到AI加速器设备文件 /dev/ai_accel
   ```
   - 检查硬件连接
   - 确认驱动程序已加载
   - 运行 `make -f Makefile.pytorch_test check-hardware`

3. **权限问题**
   ```
   Permission denied: /dev/ai_accel
   ```
   - 添加用户到ai_accel组: `sudo usermod -a -G ai_accel $USER`
   - 或使用sudo运行测试

4. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 减少批处理大小
   - 使用 `--quick` 模式
   - 检查系统内存使用情况

### 调试模式

启用详细日志输出：

```bash
# 设置环境变量启用调试
export RISCV_AI_DEBUG=1
export PYTORCH_DEBUG=1

# 运行测试
python3 scripts/pytorch_chip_test.py
```

### 性能分析

如果性能不如预期：

1. **检查硬件利用率**
   ```bash
   # 监控TPU使用率
   watch -n 1 'cat /sys/class/ai_accel/tpu*/utilization'
   ```

2. **分析内存带宽**
   - 查看测试结果中的内存性能数据
   - 确认数据传输不是瓶颈

3. **检查模型优化**
   - 确保启用了模型优化 (`optimize=True`)
   - 查看操作融合是否生效

## 扩展测试

### 添加自定义模型测试

```python
# 在scripts/pytorch_chip_test.py中添加自定义模型
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义你的模型结构
        
    def forward(self, x):
        # 定义前向传播
        return x

# 在测试函数中使用
def test_custom_model(self):
    model = CustomModel()
    # 添加测试逻辑
```

### 添加新的操作测试

```python
def test_new_operation(self):
    """测试新的AI操作"""
    # 准备测试数据
    input_data = torch.randn(...)
    
    # CPU基准
    cpu_result = cpu_operation(input_data)
    
    # AI加速
    if BACKEND_AVAILABLE:
        ai_result = riscv_ai_backend.new_operation(input_data)
        # 比较结果和性能
```

## 贡献指南

1. **代码风格**: 遵循PEP 8 Python代码规范
2. **测试**: 为新功能添加相应测试
3. **文档**: 更新README和代码注释
4. **性能**: 确保新测试不显著增加运行时间

## 许可证

本项目遵循Apache License 2.0许可证。详见主项目LICENSE文件。

## 支持

如有问题或建议：

1. 查看故障排除部分
2. 检查现有测试用例
3. 查阅硬件文档
4. 提交issue到项目仓库