#!/usr/bin/env python3
"""
RISC-V AI加速器芯片简化测试程序
适用于没有完整PyTorch集成时的基本功能测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from pathlib import Path

def test_cpu_baseline():
    """CPU基准测试"""
    print("=== CPU基准性能测试 ===")
    
    # 矩阵乘法测试
    print("矩阵乘法测试:")
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for m, n in sizes:
        a = torch.randn(m, n, dtype=torch.float32)
        b = torch.randn(n, m, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            _ = torch.mm(a, b)
        
        # 测试
        start_time = time.time()
        for _ in range(100):
            result = torch.mm(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        gflops = (2 * m * n * m) / (avg_time * 1e9)  # 2*M*N*K operations
        
        print(f"  {m}x{n}: {avg_time:.6f}s, {gflops:.2f} GFLOPS")
    
    # 卷积测试
    print("\n卷积测试:")
    conv_configs = [
        (1, 3, 32, 32, 16, 3),    # batch, in_ch, h, w, out_ch, kernel
        (1, 32, 64, 64, 64, 3),
        (1, 64, 128, 128, 128, 3),
    ]
    
    for batch, in_ch, h, w, out_ch, kernel in conv_configs:
        input_tensor = torch.randn(batch, in_ch, h, w, dtype=torch.float32)
        weight = torch.randn(out_ch, in_ch, kernel, kernel, dtype=torch.float32)
        bias = torch.randn(out_ch, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            _ = F.conv2d(input_tensor, weight, bias, padding=1)
        
        # 测试
        start_time = time.time()
        for _ in range(50):
            result = F.conv2d(input_tensor, weight, bias, padding=1)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50
        print(f"  {in_ch}x{h}x{w}->{out_ch}: {avg_time:.6f}s")
    
    # 激活函数测试
    print("\n激活函数测试:")
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        input_tensor = torch.randn(size, dtype=torch.float32)
        
        # ReLU
        start_time = time.time()
        for _ in range(1000):
            result = F.relu(input_tensor)
        relu_time = (time.time() - start_time) / 1000
        
        # Sigmoid
        start_time = time.time()
        for _ in range(1000):
            result = torch.sigmoid(input_tensor)
        sigmoid_time = (time.time() - start_time) / 1000
        
        print(f"  Size {size}: ReLU={relu_time:.8f}s, Sigmoid={sigmoid_time:.8f}s")


def test_simple_neural_network():
    """简单神经网络测试"""
    print("\n=== 简单神经网络测试 ===")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 创建模型和输入
    model = SimpleNet()
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # 测试
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                output = model(input_tensor)
            end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        throughput = batch_size / avg_time
        
        print(f"Batch size {batch_size}: {avg_time:.6f}s, {throughput:.2f} samples/sec")


def test_memory_patterns():
    """内存访问模式测试"""
    print("\n=== 内存访问模式测试 ===")
    
    # 连续内存访问
    print("连续内存访问:")
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        data = torch.randn(size, size, dtype=torch.float32)
        
        # 行优先访问
        start_time = time.time()
        for i in range(size):
            row_sum = torch.sum(data[i, :])
        row_time = time.time() - start_time
        
        # 列优先访问
        start_time = time.time()
        for j in range(size):
            col_sum = torch.sum(data[:, j])
        col_time = time.time() - start_time
        
        print(f"  {size}x{size}: 行访问={row_time:.6f}s, 列访问={col_time:.6f}s, "
              f"比率={col_time/row_time:.2f}")


def test_data_types():
    """数据类型性能测试"""
    print("\n=== 数据类型性能测试 ===")
    
    size = (1024, 1024)
    dtypes = [torch.float32, torch.float16, torch.int32, torch.int8]
    
    for dtype in dtypes:
        try:
            if dtype in [torch.int8, torch.int32]:
                # 整数类型使用不同的操作
                a = torch.randint(-128, 127, size, dtype=dtype)
                b = torch.randint(-128, 127, size, dtype=dtype)
                
                start_time = time.time()
                for _ in range(50):
                    result = a + b  # 整数加法
                end_time = time.time()
            elif dtype == torch.float16:
                # float16在某些平台可能不支持矩阵乘法，使用加法测试
                a = torch.randn(size, dtype=dtype)
                b = torch.randn(size, dtype=dtype)
                
                start_time = time.time()
                for _ in range(50):
                    result = a + b  # 使用加法而不是矩阵乘法
                end_time = time.time()
            else:
                a = torch.randn(size, dtype=dtype)
                b = torch.randn(size, dtype=dtype)
                
                start_time = time.time()
                for _ in range(50):
                    result = torch.mm(a, b)
                end_time = time.time()
            
            avg_time = (end_time - start_time) / 50
            print(f"  {dtype}: {avg_time:.6f}s")
            
        except Exception as e:
            print(f"  {dtype}: 不支持 ({e})")


def test_ai_workloads():
    """AI工作负载模拟测试"""
    print("\n=== AI工作负载模拟测试 ===")
    
    # 模拟ResNet-like块
    class ResNetBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return F.relu(out)
    
    # 测试不同通道数的ResNet块
    channels = [32, 64, 128, 256]
    input_size = 32
    
    for ch in channels:
        block = ResNetBlock(ch)
        block.eval()
        
        input_tensor = torch.randn(1, ch, input_size, input_size)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = block(input_tensor)
        
        # 测试
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                output = block(input_tensor)
            end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"ResNet块 {ch}通道: {avg_time:.6f}s")
    
    # 模拟Transformer注意力机制
    print("\nTransformer注意力机制:")
    
    def scaled_dot_product_attention(q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, v)
    
    seq_lengths = [64, 128, 256, 512]
    d_model = 512
    
    for seq_len in seq_lengths:
        q = torch.randn(1, 8, seq_len, d_model // 8)  # 8 heads
        k = torch.randn(1, 8, seq_len, d_model // 8)
        v = torch.randn(1, 8, seq_len, d_model // 8)
        
        # 预热
        for _ in range(10):
            _ = scaled_dot_product_attention(q, k, v)
        
        # 测试
        start_time = time.time()
        for _ in range(50):
            output = scaled_dot_product_attention(q, k, v)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50
        print(f"  序列长度 {seq_len}: {avg_time:.6f}s")


def generate_summary_report():
    """生成测试摘要报告"""
    print("\n" + "=" * 60)
    print("RISC-V AI加速器芯片测试摘要报告")
    print("=" * 60)
    
    print("\n测试环境:")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  Python版本: {sys.version}")
    print(f"  CPU核心数: {os.cpu_count()}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"  CUDA可用: 是 (设备数: {torch.cuda.device_count()})")
    else:
        print(f"  CUDA可用: 否")
    
    print("\n测试说明:")
    print("  - 本测试程序提供CPU基准性能数据")
    print("  - 可用于与RISC-V AI加速器性能对比")
    print("  - 测试涵盖矩阵运算、卷积、神经网络等AI工作负载")
    
    print("\n优化建议:")
    print("  - 矩阵乘法是AI加速的重点优化目标")
    print("  - 卷积操作在CNN中占主要计算量")
    print("  - 内存访问模式影响缓存效率")
    print("  - 数据类型选择影响计算精度和性能")
    
    print("\n下一步:")
    print("  1. 集成RISC-V AI后端")
    print("  2. 运行完整的pytorch_chip_test.py")
    print("  3. 对比CPU和AI加速器性能")
    print("  4. 分析性能瓶颈和优化机会")


def main():
    print("RISC-V AI加速器芯片简化测试程序")
    print("提供CPU基准性能数据，用于与AI加速器对比")
    
    try:
        # 运行各项测试
        test_cpu_baseline()
        test_simple_neural_network()
        test_memory_patterns()
        test_data_types()
        test_ai_workloads()
        
        # 生成摘要报告
        generate_summary_report()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()