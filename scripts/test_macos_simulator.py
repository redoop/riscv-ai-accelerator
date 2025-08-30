#!/usr/bin/env python3
"""
macOS RISC-V AI仿真器测试程序
验证仿真器的基本功能
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
from pathlib import Path

# 添加当前目录到 Python 路径以导入 riscv_ai_backend
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_simulator():
    """测试基本仿真器功能"""
    print("🧪 测试基本仿真器功能")
    
    try:
        import riscv_ai_backend as ai
        
        print("✅ 仿真器导入成功")
        print(f"📊 设备信息: {ai.get_device_info()}")
        print(f"🔢 设备数量: {ai.device_count()}")
        print(f"🔍 后端可用: {ai.is_available()}")
        
        # 测试矩阵乘法
        print("\n🧮 测试矩阵乘法:")
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)
        
        start_time = time.time()
        result = ai.mm(a, b)
        end_time = time.time()
        
        print(f"  输入: {a.shape} @ {b.shape}")
        print(f"  输出: {result.shape}")
        print(f"  时间: {end_time - start_time:.6f}s")
        
        # 测试激活函数
        print("\n🎯 测试激活函数:")
        x = torch.randn(1000)
        
        relu_result = ai.relu(x)
        sigmoid_result = ai.sigmoid(x)
        
        print(f"  ReLU: {x.shape} -> {relu_result.shape}")
        print(f"  Sigmoid: {x.shape} -> {sigmoid_result.shape}")
        
        # 测试卷积
        print("\n🔄 测试卷积:")
        input_tensor = torch.randn(1, 3, 32, 32)
        weight = torch.randn(16, 3, 3, 3)
        
        conv_result = ai.conv2d(input_tensor, weight, stride=[1, 1], padding=[1, 1])
        print(f"  卷积: {input_tensor.shape} -> {conv_result.shape}")
        
        # 性能统计
        print("\n📈 性能统计:")
        stats = ai.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_runtime():
    """测试运行时功能"""
    print("\n🏃 测试运行时功能")
    
    try:
        from runtime import create_runtime
        
        # 创建运行时
        runtime = create_runtime(enable_profiling=True)
        print("✅ 运行时创建成功")
        
        # 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = SimpleModel()
        model.eval()
        
        # 加载模型
        model_id = runtime.load_model_from_torch(model, "test_model", optimize=True)
        print(f"✅ 模型加载成功: {model_id}")
        
        # 测试推理
        test_input = torch.randn(5, 10)
        output = runtime.infer(model_id, test_input)
        print(f"✅ 推理成功: {test_input.shape} -> {output.shape}")
        
        # 基准测试
        benchmark_stats = runtime.benchmark_model(model_id, (5, 10), 
                                                num_iterations=50, 
                                                warmup_iterations=5)
        print(f"✅ 基准测试完成: {benchmark_stats['throughput']:.2f} inferences/sec")
        
        # 模型信息
        model_info = runtime.get_model_info(model_id)
        print(f"✅ 模型参数: {model_info['total_parameters']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 运行时测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """测试性能对比"""
    print("\n⚡ 测试性能对比")
    
    try:
        import riscv_ai_backend as ai
        
        # 测试不同大小的矩阵乘法
        sizes = [(64, 64), (128, 128), (256, 256)]
        
        for m, n in sizes:
            a = torch.randn(m, n)
            b = torch.randn(n, m)
            
            # CPU基准
            start_time = time.time()
            cpu_result = torch.mm(a, b)
            cpu_time = time.time() - start_time
            
            # AI仿真
            start_time = time.time()
            ai_result = ai.mm(a, b)
            ai_time = time.time() - start_time
            
            # 计算加速比（注意：仿真器可能比CPU慢）
            speedup = cpu_time / ai_time if ai_time > 0 else 0
            
            print(f"  {m}x{n}: CPU={cpu_time:.6f}s, AI={ai_time:.6f}s, 比率={speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🍎 macOS RISC-V AI仿真器测试")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # 运行测试
    if test_basic_simulator():
        success_count += 1
    
    if test_runtime():
        success_count += 1
    
    if test_performance_comparison():
        success_count += 1
    
    # 总结
    print("\n" + "=" * 50)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！仿真器工作正常")
        print("\n📋 下一步:")
        print("1. 运行完整测试: make -f Makefile.pytorch_test test-macos")
        print("2. 查看性能对比: python3 simple_chip_test.py")
        print("3. 自定义测试: 修改 test_macos_simulator.py")
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    # 清理
    try:
        import riscv_ai_backend
        riscv_ai_backend.cleanup()
        print("🧹 资源已清理")
    except:
        pass

if __name__ == "__main__":
    main()