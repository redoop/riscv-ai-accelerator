#!/usr/bin/env python3
"""
RISC-V AI加速器芯片PyTorch综合测试程序
测试TPU、VPU和AI指令扩展的性能和功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import os
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional

# 添加软件框架路径和 scripts 路径
sys.path.insert(0, str(Path(__file__).parent / "software" / "frameworks" / "pytorch"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    # macOS仿真器支持
    import platform
    if platform.system() == "Darwin":
        print("🍎 检测到macOS系统，启用仿真模式")
    
    import riscv_ai_backend
    from model_optimizer import RiscvAiOptimizer, RiscvAiQuantizer
    from runtime import RiscvAiRuntime, create_runtime
    BACKEND_AVAILABLE = True
    print("✓ RISC-V AI后端可用")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠ RISC-V AI后端不可用: {e}")
    print("将运行CPU基准测试")


class ChipTestSuite:
    """RISC-V AI加速器芯片测试套件"""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.results = {}
        
        if BACKEND_AVAILABLE:
            # 初始化RISC-V AI后端
            riscv_ai_backend.initialize()
            self.runtime = create_runtime(enable_profiling=enable_profiling)
            self.optimizer = RiscvAiOptimizer()
            self.quantizer = RiscvAiQuantizer()
            
            # 获取设备信息
            self.device_info = self.runtime.get_device_info()
            print(f"✓ 检测到 {self.device_info.get('tpu_count', 0)} 个TPU")
            print(f"✓ 检测到 {self.device_info.get('vpu_count', 0)} 个VPU")
        else:
            self.runtime = None
            self.optimizer = None
            self.quantizer = None
            self.device_info = {}
    
    def test_basic_operations(self) -> Dict:
        """测试基本AI操作"""
        print("\n=== 基本操作测试 ===")
        results = {}
        
        # 矩阵乘法测试
        print("测试矩阵乘法...")
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
        
        for m, n in sizes:
            a = torch.randn(m, n, dtype=torch.float32)
            b = torch.randn(n, m, dtype=torch.float32)
            
            # CPU基准
            start_time = time.time()
            result_cpu = torch.mm(a, b)
            cpu_time = time.time() - start_time
            
            # RISC-V AI加速
            if BACKEND_AVAILABLE:
                start_time = time.time()
                result_ai = riscv_ai_backend.mm(a, b)
                ai_time = time.time() - start_time
                
                # 验证正确性
                accuracy = torch.allclose(result_cpu, result_ai, rtol=1e-4, atol=1e-5)
                speedup = cpu_time / ai_time if ai_time > 0 else 0
                
                results[f"matmul_{m}x{n}"] = {
                    "cpu_time": cpu_time,
                    "ai_time": ai_time,
                    "speedup": speedup,
                    "accuracy": accuracy
                }
                
                print(f"  {m}x{n}: CPU={cpu_time:.4f}s, AI={ai_time:.4f}s, "
                      f"加速比={speedup:.2f}x, 准确性={'✓' if accuracy else '✗'}")
            else:
                results[f"matmul_{m}x{n}"] = {
                    "cpu_time": cpu_time,
                    "ai_time": None,
                    "speedup": None,
                    "accuracy": None
                }
                print(f"  {m}x{n}: CPU={cpu_time:.4f}s")
        
        # 卷积测试
        print("\n测试2D卷积...")
        conv_configs = [
            (1, 3, 32, 32, 16, 3),    # 输入通道, 输出通道, 高, 宽, 卷积核大小
            (1, 32, 64, 64, 64, 3),
            (1, 64, 128, 128, 128, 3),
        ]
        
        for batch, in_ch, h, w, out_ch, kernel in conv_configs:
            input_tensor = torch.randn(batch, in_ch, h, w, dtype=torch.float32)
            weight = torch.randn(out_ch, in_ch, kernel, kernel, dtype=torch.float32)
            bias = torch.randn(out_ch, dtype=torch.float32)
            
            # CPU基准
            start_time = time.time()
            result_cpu = F.conv2d(input_tensor, weight, bias, padding=1)
            cpu_time = time.time() - start_time
            
            # RISC-V AI加速
            if BACKEND_AVAILABLE:
                start_time = time.time()
                result_ai = riscv_ai_backend.conv2d(input_tensor, weight, bias,
                                                   stride=[1, 1], padding=[1, 1],
                                                   dilation=[1, 1], groups=1)
                ai_time = time.time() - start_time
                
                accuracy = torch.allclose(result_cpu, result_ai, rtol=1e-3, atol=1e-4)
                speedup = cpu_time / ai_time if ai_time > 0 else 0
                
                results[f"conv2d_{in_ch}x{h}x{w}_{out_ch}"] = {
                    "cpu_time": cpu_time,
                    "ai_time": ai_time,
                    "speedup": speedup,
                    "accuracy": accuracy
                }
                
                print(f"  {in_ch}x{h}x{w}->{out_ch}: CPU={cpu_time:.4f}s, AI={ai_time:.4f}s, "
                      f"加速比={speedup:.2f}x, 准确性={'✓' if accuracy else '✗'}")
            else:
                results[f"conv2d_{in_ch}x{h}x{w}_{out_ch}"] = {
                    "cpu_time": cpu_time,
                    "ai_time": None,
                    "speedup": None,
                    "accuracy": None
                }
                print(f"  {in_ch}x{h}x{w}->{out_ch}: CPU={cpu_time:.4f}s")
        
        # 激活函数测试
        print("\n测试激活函数...")
        input_sizes = [1000, 10000, 100000]
        
        for size in input_sizes:
            input_tensor = torch.randn(size, dtype=torch.float32)
            
            # ReLU测试
            start_time = time.time()
            result_cpu_relu = F.relu(input_tensor)
            cpu_relu_time = time.time() - start_time
            
            if BACKEND_AVAILABLE:
                start_time = time.time()
                result_ai_relu = riscv_ai_backend.relu(input_tensor)
                ai_relu_time = time.time() - start_time
                
                accuracy = torch.allclose(result_cpu_relu, result_ai_relu)
                speedup = cpu_relu_time / ai_relu_time if ai_relu_time > 0 else 0
                
                results[f"relu_{size}"] = {
                    "cpu_time": cpu_relu_time,
                    "ai_time": ai_relu_time,
                    "speedup": speedup,
                    "accuracy": accuracy
                }
                
                print(f"  ReLU({size}): CPU={cpu_relu_time:.6f}s, AI={ai_relu_time:.6f}s, "
                      f"加速比={speedup:.2f}x")
        
        return results
    
    def test_neural_networks(self) -> Dict:
        """测试神经网络模型"""
        print("\n=== 神经网络模型测试 ===")
        results = {}
        
        # 定义测试模型
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
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # 创建模型和输入
        model = SimpleNet()
        model.eval()
        sample_input = torch.randn(1, 3, 32, 32)
        
        # CPU基准测试
        print("CPU基准测试...")
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                output_cpu = model(sample_input)
            cpu_time = time.time() - start_time
        
        print(f"CPU推理时间: {cpu_time:.4f}s (100次)")
        
        if BACKEND_AVAILABLE:
            # 加载和优化模型
            print("加载模型到RISC-V AI运行时...")
            model_id = self.runtime.load_model_from_torch(model, "simple_net",
                                                        optimize=True, 
                                                        sample_input=sample_input)
            
            # AI加速推理测试
            print("RISC-V AI加速推理测试...")
            start_time = time.time()
            for _ in range(100):
                output_ai = self.runtime.infer(model_id, sample_input)
            ai_time = time.time() - start_time
            
            # 验证准确性
            accuracy = torch.allclose(output_cpu, output_ai, rtol=1e-3, atol=1e-4)
            speedup = cpu_time / ai_time if ai_time > 0 else 0
            
            print(f"AI推理时间: {ai_time:.4f}s (100次)")
            print(f"加速比: {speedup:.2f}x")
            print(f"准确性: {'✓' if accuracy else '✗'}")
            
            # 性能基准测试
            print("运行详细性能基准...")
            benchmark_stats = self.runtime.benchmark_model(model_id, (1, 3, 32, 32),
                                                         num_iterations=1000,
                                                         warmup_iterations=100)
            
            results["simple_net"] = {
                "cpu_time": cpu_time,
                "ai_time": ai_time,
                "speedup": speedup,
                "accuracy": accuracy,
                "benchmark_stats": benchmark_stats
            }
            
            print(f"平均推理时间: {benchmark_stats['mean_time']:.6f}s")
            print(f"吞吐量: {benchmark_stats['throughput']:.2f} inferences/sec")
            print(f"标准差: {benchmark_stats['std_time']:.6f}s")
        
        return results
    
    def test_quantization(self) -> Dict:
        """测试量化功能"""
        print("\n=== 量化测试 ===")
        results = {}
        
        if not BACKEND_AVAILABLE:
            print("跳过量化测试 - RISC-V AI后端不可用")
            return results
        
        # 创建测试模型
        model = models.resnet18(pretrained=False)
        model.eval()
        
        # 创建校准数据
        print("准备校准数据...")
        calibration_data = []
        for _ in range(100):
            data = torch.randn(1, 3, 224, 224)
            target = torch.randint(0, 1000, (1,))
            calibration_data.append((data, target))
        
        calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=1)
        
        # 测试不同量化方案
        quantization_schemes = ["int8", "int16"]
        
        for scheme in quantization_schemes:
            print(f"\n测试{scheme.upper()}量化...")
            
            try:
                # 量化模型
                quantized_model = self.quantizer.quantize_model(model, calibration_loader, scheme)
                
                # 测试推理
                sample_input = torch.randn(1, 3, 224, 224)
                
                with torch.no_grad():
                    # 原始模型
                    start_time = time.time()
                    output_fp32 = model(sample_input)
                    fp32_time = time.time() - start_time
                    
                    # 量化模型
                    start_time = time.time()
                    output_quantized = quantized_model(sample_input)
                    quantized_time = time.time() - start_time
                
                # 计算准确性损失
                accuracy_loss = torch.mean(torch.abs(output_fp32 - output_quantized)).item()
                speedup = fp32_time / quantized_time if quantized_time > 0 else 0
                
                results[f"quantization_{scheme}"] = {
                    "fp32_time": fp32_time,
                    "quantized_time": quantized_time,
                    "speedup": speedup,
                    "accuracy_loss": accuracy_loss
                }
                
                print(f"  FP32时间: {fp32_time:.6f}s")
                print(f"  {scheme.upper()}时间: {quantized_time:.6f}s")
                print(f"  加速比: {speedup:.2f}x")
                print(f"  准确性损失: {accuracy_loss:.6f}")
                
            except Exception as e:
                print(f"  {scheme.upper()}量化失败: {e}")
                results[f"quantization_{scheme}"] = {"error": str(e)}
        
        return results
    
    def test_memory_performance(self) -> Dict:
        """测试内存性能"""
        print("\n=== 内存性能测试 ===")
        results = {}
        
        if not BACKEND_AVAILABLE:
            print("跳过内存性能测试 - RISC-V AI后端不可用")
            return results
        
        # 测试不同大小的内存分配和传输
        memory_sizes = [1, 10, 100, 1000]  # MB
        
        for size_mb in memory_sizes:
            size_bytes = size_mb * 1024 * 1024
            num_elements = size_bytes // 4  # float32
            
            print(f"测试 {size_mb}MB 内存传输...")
            
            # 创建测试数据
            data = torch.randn(num_elements, dtype=torch.float32)
            
            # 测试内存分配
            start_time = time.time()
            device_data = riscv_ai_backend.allocate_memory(size_bytes)
            alloc_time = time.time() - start_time
            
            # 测试数据传输 (Host -> Device)
            start_time = time.time()
            riscv_ai_backend.copy_to_device(data, device_data)
            h2d_time = time.time() - start_time
            
            # 测试数据传输 (Device -> Host)
            start_time = time.time()
            result_data = riscv_ai_backend.copy_from_device(device_data, num_elements)
            d2h_time = time.time() - start_time
            
            # 释放内存
            riscv_ai_backend.free_memory(device_data)
            
            # 计算带宽
            h2d_bandwidth = size_mb / h2d_time if h2d_time > 0 else 0
            d2h_bandwidth = size_mb / d2h_time if d2h_time > 0 else 0
            
            results[f"memory_{size_mb}mb"] = {
                "alloc_time": alloc_time,
                "h2d_time": h2d_time,
                "d2h_time": d2h_time,
                "h2d_bandwidth_mbps": h2d_bandwidth,
                "d2h_bandwidth_mbps": d2h_bandwidth
            }
            
            print(f"  分配时间: {alloc_time:.6f}s")
            print(f"  H2D传输: {h2d_time:.6f}s ({h2d_bandwidth:.2f} MB/s)")
            print(f"  D2H传输: {d2h_time:.6f}s ({d2h_bandwidth:.2f} MB/s)")
        
        return results
    
    def test_concurrent_execution(self) -> Dict:
        """测试并发执行"""
        print("\n=== 并发执行测试 ===")
        results = {}
        
        if not BACKEND_AVAILABLE:
            print("跳过并发执行测试 - RISC-V AI后端不可用")
            return results
        
        # 测试多TPU并发
        tpu_count = self.device_info.get('tpu_count', 0)
        if tpu_count > 1:
            print(f"测试 {tpu_count} 个TPU并发执行...")
            
            # 创建多个矩阵乘法任务
            matrices = []
            for i in range(tpu_count):
                a = torch.randn(512, 512, dtype=torch.float32)
                b = torch.randn(512, 512, dtype=torch.float32)
                matrices.append((a, b))
            
            # 串行执行基准
            start_time = time.time()
            for a, b in matrices:
                result = riscv_ai_backend.mm(a, b)
            serial_time = time.time() - start_time
            
            # 并发执行
            start_time = time.time()
            tasks = []
            for i, (a, b) in enumerate(matrices):
                task = riscv_ai_backend.mm_async(a, b, device_id=i % tpu_count)
                tasks.append(task)
            
            # 等待所有任务完成
            results_concurrent = []
            for task in tasks:
                result = riscv_ai_backend.wait_task(task)
                results_concurrent.append(result)
            
            concurrent_time = time.time() - start_time
            
            speedup = serial_time / concurrent_time if concurrent_time > 0 else 0
            
            results["concurrent_tpu"] = {
                "serial_time": serial_time,
                "concurrent_time": concurrent_time,
                "speedup": speedup,
                "tpu_count": tpu_count
            }
            
            print(f"  串行时间: {serial_time:.4f}s")
            print(f"  并发时间: {concurrent_time:.4f}s")
            print(f"  并发加速比: {speedup:.2f}x")
        else:
            print("只有一个TPU，跳过并发测试")
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """运行综合测试"""
        print("开始RISC-V AI加速器芯片综合测试")
        print("=" * 50)
        
        all_results = {}
        
        # 运行各项测试
        all_results["basic_operations"] = self.test_basic_operations()
        all_results["neural_networks"] = self.test_neural_networks()
        all_results["quantization"] = self.test_quantization()
        all_results["memory_performance"] = self.test_memory_performance()
        all_results["concurrent_execution"] = self.test_concurrent_execution()
        
        # 添加设备信息
        all_results["device_info"] = self.device_info
        all_results["backend_available"] = BACKEND_AVAILABLE
        
        return all_results
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None):
        """生成测试报告"""
        print("\n" + "=" * 50)
        print("测试报告摘要")
        print("=" * 50)
        
        if BACKEND_AVAILABLE:
            print(f"✓ RISC-V AI后端: 可用")
            print(f"✓ TPU数量: {self.device_info.get('tpu_count', 0)}")
            print(f"✓ VPU数量: {self.device_info.get('vpu_count', 0)}")
        else:
            print("⚠ RISC-V AI后端: 不可用")
        
        # 基本操作性能摘要
        if "basic_operations" in results:
            print("\n基本操作性能:")
            basic_ops = results["basic_operations"]
            
            # 矩阵乘法性能
            matmul_speedups = [v["speedup"] for k, v in basic_ops.items() 
                             if k.startswith("matmul_") and v.get("speedup")]
            if matmul_speedups:
                avg_speedup = np.mean(matmul_speedups)
                max_speedup = np.max(matmul_speedups)
                print(f"  矩阵乘法平均加速比: {avg_speedup:.2f}x")
                print(f"  矩阵乘法最大加速比: {max_speedup:.2f}x")
            
            # 卷积性能
            conv_speedups = [v["speedup"] for k, v in basic_ops.items() 
                           if k.startswith("conv2d_") and v.get("speedup")]
            if conv_speedups:
                avg_speedup = np.mean(conv_speedups)
                max_speedup = np.max(conv_speedups)
                print(f"  卷积平均加速比: {avg_speedup:.2f}x")
                print(f"  卷积最大加速比: {max_speedup:.2f}x")
        
        # 神经网络性能
        if "neural_networks" in results and "simple_net" in results["neural_networks"]:
            nn_result = results["neural_networks"]["simple_net"]
            if nn_result.get("speedup"):
                print(f"\n神经网络推理加速比: {nn_result['speedup']:.2f}x")
                if "benchmark_stats" in nn_result:
                    stats = nn_result["benchmark_stats"]
                    print(f"推理吞吐量: {stats['throughput']:.2f} inferences/sec")
        
        # 量化性能
        if "quantization" in results:
            print("\n量化性能:")
            for scheme in ["int8", "int16"]:
                key = f"quantization_{scheme}"
                if key in results["quantization"] and "speedup" in results["quantization"][key]:
                    speedup = results["quantization"][key]["speedup"]
                    loss = results["quantization"][key]["accuracy_loss"]
                    print(f"  {scheme.upper()}: 加速比={speedup:.2f}x, 准确性损失={loss:.6f}")
        
        # 保存详细结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n详细结果已保存到: {output_file}")
        
        print("\n测试完成!")
    
    def cleanup(self):
        """清理资源"""
        if BACKEND_AVAILABLE:
            riscv_ai_backend.cleanup()


def main():
    parser = argparse.ArgumentParser(description="RISC-V AI加速器芯片PyTorch测试程序")
    parser.add_argument("--output", "-o", type=str, default="test_results.json",
                       help="测试结果输出文件")
    parser.add_argument("--no-profiling", action="store_true",
                       help="禁用性能分析")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式")
    
    args = parser.parse_args()
    
    # 创建测试套件
    test_suite = ChipTestSuite(enable_profiling=not args.no_profiling)
    
    try:
        # 运行测试
        results = test_suite.run_comprehensive_test()
        
        # 生成报告
        test_suite.generate_report(results, args.output)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        test_suite.cleanup()


if __name__ == "__main__":
    main()