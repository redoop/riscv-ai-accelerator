"""
简化的RISC-V AI运行时模块 (macOS仿真版)
"""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径以导入 riscv_ai_backend
sys.path.insert(0, str(Path(__file__).parent))

try:
    import riscv_ai_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

class RiscvAiRuntime:
    """RISC-V AI运行时"""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.models = {}
        self.performance_stats = {}
        
        if BACKEND_AVAILABLE:
            self.device_info = riscv_ai_backend.get_device_info()
        else:
            self.device_info = {"backend_available": False}
    
    def get_device_info(self) -> Dict:
        """获取设备信息"""
        return self.device_info
    
    def load_model_from_torch(self, model: nn.Module, model_id: str, 
                            optimize: bool = True, sample_input: Optional[torch.Tensor] = None) -> str:
        """从PyTorch模型加载"""
        print(f"📥 加载模型: {model_id}")
        
        # 在仿真模式下，我们只是保存模型引用
        self.models[model_id] = {
            "model": model,
            "optimized": optimize,
            "sample_input": sample_input,
            "loaded_at": time.time()
        }
        
        if optimize:
            print(f"⚡ 模型优化已启用 (仿真模式)")
        
        return model_id
    
    def load_model(self, model_path: str, model_id: str, 
                  optimize: bool = True, sample_input: Optional[torch.Tensor] = None) -> str:
        """从文件加载模型"""
        print(f"📂 从文件加载模型: {model_path}")
        
        # 加载PyTorch模型
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        return self.load_model_from_torch(model, model_id, optimize, sample_input)
    
    def infer(self, model_id: str, input_data: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        if model_id not in self.models:
            raise ValueError(f"模型未找到: {model_id}")
        
        model_info = self.models[model_id]
        model = model_info["model"]
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        with torch.no_grad():
            if BACKEND_AVAILABLE and model_info["optimized"]:
                # 在仿真模式下，我们仍然使用原始PyTorch模型
                # 但会记录为"加速"执行
                output = model(input_data)
                # 添加一些延迟来模拟加速器通信开销
                time.sleep(0.001)  # 1ms延迟
            else:
                output = model(input_data)
        
        # 记录性能统计
        end_time = time.time()
        exec_time = end_time - start_time
        
        if self.enable_profiling:
            if model_id not in self.performance_stats:
                self.performance_stats[model_id] = {
                    "total_inferences": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0
                }
            
            stats = self.performance_stats[model_id]
            stats["total_inferences"] += 1
            stats["total_time"] += exec_time
            stats["min_time"] = min(stats["min_time"], exec_time)
            stats["max_time"] = max(stats["max_time"], exec_time)
        
        return output
    
    def benchmark_model(self, model_id: str, input_shape: Tuple, 
                       num_iterations: int = 100, warmup_iterations: int = 10) -> Dict:
        """基准测试模型"""
        print(f"🏁 基准测试模型: {model_id} ({num_iterations}次迭代)")
        
        if model_id not in self.models:
            raise ValueError(f"模型未找到: {model_id}")
        
        # 创建测试输入
        test_input = torch.randn(*input_shape)
        
        # 预热
        print(f"🔥 预热 ({warmup_iterations}次)...")
        for _ in range(warmup_iterations):
            _ = self.infer(model_id, test_input)
        
        # 基准测试
        print(f"⏱️  基准测试 ({num_iterations}次)...")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.infer(model_id, test_input)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % (num_iterations // 10) == 0:
                print(f"  进度: {i + 1}/{num_iterations}")
        
        # 计算统计信息
        import statistics
        
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0.0
        
        results = {
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": throughput,
            "total_iterations": num_iterations,
            "input_shape": input_shape
        }
        
        print(f"📊 基准测试完成:")
        print(f"  平均时间: {mean_time:.6f}s")
        print(f"  标准差: {std_time:.6f}s")
        print(f"  吞吐量: {throughput:.2f} inferences/sec")
        
        return results
    
    def get_performance_stats(self, model_id: str) -> Dict:
        """获取性能统计"""
        if model_id not in self.performance_stats:
            return {}
        
        stats = self.performance_stats[model_id].copy()
        
        if stats["total_inferences"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_inferences"]
            stats["throughput"] = stats["total_inferences"] / stats["total_time"]
        else:
            stats["average_time"] = 0.0
            stats["throughput"] = 0.0
        
        return stats
    
    def list_models(self) -> List[str]:
        """列出已加载的模型"""
        return list(self.models.keys())
    
    def get_model_info(self, model_id: str) -> Dict:
        """获取模型信息"""
        if model_id not in self.models:
            return {}
        
        model_info = self.models[model_id]
        model = model_info["model"]
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_id": model_id,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "optimized": model_info["optimized"],
            "loaded_at": model_info["loaded_at"]
        }

def create_runtime(enable_profiling: bool = True) -> RiscvAiRuntime:
    """创建运行时实例"""
    return RiscvAiRuntime(enable_profiling)
