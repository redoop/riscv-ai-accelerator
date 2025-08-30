#!/usr/bin/env python3
"""
macOS RISC-V AI仿真器安装脚本
将仿真器集成到现有的PyTorch测试框架中
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import tempfile

def create_riscv_ai_backend_module():
    """创建riscv_ai_backend模块"""
    print("📦 创建riscv_ai_backend模块...")
    
    # 创建模块目录
    module_dir = Path("scripts/riscv_ai_backend")
    module_dir.mkdir(exist_ok=True)
    
    # 创建__init__.py文件
    init_content = '''"""
RISC-V AI Backend for macOS
提供RISC-V AI加速器的macOS仿真支持
"""

# 导入仿真后端
from .riscv_ai_backend_macos import *

# 版本信息
__version__ = "1.0.0-macos-simulator"
__author__ = "RISC-V AI Team"
__description__ = "RISC-V AI Accelerator macOS Simulator"

# 自动初始化标志
_auto_initialize = True

def set_auto_initialize(enabled: bool):
    """设置是否自动初始化"""
    global _auto_initialize
    _auto_initialize = enabled

# 尝试自动初始化
if _auto_initialize:
    try:
        initialize()
    except Exception as e:
        import warnings
        warnings.warn(f"自动初始化失败: {e}")
'''
    
    with open(module_dir / "__init__.py", "w") as f:
        f.write(init_content)
    
    # 复制仿真器文件
    shutil.copy2("scripts/riscv_ai_backend_macos.py", module_dir / "riscv_ai_backend_macos.py")
    shutil.copy2("scripts/macos_ai_simulator.py", module_dir / "macos_ai_simulator.py")
    
    print(f"✅ 模块创建完成: {module_dir}")
    return module_dir

def create_runtime_module():
    """创建简化的runtime模块"""
    print("🏃 创建runtime模块...")
    
    runtime_content = '''"""
简化的RISC-V AI运行时模块 (macOS仿真版)
"""

import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
import os

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
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
'''
    
    with open("runtime.py", "w") as f:
        f.write(runtime_content)
    
    print("✅ runtime模块创建完成")

def create_model_optimizer():
    """创建模型优化器模块"""
    print("⚡ 创建model_optimizer模块...")
    
    optimizer_content = '''"""
RISC-V AI模型优化器 (macOS仿真版)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

class RiscvAiOptimizer:
    """RISC-V AI模型优化器"""
    
    def __init__(self):
        self.optimization_levels = {
            "O0": "无优化",
            "O1": "基础优化", 
            "O2": "标准优化",
            "O3": "激进优化"
        }
    
    def optimize_model(self, model: nn.Module, sample_input: torch.Tensor, 
                      optimization_level: str = "O2") -> nn.Module:
        """优化模型"""
        print(f"⚡ 优化模型 (级别: {optimization_level})")
        
        if optimization_level not in self.optimization_levels:
            raise ValueError(f"不支持的优化级别: {optimization_level}")
        
        # 在仿真模式下，我们返回原始模型
        # 实际实现中会进行操作融合、内存优化等
        optimized_model = model
        
        print(f"✅ 模型优化完成: {self.optimization_levels[optimization_level]}")
        
        return optimized_model

class RiscvAiQuantizer:
    """RISC-V AI量化器"""
    
    def __init__(self):
        self.supported_schemes = ["int8", "int16", "fp16"]
    
    def quantize_model(self, model: nn.Module, calibration_loader, 
                      quantization_scheme: str = "int8") -> nn.Module:
        """量化模型"""
        print(f"🔢 量化模型 (方案: {quantization_scheme})")
        
        if quantization_scheme not in self.supported_schemes:
            raise ValueError(f"不支持的量化方案: {quantization_scheme}")
        
        # 在仿真模式下，我们返回原始模型
        # 实际实现中会进行权重量化
        quantized_model = model
        
        print(f"✅ 模型量化完成: {quantization_scheme}")
        
        return quantized_model
'''
    
    with open("model_optimizer.py", "w") as f:
        f.write(optimizer_content)
    
    print("✅ model_optimizer模块创建完成")

def update_pytorch_test():
    """更新PyTorch测试程序以支持仿真器"""
    print("🔄 更新PyTorch测试程序...")
    
    # 读取现有的测试程序
    test_file = "pytorch_chip_test.py"
    if not os.path.exists(test_file):
        print(f"⚠️  测试文件不存在: {test_file}")
        return
    
    with open(test_file, 'r') as f:
        content = f.read()
    
    # 添加仿真器支持的导入语句
    import_addition = '''
# macOS仿真器支持
import platform
if platform.system() == "Darwin":
    print("🍎 检测到macOS系统，启用仿真模式")
    try:
        # 尝试导入仿真器
        import riscv_ai_backend
        print("✅ RISC-V AI仿真器已加载")
    except ImportError:
        print("⚠️  仿真器未安装，请运行: python3 install_macos_simulator.py")
'''
    
    # 在导入部分后添加仿真器支持
    if "# macOS仿真器支持" not in content:
        # 找到导入部分的结束位置
        import_end = content.find('BACKEND_AVAILABLE = True')
        if import_end != -1:
            insert_pos = content.find('\n', import_end) + 1
            content = content[:insert_pos] + import_addition + content[insert_pos:]
            
            # 写回文件
            with open(test_file, 'w') as f:
                f.write(content)
            
            print("✅ 测试程序已更新")
        else:
            print("⚠️  无法找到合适的插入位置")
    else:
        print("✅ 测试程序已包含仿真器支持")

def create_macos_specific_makefile():
    """创建macOS专用的Makefile目标"""
    print("📝 更新Makefile...")
    
    makefile_addition = '''
# macOS仿真器支持
.PHONY: install-simulator
install-simulator:
	@echo "🍎 安装macOS RISC-V AI仿真器..."
	$(PYTHON) install_macos_simulator.py
	@echo "✅ 仿真器安装完成"

.PHONY: test-simulator
test-simulator: install-simulator
	@echo "🧪 测试仿真器功能..."
	$(PYTHON) -c "import riscv_ai_backend; print('仿真器版本:', riscv_ai_backend.__version__)"
	$(PYTHON) riscv_ai_backend_macos.py
	@echo "✅ 仿真器测试完成"

.PHONY: test-macos
test-macos: install-simulator test-simple
	@echo "🍎 运行macOS完整测试..."
	@if [ "$(shell uname -s)" = "Darwin" ]; then \
		echo "在macOS上运行仿真测试..."; \
		$(PYTHON) $(COMPREHENSIVE_TEST) --output $(OUTPUT_DIR)/macos_results.json \
			2>&1 | tee $(LOGS_DIR)/macos_test.log; \
	else \
		echo "⚠️  此目标仅适用于macOS系统"; \
	fi

.PHONY: demo-simulator
demo-simulator: install-simulator
	@echo "🎬 运行仿真器演示..."
	$(PYTHON) -c "
import torch; \
import riscv_ai_backend as ai; \
print('🚀 RISC-V AI仿真器演示'); \
print('设备信息:', ai.get_device_info()); \
a = torch.randn(64, 64); \
b = torch.randn(64, 64); \
c = ai.mm(a, b); \
print('矩阵乘法完成:', c.shape); \
print('性能统计:', ai.get_performance_stats()); \
"
'''
    
    makefile_path = "Makefile.pytorch_test"
    
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    if "# macOS仿真器支持" not in content:
        content += makefile_addition
        
        with open(makefile_path, 'w') as f:
            f.write(content)
        
        print("✅ Makefile已更新")
    else:
        print("✅ Makefile已包含仿真器支持")

def main():
    """主安装函数"""
    print("🍎 RISC-V AI加速器macOS仿真器安装程序")
    print("=" * 50)
    
    try:
        # 检查系统
        if os.uname().sysname != "Darwin":
            print("⚠️  此安装程序专为macOS设计")
            print("   在其他系统上可能无法正常工作")
        
        # 检查Python依赖
        print("🔍 检查依赖...")
        try:
            import torch
            import numpy
            print("✅ PyTorch和NumPy已安装")
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("请运行: pip install torch numpy")
            return False
        
        # 创建模块
        module_dir = create_riscv_ai_backend_module()
        create_runtime_module()
        create_model_optimizer()
        
        # 更新测试程序
        update_pytorch_test()
        
        # 更新Makefile
        create_macos_specific_makefile()
        
        print("\n🎉 安装完成!")
        print("\n📋 使用方法:")
        print("1. 测试仿真器:")
        print("   make -f Makefile.pytorch_test test-simulator")
        print("\n2. 运行演示:")
        print("   make -f Makefile.pytorch_test demo-simulator")
        print("\n3. 运行完整测试:")
        print("   make -f Makefile.pytorch_test test-macos")
        print("\n4. 直接使用:")
        print("   python3 -c 'import sys; sys.path.insert(0, \"scripts\"); import riscv_ai_backend; print(riscv_ai_backend.get_device_info())'")
        
        return True
        
    except Exception as e:
        print(f"❌ 安装失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)