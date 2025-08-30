"""
PyTorch Runtime Support for RISC-V AI Accelerator
Provides model loading, execution, and performance monitoring
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import time
import logging
import json
import os
from pathlib import Path

try:
    import riscv_ai_backend
    from .model_optimizer import RiscvAiOptimizer, RiscvAiQuantizer
except ImportError:
    riscv_ai_backend = None
    logging.warning("RISC-V AI backend not available")

class RiscvAiRuntime:
    """Runtime environment for RISC-V AI accelerator"""
    
    def __init__(self, device_id: int = 0, enable_profiling: bool = False):
        self.device_id = device_id
        self.enable_profiling = enable_profiling
        self.models = {}
        self.performance_stats = {}
        self.optimizer = RiscvAiOptimizer(device_id) if riscv_ai_backend else None
        self.quantizer = RiscvAiQuantizer() if riscv_ai_backend else None
        
        if riscv_ai_backend:
            riscv_ai_backend.initialize()
            
    def __del__(self):
        if riscv_ai_backend:
            riscv_ai_backend.cleanup()
    
    def load_model(self, model_path: str, model_name: str = None,
                  optimize: bool = True, quantize: bool = False,
                  sample_input: torch.Tensor = None) -> str:
        """
        Load and prepare model for inference
        
        Args:
            model_path: Path to model file (.pt, .pth, .onnx)
            model_name: Optional name for the model
            optimize: Whether to optimize the model
            quantize: Whether to quantize the model
            sample_input: Sample input for optimization
            
        Returns:
            Model identifier
        """
        if model_name is None:
            model_name = Path(model_path).stem
            
        logging.info(f"Loading model: {model_name} from {model_path}")
        
        # Load model based on file extension
        if model_path.endswith('.onnx'):
            model = self._load_onnx_model(model_path)
        else:
            model = torch.load(model_path, map_location='cpu')
            
        # Optimize model if requested
        if optimize and self.optimizer and sample_input is not None:
            model = self.optimizer.optimize_model(model, sample_input)
            
        # Quantize model if requested
        if quantize and self.quantizer:
            # For quantization, we need calibration data
            # This is a simplified version
            model = self.quantizer.quantize_model(model, None)
            
        # Store model
        self.models[model_name] = {
            'model': model,
            'optimized': optimize,
            'quantized': quantize,
            'load_time': time.time()
        }
        
        logging.info(f"Model {model_name} loaded successfully")
        return model_name
    
    def _load_onnx_model(self, model_path: str) -> nn.Module:
        """Load ONNX model and convert to PyTorch"""
        try:
            import onnx
            import onnx2torch
            
            onnx_model = onnx.load(model_path)
            pytorch_model = onnx2torch.convert(onnx_model)
            return pytorch_model
        except ImportError:
            raise ImportError("ONNX support requires 'onnx' and 'onnx2torch' packages")
    
    def infer(self, model_name: str, input_data: Union[torch.Tensor, np.ndarray],
             batch_size: int = 1) -> torch.Tensor:
        """
        Run inference on loaded model
        
        Args:
            model_name: Name of the loaded model
            input_data: Input data for inference
            batch_size: Batch size for inference
            
        Returns:
            Model output
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Convert input data to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data)
        else:
            input_tensor = input_data
            
        # Ensure correct batch size
        if input_tensor.dim() > 0 and input_tensor.size(0) != batch_size:
            if batch_size == 1:
                input_tensor = input_tensor.unsqueeze(0)
            else:
                # Repeat or slice to match batch size
                current_batch = input_tensor.size(0)
                if current_batch < batch_size:
                    repeats = (batch_size + current_batch - 1) // current_batch
                    input_tensor = input_tensor.repeat(repeats, *([1] * (input_tensor.dim() - 1)))
                input_tensor = input_tensor[:batch_size]
        
        # Run inference
        start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            if riscv_ai_backend and riscv_ai_backend.is_available():
                # Move to RISC-V AI device if available
                # For now, run on CPU with AI acceleration
                output = model(input_tensor)
            else:
                output = model(input_tensor)
                
        inference_time = time.time() - start_time
        
        # Update performance statistics
        if self.enable_profiling:
            self._update_performance_stats(model_name, inference_time, batch_size)
            
        return output
    
    def benchmark_model(self, model_name: str, input_shape: Tuple[int, ...],
                       num_iterations: int = 100, warmup_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            model_name: Name of the loaded model
            input_shape: Shape of input tensor
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Performance statistics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        logging.info(f"Benchmarking model {model_name}")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        for _ in range(warmup_iterations):
            self.infer(model_name, dummy_input)
            
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            self.infer(model_name, dummy_input)
            times.append(time.time() - start_time)
            
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'throughput': 1.0 / np.mean(times),  # inferences per second
            'total_iterations': num_iterations
        }
        
        logging.info(f"Benchmark results for {model_name}: {stats}")
        return stats
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'name': model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'optimized': model_info['optimized'],
            'quantized': model_info['quantized'],
            'load_time': model_info['load_time']
        }
        
        # Add model structure info if available
        try:
            info['model_structure'] = str(model)
        except:
            pass
            
        return info
    
    def list_models(self) -> List[str]:
        """List all loaded models"""
        return list(self.models.keys())
    
    def unload_model(self, model_name: str):
        """Unload model from memory"""
        if model_name in self.models:
            del self.models[model_name]
            logging.info(f"Model {model_name} unloaded")
        else:
            logging.warning(f"Model {model_name} not found")
    
    def get_performance_stats(self, model_name: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if model_name:
            return self.performance_stats.get(model_name, {})
        else:
            return self.performance_stats
    
    def save_performance_stats(self, filepath: str):
        """Save performance statistics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.performance_stats, f, indent=2)
    
    def _update_performance_stats(self, model_name: str, inference_time: float, batch_size: int):
        """Update performance statistics"""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                'total_inferences': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'times': []
            }
            
        stats = self.performance_stats[model_name]
        stats['total_inferences'] += batch_size
        stats['total_time'] += inference_time
        stats['min_time'] = min(stats['min_time'], inference_time)
        stats['max_time'] = max(stats['max_time'], inference_time)
        stats['times'].append(inference_time)
        
        # Keep only recent times (last 1000)
        if len(stats['times']) > 1000:
            stats['times'] = stats['times'][-1000:]
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get RISC-V AI device information"""
        info = {
            'device_id': self.device_id,
            'backend_available': riscv_ai_backend is not None
        }
        
        if riscv_ai_backend:
            info.update({
                'is_available': riscv_ai_backend.is_available(),
                'device_count': riscv_ai_backend.device_count()
            })
            
        return info


class ModelFormat:
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


def create_runtime(device_id: int = 0, enable_profiling: bool = False) -> RiscvAiRuntime:
    """Create RISC-V AI runtime instance"""
    return RiscvAiRuntime(device_id, enable_profiling)


def load_and_optimize_model(model_path: str, sample_input: torch.Tensor,
                           optimization_level: str = "O2") -> nn.Module:
    """Convenience function to load and optimize a model"""
    runtime = create_runtime()
    optimizer = RiscvAiOptimizer()
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model, sample_input, optimization_level)
    
    return optimized_model