#!/usr/bin/env python3
"""
Integration tests for PyTorch RISC-V AI backend
Tests model loading, optimization, quantization, and inference
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import riscv_ai_backend
    from model_optimizer import RiscvAiOptimizer, RiscvAiQuantizer
    from runtime import RiscvAiRuntime, create_runtime
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    print("Warning: RISC-V AI backend not available, running fallback tests")


class SimpleConvNet(nn.Module):
    """Simple CNN for testing"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestRiscvAiBackend(unittest.TestCase):
    """Test RISC-V AI backend functionality"""
    
    def setUp(self):
        self.device_available = BACKEND_AVAILABLE
        if self.device_available:
            riscv_ai_backend.initialize()
    
    def tearDown(self):
        if self.device_available:
            riscv_ai_backend.cleanup()
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_backend_initialization(self):
        """Test backend initialization and cleanup"""
        self.assertTrue(riscv_ai_backend.is_available())
        self.assertGreaterEqual(riscv_ai_backend.device_count(), 1)
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_matrix_multiplication(self):
        """Test matrix multiplication operation"""
        a = torch.randn(64, 128, dtype=torch.float32)
        b = torch.randn(128, 256, dtype=torch.float32)
        
        # Test with RISC-V AI backend
        result_ai = riscv_ai_backend.mm(a, b)
        
        # Test with standard PyTorch
        result_torch = torch.mm(a, b)
        
        # Compare results (allow small numerical differences)
        self.assertTrue(torch.allclose(result_ai, result_torch, rtol=1e-4, atol=1e-5))
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_convolution(self):
        """Test 2D convolution operation"""
        input_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        weight = torch.randn(64, 3, 3, 3, dtype=torch.float32)
        bias = torch.randn(64, dtype=torch.float32)
        
        # Test with RISC-V AI backend
        result_ai = riscv_ai_backend.conv2d(input_tensor, weight, bias, 
                                           stride=[1, 1], padding=[1, 1],
                                           dilation=[1, 1], groups=1)
        
        # Test with standard PyTorch
        result_torch = F.conv2d(input_tensor, weight, bias, 
                               stride=1, padding=1, dilation=1, groups=1)
        
        # Compare results
        self.assertTrue(torch.allclose(result_ai, result_torch, rtol=1e-3, atol=1e-4))
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_activation_functions(self):
        """Test activation functions"""
        input_tensor = torch.randn(100, dtype=torch.float32)
        
        # Test ReLU
        result_ai_relu = riscv_ai_backend.relu(input_tensor)
        result_torch_relu = F.relu(input_tensor)
        self.assertTrue(torch.allclose(result_ai_relu, result_torch_relu))
        
        # Test Sigmoid
        result_ai_sigmoid = riscv_ai_backend.sigmoid(input_tensor)
        result_torch_sigmoid = torch.sigmoid(input_tensor)
        self.assertTrue(torch.allclose(result_ai_sigmoid, result_torch_sigmoid, rtol=1e-4))
        
        # Test Tanh
        result_ai_tanh = riscv_ai_backend.tanh(input_tensor)
        result_torch_tanh = torch.tanh(input_tensor)
        self.assertTrue(torch.allclose(result_ai_tanh, result_torch_tanh, rtol=1e-4))
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_pooling_operations(self):
        """Test pooling operations"""
        input_tensor = torch.randn(1, 32, 16, 16, dtype=torch.float32)
        
        # Test MaxPool2D
        result_ai_max = riscv_ai_backend.max_pool2d(input_tensor, 
                                                   kernel_size=[2, 2],
                                                   stride=[2, 2], 
                                                   padding=[0, 0])
        result_torch_max = F.max_pool2d(input_tensor, kernel_size=2, stride=2)
        self.assertTrue(torch.allclose(result_ai_max, result_torch_max))
        
        # Test AvgPool2D
        result_ai_avg = riscv_ai_backend.avg_pool2d(input_tensor,
                                                   kernel_size=[2, 2],
                                                   stride=[2, 2],
                                                   padding=[0, 0])
        result_torch_avg = F.avg_pool2d(input_tensor, kernel_size=2, stride=2)
        self.assertTrue(torch.allclose(result_ai_avg, result_torch_avg, rtol=1e-4))


class TestModelOptimizer(unittest.TestCase):
    """Test model optimization functionality"""
    
    def setUp(self):
        self.model = SimpleConvNet()
        self.sample_input = torch.randn(1, 3, 32, 32)
        if BACKEND_AVAILABLE:
            self.optimizer = RiscvAiOptimizer()
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_model_optimization(self):
        """Test model optimization"""
        optimized_model = self.optimizer.optimize_model(self.model, self.sample_input, "O2")
        
        # Test that optimized model produces similar results
        self.model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            original_output = self.model(self.sample_input)
            optimized_output = optimized_model(self.sample_input)
            
        # Results should be close
        self.assertTrue(torch.allclose(original_output, optimized_output, rtol=1e-3, atol=1e-4))
    
    def test_operation_fusion(self):
        """Test operation fusion"""
        if not BACKEND_AVAILABLE:
            self.skipTest("RISC-V AI backend not available")
            
        # Create a model with fusable operations
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        optimized_model = self.optimizer.optimize_model(model, self.sample_input, "O2")
        
        # Check that optimization was applied
        self.assertIsNotNone(optimized_model)


class TestQuantization(unittest.TestCase):
    """Test model quantization functionality"""
    
    def setUp(self):
        self.model = SimpleConvNet()
        if BACKEND_AVAILABLE:
            self.quantizer = RiscvAiQuantizer()
    
    @unittest.skipUnless(BACKEND_AVAILABLE, "RISC-V AI backend not available")
    def test_int8_quantization(self):
        """Test INT8 quantization"""
        # Create dummy calibration data
        calibration_data = [(torch.randn(1, 3, 32, 32), torch.tensor([0])) for _ in range(10)]
        calibration_loader = torch.utils.data.DataLoader(calibration_data, batch_size=1)
        
        quantized_model = self.quantizer.quantize_model(self.model, calibration_loader, "int8")
        
        # Test that quantized model can run inference
        sample_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = quantized_model(sample_input)
            
        self.assertEqual(output.shape, (1, 10))


class TestRuntime(unittest.TestCase):
    """Test runtime functionality"""
    
    def setUp(self):
        self.runtime = create_runtime(enable_profiling=True)
        self.model = SimpleConvNet()
        self.sample_input = torch.randn(1, 3, 32, 32)
        
        # Create temporary model file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(self.model, self.model_path)
    
    def tearDown(self):
        # Cleanup temporary files
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        os.rmdir(self.temp_dir)
    
    def test_model_loading(self):
        """Test model loading and management"""
        model_id = self.runtime.load_model(self.model_path, "test_model", 
                                          optimize=False, sample_input=self.sample_input)
        
        self.assertEqual(model_id, "test_model")
        self.assertIn("test_model", self.runtime.list_models())
        
        # Test model info
        info = self.runtime.get_model_info("test_model")
        self.assertIn("total_parameters", info)
        self.assertGreater(info["total_parameters"], 0)
    
    def test_inference(self):
        """Test model inference"""
        self.runtime.load_model(self.model_path, "test_model", 
                               optimize=False, sample_input=self.sample_input)
        
        output = self.runtime.infer("test_model", self.sample_input)
        self.assertEqual(output.shape, (1, 10))
    
    def test_benchmarking(self):
        """Test model benchmarking"""
        self.runtime.load_model(self.model_path, "test_model", 
                               optimize=False, sample_input=self.sample_input)
        
        stats = self.runtime.benchmark_model("test_model", (1, 3, 32, 32), 
                                            num_iterations=10, warmup_iterations=2)
        
        self.assertIn("mean_time", stats)
        self.assertIn("throughput", stats)
        self.assertGreater(stats["throughput"], 0)
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        self.runtime.load_model(self.model_path, "test_model", 
                               optimize=False, sample_input=self.sample_input)
        
        # Run some inferences
        for _ in range(5):
            self.runtime.infer("test_model", self.sample_input)
        
        stats = self.runtime.get_performance_stats("test_model")
        self.assertGreater(stats["total_inferences"], 0)
        self.assertGreater(stats["total_time"], 0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_complete_workflow(self):
        """Test complete model optimization and inference workflow"""
        # Create model
        model = SimpleConvNet()
        sample_input = torch.randn(1, 3, 32, 32)
        
        # Create runtime
        runtime = create_runtime(enable_profiling=True)
        
        # Save model temporarily
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            model_path = f.name
        
        try:
            # Load and optimize model
            model_id = runtime.load_model(model_path, "test_model", 
                                        optimize=True, sample_input=sample_input)
            
            # Run inference
            output = runtime.infer(model_id, sample_input)
            self.assertEqual(output.shape, (1, 10))
            
            # Benchmark model
            stats = runtime.benchmark_model(model_id, (1, 3, 32, 32), 
                                          num_iterations=5, warmup_iterations=1)
            self.assertGreater(stats["throughput"], 0)
            
            # Check device info
            device_info = runtime.get_device_info()
            self.assertIn("backend_available", device_info)
            
        finally:
            # Cleanup
            os.unlink(model_path)


def run_performance_tests():
    """Run performance comparison tests"""
    print("Running performance comparison tests...")
    
    # Test matrix multiplication performance
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for m, n in sizes:
        a = torch.randn(m, n, dtype=torch.float32)
        b = torch.randn(n, m, dtype=torch.float32)
        
        # Time PyTorch implementation
        start_time = time.time()
        for _ in range(100):
            result_torch = torch.mm(a, b)
        torch_time = time.time() - start_time
        
        # Time RISC-V AI implementation if available
        if BACKEND_AVAILABLE:
            start_time = time.time()
            for _ in range(100):
                result_ai = riscv_ai_backend.mm(a, b)
            ai_time = time.time() - start_time
            
            speedup = torch_time / ai_time
            print(f"Matrix {m}x{n}: PyTorch={torch_time:.4f}s, RISC-V AI={ai_time:.4f}s, Speedup={speedup:.2f}x")
        else:
            print(f"Matrix {m}x{n}: PyTorch={torch_time:.4f}s, RISC-V AI=N/A")


if __name__ == "__main__":
    import time
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()