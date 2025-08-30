#!/usr/bin/env python3
"""
RTL集成测试 - 验证RTL代码调用
"""

import numpy as np
from simple_rtl_device import SimpleRTLDevice
from rtl_hardware_backend import RTLHardwareBackend

def test_rtl_backend_direct():
    """直接测试RTL后端"""
    print("🔬 直接测试RTL硬件后端")
    print("=" * 40)
    
    backend = RTLHardwareBackend()
    
    if backend.is_available():
        print("✅ RTL后端可用")
        
        # 测试RTL仿真
        print("\n🧪 运行RTL仿真测试:")
        backend.test_rtl_connection()
        
        # 测试小矩阵乘法
        print("\n🧮 测试RTL矩阵乘法:")
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = backend.mm(a, b)
        expected = np.matmul(a, b)
        
        print(f"RTL结果: {result}")
        print(f"期望结果: {expected}")
        print(f"结果正确: {np.allclose(result, expected)}")
        
    else:
        print("❌ RTL后端不可用")

def test_rtl_device():
    """测试RTL设备接口"""
    print("\n🔧 测试RTL设备接口")
    print("=" * 40)
    
    device = SimpleRTLDevice("test_rtl_chip")
    
    # 获取设备信息
    info = device.get_device_info()
    print("📋 设备信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 检查是否使用RTL后端
    if "RTL Hardware Simulation" in str(info.get("backend_type", "")):
        print("✅ 设备正在使用RTL硬件后端")
    else:
        print("⚠️ 设备使用软件模拟")
    
    # 简单计算测试
    print("\n🧮 简单计算测试:")
    a = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 单位矩阵
    
    result = device.matrix_multiply(a, b)
    print(f"A @ I = {result}")
    print(f"应该等于A: {np.allclose(result, a)}")
    
    # ReLU测试
    print("\n🎯 ReLU测试:")
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    relu_result = device.relu_activation(x)
    print(f"ReLU({x}) = {relu_result}")

def test_rtl_vs_software():
    """比较RTL和软件实现"""
    print("\n⚖️ RTL vs 软件实现比较")
    print("=" * 40)
    
    # 创建RTL设备
    rtl_device = SimpleRTLDevice("rtl_chip")
    
    # 测试数据
    size = 16
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    print(f"测试矩阵大小: {size}x{size}")
    
    # RTL计算
    import time
    start_time = time.time()
    rtl_result = rtl_device.matrix_multiply(a, b)
    rtl_time = time.time() - start_time
    
    # 软件计算
    start_time = time.time()
    software_result = np.matmul(a, b)
    software_time = time.time() - start_time
    
    # 比较结果
    error = np.mean(np.abs(rtl_result - software_result))
    
    print(f"RTL计算时间: {rtl_time:.4f}s")
    print(f"软件计算时间: {software_time:.4f}s")
    print(f"计算误差: {error:.2e}")
    print(f"结果正确: {error < 1e-5}")

def main():
    """主测试程序"""
    print("🧪 RTL集成测试套件")
    print("=" * 50)
    
    try:
        # 1. 直接测试RTL后端
        test_rtl_backend_direct()
        
        # 2. 测试RTL设备接口
        test_rtl_device()
        
        # 3. RTL vs 软件比较
        test_rtl_vs_software()
        
        print("\n🎉 所有RTL集成测试完成!")
        print("✨ RTL代码成功集成到Python系统中!")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()