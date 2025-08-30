#!/usr/bin/env python3
"""
RTL设备演示 - 将RTL代码作为设备在macOS上运行
基于我们之前创建的RTL仿真后端
"""

import ctypes
import numpy as np
import time
import threading
import os
from ctypes import POINTER, c_float, c_uint32, c_void_p, c_bool

class RTLDeviceSimulator:
    """RTL设备模拟器 - 将RTL仿真包装为设备接口"""
    
    def __init__(self):
        # 检查RTL库是否存在
        lib_path = './librtl_simulator.so'
        if not os.path.exists(lib_path):
            print("❌ RTL库不存在，请先运行 python3 create_rtl_backend.py")
            raise FileNotFoundError(f"RTL库不存在: {lib_path}")
        
        # 加载RTL仿真库
        self.lib = ctypes.CDLL(lib_path)
        
        # 定义函数签名
        self.lib.create_rtl_simulator.restype = c_void_p
        self.lib.destroy_rtl_simulator.argtypes = [c_void_p]
        self.lib.rtl_matrix_multiply.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        
        # 创建RTL仿真器实例
        self.simulator = self.lib.create_rtl_simulator()
        if not self.simulator:
            raise RuntimeError("无法创建RTL仿真器")
        
        # 设备状态
        self.device_id = "RTL_CHIP_0"
        self.is_active = True
        self.operation_count = 0
        self.total_compute_time = 0.0
        
        print(f"🔧 RTL设备已初始化: {self.device_id}")
    
    def __del__(self):
        if hasattr(self, 'simulator') and self.simulator:
            self.lib.destroy_rtl_simulator(self.simulator)
    
    def get_device_info(self):
        """获取设备信息"""
        return {
            "device_id": self.device_id,
            "device_type": "RISC-V AI Chip (RTL Simulation)",
            "vendor": "AI Chip Design Team",
            "version": "1.0.0",
            "status": "active" if self.is_active else "inactive",
            "capabilities": [
                "Matrix Multiplication",
                "TPU Acceleration", 
                "Multi-precision (INT8/FP16/FP32)",
                "Hardware Simulation"
            ],
            "performance": {
                "operations_completed": self.operation_count,
                "total_compute_time": f"{self.total_compute_time:.4f}s",
                "avg_operation_time": f"{self.total_compute_time/max(1,self.operation_count):.4f}s"
            }
        }
    
    def read_register(self, address):
        """模拟寄存器读取"""
        # 模拟一些设备寄存器
        registers = {
            0x0000: 0x12345678,  # DEVICE_ID
            0x0004: 0x00000001 if self.is_active else 0x00000000,  # STATUS
            0x0008: self.operation_count,  # OP_COUNT
            0x000C: int(self.total_compute_time * 1000),  # COMPUTE_TIME_MS
            0x1000: 0x00000001,  # TPU_READY
            0x1004: 0x00000007,  # TPU_CAPS (INT8|FP16|FP32)
        }
        
        return registers.get(address, 0x00000000)
    
    def write_register(self, address, value):
        """模拟寄存器写入"""
        if address == 0x0004:  # STATUS register
            self.is_active = bool(value & 0x01)
            print(f"🔧 设备状态更新: {'active' if self.is_active else 'inactive'}")
        elif address == 0x0008:  # Reset operation count
            if value == 0:
                self.operation_count = 0
                self.total_compute_time = 0.0
                print("🔄 性能计数器已重置")
    
    def tpu_matrix_multiply(self, a, b):
        """使用TPU进行矩阵乘法"""
        if not self.is_active:
            raise RuntimeError("设备未激活")
        
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"矩阵维度不匹配: {a.shape} @ {b.shape}")
        
        print(f"🧮 TPU矩阵乘法: {a.shape} @ {b.shape}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 准备数据
        size = a.shape[0]
        result = np.zeros((size, b.shape[1]), dtype=np.float32)
        
        # 转换为C数组
        a_ptr = a.astype(np.float32).ctypes.data_as(POINTER(c_float))
        b_ptr = b.astype(np.float32).ctypes.data_as(POINTER(c_float))
        result_ptr = result.ctypes.data_as(POINTER(c_float))
        
        # 调用RTL仿真器
        self.lib.rtl_matrix_multiply(self.simulator, a_ptr, b_ptr, result_ptr, size)
        
        # 记录性能
        compute_time = time.time() - start_time
        self.operation_count += 1
        self.total_compute_time += compute_time
        
        print(f"✅ TPU计算完成，耗时: {compute_time:.4f}s")
        
        return result
    
    def benchmark_performance(self, sizes=[32, 64, 128]):
        """性能基准测试"""
        print(f"🚀 开始TPU性能基准测试...")
        results = {}
        
        for size in sizes:
            print(f"\n📊 测试 {size}x{size} 矩阵乘法:")
            
            # 生成测试数据
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # TPU测试
            start_time = time.time()
            tpu_result = self.tpu_matrix_multiply(a, b)
            tpu_time = time.time() - start_time
            
            # CPU参考测试
            start_time = time.time()
            cpu_result = np.matmul(a, b)
            cpu_time = time.time() - start_time
            
            # 计算指标
            error = np.mean(np.abs(tpu_result - cpu_result))
            gflops = (2 * size**3) / (tpu_time * 1e9)
            speedup = cpu_time / tpu_time if tpu_time > 0 else 0
            
            results[size] = {
                "tpu_time": tpu_time,
                "cpu_time": cpu_time,
                "speedup": speedup,
                "gflops": gflops,
                "error": error,
                "accuracy": "PASS" if error < 1e-3 else "FAIL"
            }
            
            print(f"  TPU时间: {tpu_time:.4f}s")
            print(f"  CPU时间: {cpu_time:.4f}s")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  性能: {gflops:.2f} GFLOPS")
            print(f"  精度: {results[size]['accuracy']} (误差: {error:.2e})")
        
        return results
    
    def stress_test(self, duration=10, matrix_size=64):
        """压力测试"""
        print(f"⚡ 开始压力测试 ({duration}秒, {matrix_size}x{matrix_size}矩阵)...")
        
        start_time = time.time()
        operations = 0
        errors = 0
        
        while time.time() - start_time < duration:
            try:
                # 生成随机矩阵
                a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
                b = np.random.randn(matrix_size, matrix_size).astype(np.float32)
                
                # 执行计算
                result = self.tpu_matrix_multiply(a, b)
                
                # 验证结果
                cpu_result = np.matmul(a, b)
                error = np.mean(np.abs(result - cpu_result))
                
                if error > 1e-3:
                    errors += 1
                
                operations += 1
                
                # 每100次操作报告一次
                if operations % 100 == 0:
                    elapsed = time.time() - start_time
                    ops_per_sec = operations / elapsed
                    print(f"  进度: {operations} 操作, {ops_per_sec:.1f} ops/s, {errors} 错误")
            
            except Exception as e:
                errors += 1
                print(f"  错误: {e}")
        
        total_time = time.time() - start_time
        ops_per_sec = operations / total_time
        error_rate = errors / operations * 100 if operations > 0 else 0
        
        print(f"\n📈 压力测试结果:")
        print(f"  总操作数: {operations}")
        print(f"  总时间: {total_time:.2f}s")
        print(f"  吞吐量: {ops_per_sec:.2f} ops/s")
        print(f"  错误率: {error_rate:.2f}%")
        print(f"  状态: {'PASS' if error_rate < 1.0 else 'FAIL'}")
        
        return {
            "operations": operations,
            "total_time": total_time,
            "ops_per_sec": ops_per_sec,
            "error_rate": error_rate,
            "status": "PASS" if error_rate < 1.0 else "FAIL"
        }

class RTLDeviceManager:
    """RTL设备管理器"""
    
    def __init__(self):
        self.devices = {}
        self.device_count = 0
    
    def create_device(self, device_name=None):
        """创建RTL设备"""
        if device_name is None:
            device_name = f"rtl_chip_{self.device_count}"
        
        if device_name in self.devices:
            print(f"⚠️ 设备 {device_name} 已存在")
            return self.devices[device_name]
        
        try:
            device = RTLDeviceSimulator()
            device.device_id = device_name
            self.devices[device_name] = device
            self.device_count += 1
            
            print(f"✅ 设备创建成功: {device_name}")
            return device
        except Exception as e:
            print(f"❌ 设备创建失败: {e}")
            return None
    
    def list_devices(self):
        """列出所有设备"""
        return list(self.devices.keys())
    
    def get_device(self, device_name):
        """获取设备"""
        return self.devices.get(device_name)
    
    def remove_device(self, device_name):
        """移除设备"""
        if device_name in self.devices:
            del self.devices[device_name]
            print(f"🗑️ 设备已移除: {device_name}")
            return True
        return False
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            "total_devices": len(self.devices),
            "active_devices": sum(1 for d in self.devices.values() if d.is_active),
            "devices": {}
        }
        
        for name, device in self.devices.items():
            status["devices"][name] = device.get_device_info()
        
        return status

def main():
    """主演示程序"""
    print("🔧 RTL设备演示程序")
    print("=" * 50)
    
    try:
        # 1. 创建设备管理器
        print("\n1️⃣ 创建设备管理器...")
        manager = RTLDeviceManager()
        
        # 2. 创建RTL设备
        print("\n2️⃣ 创建RTL设备...")
        device = manager.create_device("ai_chip_demo")
        
        if not device:
            print("❌ 设备创建失败，退出程序")
            return
        
        # 3. 获取设备信息
        print("\n3️⃣ 设备信息:")
        info = device.get_device_info()
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
        
        # 4. 寄存器操作演示
        print("\n4️⃣ 寄存器操作演示:")
        device_id = device.read_register(0x0000)
        status = device.read_register(0x0004)
        print(f"  设备ID: 0x{device_id:08X}")
        print(f"  状态寄存器: 0x{status:08X}")
        
        # 5. 简单矩阵乘法测试
        print("\n5️⃣ 简单矩阵乘法测试:")
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        
        print(f"  矩阵A: {a.tolist()}")
        print(f"  矩阵B: {b.tolist()}")
        
        result = device.tpu_matrix_multiply(a, b)
        cpu_result = np.matmul(a, b)
        
        print(f"  TPU结果: {result.tolist()}")
        print(f"  CPU参考: {cpu_result.tolist()}")
        print(f"  误差: {np.mean(np.abs(result - cpu_result)):.2e}")
        
        # 6. 性能基准测试
        print("\n6️⃣ 性能基准测试:")
        benchmark_results = device.benchmark_performance([32, 64])
        
        print("\n📊 基准测试总结:")
        for size, result in benchmark_results.items():
            print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS, "
                  f"{result['speedup']:.2f}x speedup, {result['accuracy']}")
        
        # 7. 压力测试 (可选)
        print("\n7️⃣ 压力测试 (5秒):")
        stress_result = device.stress_test(duration=5, matrix_size=32)
        
        # 8. 系统状态
        print("\n8️⃣ 系统状态:")
        system_status = manager.get_system_status()
        print(f"  总设备数: {system_status['total_devices']}")
        print(f"  活跃设备数: {system_status['active_devices']}")
        
        print("\n🎉 RTL设备演示完成!")
        print("✨ RTL代码已成功作为设备在macOS上运行!")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()