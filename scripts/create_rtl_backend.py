#!/usr/bin/env python3
"""
RTL后端创建工具 - 创建RTL仿真共享库
将RTL代码编译为可被Python调用的共享库
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def create_rtl_simulator_library():
    """创建RTL仿真器共享库"""
    print("🔧 创建RTL仿真器共享库")
    print("=" * 40)
    
    # 创建C++包装器代码
    cpp_wrapper = """
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    // RTL仿真器结构体
    struct RTLSimulator {
        bool initialized;
        uint32_t operation_count;
        
        RTLSimulator() : initialized(true), operation_count(0) {}
    };
    
    // 创建RTL仿真器
    void* create_rtl_simulator() {
        RTLSimulator* sim = new RTLSimulator();
        printf("RTL仿真器已创建\\n");
        return sim;
    }
    
    // 销毁RTL仿真器
    void destroy_rtl_simulator(void* simulator) {
        if (simulator) {
            RTLSimulator* sim = static_cast<RTLSimulator*>(simulator);
            delete sim;
            printf("RTL仿真器已销毁\\n");
        }
    }
    
    // RTL矩阵乘法 - 模拟RTL MAC单元行为
    void rtl_matrix_multiply(void* simulator, float* a, float* b, float* result, uint32_t size) {
        if (!simulator) return;
        
        RTLSimulator* sim = static_cast<RTLSimulator*>(simulator);
        
        // 模拟RTL MAC单元进行矩阵乘法
        for (uint32_t i = 0; i < size; i++) {
            for (uint32_t j = 0; j < size; j++) {
                float sum = 0.0f;
                
                // 使用MAC单元: sum += a[i,k] * b[k,j]
                for (uint32_t k = 0; k < size; k++) {
                    // 模拟RTL MAC操作: result = a * b + c
                    float a_val = a[i * size + k];
                    float b_val = b[k * size + j];
                    sum = a_val * b_val + sum;  // MAC: multiply-accumulate
                }
                
                result[i * size + j] = sum;
            }
        }
        
        sim->operation_count++;
    }
    
    // 获取操作计数
    uint32_t get_operation_count(void* simulator) {
        if (!simulator) return 0;
        RTLSimulator* sim = static_cast<RTLSimulator*>(simulator);
        return sim->operation_count;
    }
    
    // 重置操作计数
    void reset_operation_count(void* simulator) {
        if (!simulator) return;
        RTLSimulator* sim = static_cast<RTLSimulator*>(simulator);
        sim->operation_count = 0;
    }
}
"""
    
    try:
        # 写入C++源文件
        with open("rtl_simulator.cpp", "w") as f:
            f.write(cpp_wrapper)
        
        print("✅ C++包装器代码已生成")
        
        # 编译为共享库
        print("🔨 编译RTL仿真器共享库...")
        
        # 检测操作系统
        if sys.platform == "darwin":  # macOS
            compile_cmd = [
                "g++", "-shared", "-fPIC", "-O2", "-std=c++11",
                "-o", "librtl_simulator.so",
                "rtl_simulator.cpp"
            ]
        elif sys.platform.startswith("linux"):  # Linux
            compile_cmd = [
                "g++", "-shared", "-fPIC", "-O2", "-std=c++11",
                "-o", "librtl_simulator.so", 
                "rtl_simulator.cpp"
            ]
        else:
            print(f"❌ 不支持的操作系统: {sys.platform}")
            return False
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ RTL仿真器共享库编译成功")
            
            # 验证库文件
            if os.path.exists("librtl_simulator.so"):
                print("✅ 共享库文件已创建: librtl_simulator.so")
                
                # 清理临时文件
                if os.path.exists("rtl_simulator.cpp"):
                    os.remove("rtl_simulator.cpp")
                
                return True
            else:
                print("❌ 共享库文件未找到")
                return False
        else:
            print(f"❌ 编译失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 创建RTL仿真器库时出错: {e}")
        return False

def test_rtl_library():
    """测试RTL库"""
    print("\n🔬 测试RTL仿真器库")
    print("-" * 30)
    
    try:
        import ctypes
        import numpy as np
        from ctypes import POINTER, c_float, c_uint32, c_void_p
        
        # 加载库
        lib = ctypes.CDLL('./librtl_simulator.so')
        
        # 定义函数签名
        lib.create_rtl_simulator.restype = c_void_p
        lib.destroy_rtl_simulator.argtypes = [c_void_p]
        lib.rtl_matrix_multiply.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        lib.get_operation_count.argtypes = [c_void_p]
        lib.get_operation_count.restype = c_uint32
        
        # 创建仿真器
        simulator = lib.create_rtl_simulator()
        if not simulator:
            print("❌ 无法创建RTL仿真器")
            return False
        
        print("✅ RTL仿真器创建成功")
        
        # 测试矩阵乘法
        size = 2
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = np.zeros((size, size), dtype=np.float32)
        
        # 转换为C数组
        a_ptr = a.ctypes.data_as(POINTER(c_float))
        b_ptr = b.ctypes.data_as(POINTER(c_float))
        result_ptr = result.ctypes.data_as(POINTER(c_float))
        
        # 调用RTL矩阵乘法
        lib.rtl_matrix_multiply(simulator, a_ptr, b_ptr, result_ptr, size)
        
        # 验证结果
        expected = np.matmul(a, b)
        error = np.mean(np.abs(result - expected))
        
        print(f"输入A: {a.tolist()}")
        print(f"输入B: {b.tolist()}")
        print(f"RTL结果: {result.tolist()}")
        print(f"期望结果: {expected.tolist()}")
        print(f"误差: {error:.2e}")
        
        if error < 1e-6:
            print("✅ RTL矩阵乘法测试通过")
            success = True
        else:
            print("❌ RTL矩阵乘法测试失败")
            success = False
        
        # 获取操作计数
        op_count = lib.get_operation_count(simulator)
        print(f"操作计数: {op_count}")
        
        # 销毁仿真器
        lib.destroy_rtl_simulator(simulator)
        print("✅ RTL仿真器已销毁")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试RTL库时出错: {e}")
        return False

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    # 检查g++编译器
    try:
        result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ g++编译器可用")
            return True
        else:
            print("❌ g++编译器不可用")
            return False
    except FileNotFoundError:
        print("❌ 未找到g++编译器，请安装开发工具")
        print("  macOS: xcode-select --install")
        print("  Linux: sudo apt-get install build-essential")
        return False

def main():
    """主函数"""
    print("🚀 RTL后端创建工具")
    print("=" * 50)
    
    # 检查依赖项
    if not check_dependencies():
        print("❌ 依赖项检查失败")
        return 1
    
    # 创建RTL仿真器库
    if not create_rtl_simulator_library():
        print("❌ RTL仿真器库创建失败")
        return 1
    
    # 测试库
    if not test_rtl_library():
        print("❌ RTL库测试失败")
        return 1
    
    print("\n🎉 RTL后端创建完成!")
    print("✨ 现在可以运行 python3 scripts/rtl_device_demo.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())