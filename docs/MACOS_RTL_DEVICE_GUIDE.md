# 🖥️ macOS上运行RTL代码并作为设备加载的完整指南

## 📋 概述

在macOS上将RTL代码作为设备加载主要有以下几种方法：

1. **软件仿真器** - 将RTL编译为软件库
2. **FPGA开发板** - 通过USB/PCIe连接的FPGA
3. **虚拟设备驱动** - 创建内核扩展或用户空间驱动
4. **Docker容器** - 在容器中运行RTL仿真
5. **云端FPGA** - 使用AWS F1等云服务

---

## 🔧 方法1: 软件仿真器作为设备 (推荐)

### 1.1 使用Verilator创建共享库

我们之前已经创建了基础版本，现在扩展为完整的设备接口：

```bash
# 安装依赖
brew install verilator
brew install python3
pip3 install ctypes numpy
```

### 1.2 创建设备接口

```cpp
// rtl_device_interface.cpp
#include "Vriscv_ai_chip.h"
#include "verilated.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <memory>

class RTLDevice {
private:
    std::unique_ptr<Vriscv_ai_chip> chip;
    std::thread simulation_thread;
    std::mutex device_mutex;
    std::queue<uint32_t> command_queue;
    bool running;
    uint64_t cycle_count;
    
public:
    RTLDevice() : running(false), cycle_count(0) {
        chip = std::make_unique<Vriscv_ai_chip>();
        Verilated::traceEverOn(true);
        
        // 初始化芯片
        reset_chip();
        std::cout << "🔧 RTL设备已初始化" << std::endl;
    }
    
    ~RTLDevice() {
        stop_device();
    }
    
    // 启动设备
    bool start_device() {
        if (running) return true;
        
        running = true;
        simulation_thread = std::thread(&RTLDevice::simulation_loop, this);
        std::cout << "🚀 RTL设备已启动" << std::endl;
        return true;
    }
    
    // 停止设备
    void stop_device() {
        if (!running) return;
        
        running = false;
        if (simulation_thread.joinable()) {
            simulation_thread.join();
        }
        std::cout << "⏹️ RTL设备已停止" << std::endl;
    }
    
    // 设备I/O接口
    uint32_t read_register(uint32_t addr) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        // 设置地址和读取信号
        chip->addr = addr;
        chip->read_enable = 1;
        
        // 执行一个时钟周期
        chip->clk = 0;
        chip->eval();
        chip->clk = 1;
        chip->eval();
        
        chip->read_enable = 0;
        return chip->data_out;
    }
    
    void write_register(uint32_t addr, uint32_t data) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        // 设置地址、数据和写入信号
        chip->addr = addr;
        chip->data_in = data;
        chip->write_enable = 1;
        
        // 执行一个时钟周期
        chip->clk = 0;
        chip->eval();
        chip->clk = 1;
        chip->eval();
        
        chip->write_enable = 0;
    }
    
    // TPU计算接口
    bool tpu_matrix_multiply(float* a, float* b, float* result, int size) {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        // 配置TPU
        write_register(0x1000, 0x01); // TPU_CTRL: 启用
        write_register(0x1004, size); // MATRIX_SIZE
        write_register(0x1008, 0x00); // OP_TYPE: 矩阵乘法
        
        // 加载数据 (简化版)
        for (int i = 0; i < size * size; i++) {
            write_register(0x2000 + i * 4, *((uint32_t*)&a[i]));
            write_register(0x3000 + i * 4, *((uint32_t*)&b[i]));
        }
        
        // 启动计算
        write_register(0x1000, 0x03); // TPU_CTRL: 启用+开始
        
        // 等待完成
        uint32_t status;
        int timeout = 10000;
        do {
            status = read_register(0x100C); // TPU_STATUS
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        } while (!(status & 0x01) && --timeout > 0);
        
        if (timeout <= 0) {
            std::cerr << "❌ TPU计算超时" << std::endl;
            return false;
        }
        
        // 读取结果
        for (int i = 0; i < size * size; i++) {
            uint32_t result_bits = read_register(0x4000 + i * 4);
            result[i] = *((float*)&result_bits);
        }
        
        std::cout << "✅ TPU矩阵乘法完成" << std::endl;
        return true;
    }
    
    // 获取设备状态
    struct DeviceStatus {
        bool is_running;
        uint64_t cycle_count;
        uint32_t tpu_status;
        uint32_t memory_usage;
    };
    
    DeviceStatus get_device_status() {
        std::lock_guard<std::mutex> lock(device_mutex);
        
        DeviceStatus status;
        status.is_running = running;
        status.cycle_count = cycle_count;
        status.tpu_status = read_register(0x100C);
        status.memory_usage = read_register(0x5000);
        
        return status;
    }
    
private:
    void reset_chip() {
        chip->rst_n = 0;
        chip->clk = 0;
        chip->eval();
        chip->clk = 1;
        chip->eval();
        chip->rst_n = 1;
        chip->eval();
    }
    
    void simulation_loop() {
        while (running) {
            std::lock_guard<std::mutex> lock(device_mutex);
            
            // 执行时钟周期
            chip->clk = 0;
            chip->eval();
            chip->clk = 1;
            chip->eval();
            
            cycle_count++;
            
            // 控制仿真速度
            std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        }
    }
};

// C接口供Python调用
extern "C" {
    RTLDevice* create_rtl_device() {
        return new RTLDevice();
    }
    
    void destroy_rtl_device(RTLDevice* device) {
        delete device;
    }
    
    bool start_rtl_device(RTLDevice* device) {
        return device->start_device();
    }
    
    void stop_rtl_device(RTLDevice* device) {
        device->stop_device();
    }
    
    uint32_t read_rtl_register(RTLDevice* device, uint32_t addr) {
        return device->read_register(addr);
    }
    
    void write_rtl_register(RTLDevice* device, uint32_t addr, uint32_t data) {
        device->write_register(addr, data);
    }
    
    bool rtl_tpu_matmul(RTLDevice* device, float* a, float* b, float* result, int size) {
        return device->tpu_matrix_multiply(a, b, result, size);
    }
}
```

### 1.3 Python设备驱动

```python
# rtl_device_driver.py
import ctypes
import numpy as np
import threading
import time
from ctypes import POINTER, c_float, c_uint32, c_bool, c_void_p

class RTLDeviceDriver:
    """RTL设备驱动 - 将RTL仿真作为硬件设备"""
    
    def __init__(self, lib_path="./librtl_device.so"):
        # 加载RTL设备库
        self.lib = ctypes.CDLL(lib_path)
        
        # 定义函数签名
        self.lib.create_rtl_device.restype = c_void_p
        self.lib.destroy_rtl_device.argtypes = [c_void_p]
        self.lib.start_rtl_device.argtypes = [c_void_p]
        self.lib.start_rtl_device.restype = c_bool
        self.lib.stop_rtl_device.argtypes = [c_void_p]
        
        self.lib.read_rtl_register.argtypes = [c_void_p, c_uint32]
        self.lib.read_rtl_register.restype = c_uint32
        self.lib.write_rtl_register.argtypes = [c_void_p, c_uint32, c_uint32]
        
        self.lib.rtl_tpu_matmul.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        self.lib.rtl_tpu_matmul.restype = c_bool
        
        # 创建设备实例
        self.device = self.lib.create_rtl_device()
        if not self.device:
            raise RuntimeError("无法创建RTL设备")
        
        # 启动设备
        if not self.lib.start_rtl_device(self.device):
            raise RuntimeError("无法启动RTL设备")
        
        print("🔧 RTL设备驱动已初始化")
    
    def __del__(self):
        if hasattr(self, 'device') and self.device:
            self.lib.stop_rtl_device(self.device)
            self.lib.destroy_rtl_device(self.device)
    
    def read_register(self, addr):
        """读取设备寄存器"""
        return self.lib.read_rtl_register(self.device, addr)
    
    def write_register(self, addr, data):
        """写入设备寄存器"""
        self.lib.write_rtl_register(self.device, addr, data)
    
    def get_device_info(self):
        """获取设备信息"""
        return {
            "device_type": "RTL Simulation Device",
            "vendor": "RISC-V AI Chip",
            "version": "1.0.0",
            "tpu_status": self.read_register(0x100C),
            "memory_base": 0x2000,
            "tpu_base": 0x1000
        }
    
    def tpu_matrix_multiply(self, a, b):
        """使用TPU进行矩阵乘法"""
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"矩阵维度不匹配: {a.shape} @ {b.shape}")
        
        size = a.shape[0]
        result = np.zeros((size, b.shape[1]), dtype=np.float32)
        
        # 转换为C数组
        a_ptr = a.astype(np.float32).ctypes.data_as(POINTER(c_float))
        b_ptr = b.astype(np.float32).ctypes.data_as(POINTER(c_float))
        result_ptr = result.ctypes.data_as(POINTER(c_float))
        
        # 调用RTL设备
        success = self.lib.rtl_tpu_matmul(self.device, a_ptr, b_ptr, result_ptr, size)
        
        if not success:
            raise RuntimeError("TPU矩阵乘法失败")
        
        return result
    
    def benchmark_performance(self, sizes=[64, 128, 256]):
        """性能基准测试"""
        results = {}
        
        for size in sizes:
            print(f"🧮 测试 {size}x{size} 矩阵乘法...")
            
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # RTL设备测试
            start_time = time.time()
            rtl_result = self.tpu_matrix_multiply(a, b)
            rtl_time = time.time() - start_time
            
            # CPU参考测试
            start_time = time.time()
            cpu_result = np.matmul(a, b)
            cpu_time = time.time() - start_time
            
            # 计算误差
            error = np.mean(np.abs(rtl_result - cpu_result))
            
            results[size] = {
                "rtl_time": rtl_time,
                "cpu_time": cpu_time,
                "speedup": cpu_time / rtl_time if rtl_time > 0 else 0,
                "error": error,
                "gflops": (2 * size**3) / (rtl_time * 1e9) if rtl_time > 0 else 0
            }
            
            print(f"  RTL时间: {rtl_time:.4f}s")
            print(f"  CPU时间: {cpu_time:.4f}s") 
            print(f"  加速比: {results[size]['speedup']:.2f}x")
            print(f"  GFLOPS: {results[size]['gflops']:.2f}")
            print(f"  误差: {error:.2e}")
        
        return results

# 设备管理器
class RTLDeviceManager:
    """RTL设备管理器 - 管理多个RTL设备实例"""
    
    def __init__(self):
        self.devices = {}
        self.device_count = 0
    
    def create_device(self, device_name=None):
        """创建新的RTL设备"""
        if device_name is None:
            device_name = f"rtl_device_{self.device_count}"
        
        if device_name in self.devices:
            raise ValueError(f"设备 {device_name} 已存在")
        
        try:
            device = RTLDeviceDriver()
            self.devices[device_name] = device
            self.device_count += 1
            
            print(f"✅ 创建设备: {device_name}")
            return device_name
        except Exception as e:
            print(f"❌ 创建设备失败: {e}")
            return None
    
    def get_device(self, device_name):
        """获取设备实例"""
        return self.devices.get(device_name)
    
    def list_devices(self):
        """列出所有设备"""
        return list(self.devices.keys())
    
    def remove_device(self, device_name):
        """移除设备"""
        if device_name in self.devices:
            del self.devices[device_name]
            print(f"🗑️ 移除设备: {device_name}")
            return True
        return False
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            "total_devices": len(self.devices),
            "active_devices": len([d for d in self.devices.values() if d]),
            "devices": {}
        }
        
        for name, device in self.devices.items():
            try:
                status["devices"][name] = device.get_device_info()
            except:
                status["devices"][name] = {"status": "error"}
        
        return status

if __name__ == "__main__":
    # 示例使用
    print("🔧 RTL设备驱动测试")
    
    try:
        # 创建设备管理器
        manager = RTLDeviceManager()
        
        # 创建设备
        device_name = manager.create_device("ai_chip_0")
        device = manager.get_device(device_name)
        
        if device:
            # 获取设备信息
            info = device.get_device_info()
            print(f"📊 设备信息: {info}")
            
            # 性能测试
            print("\n🚀 开始性能基准测试...")
            results = device.benchmark_performance([32, 64])
            
            print("\n📈 性能测试结果:")
            for size, result in results.items():
                print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS")
        
        # 系统状态
        status = manager.get_system_status()
        print(f"\n🖥️ 系统状态: {status}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
```

### 1.4 编译脚本

```makefile
# Makefile.device
VERILATOR = verilator
VERILATOR_FLAGS = --cc --exe --build --trace -Wall
RTL_DIR = rtl
TOP_MODULE = riscv_ai_chip

# RTL源文件
RTL_SOURCES = \
	$(RTL_DIR)/top/$(TOP_MODULE).sv \
	$(RTL_DIR)/core/*.sv \
	$(RTL_DIR)/accelerators/*.sv \
	$(RTL_DIR)/memory/*.sv

# 编译RTL设备库
librtl_device.so: rtl_device_interface.cpp $(RTL_SOURCES)
	@echo "🔨 编译RTL设备库..."
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		-I$(RTL_DIR) \
		$(RTL_SOURCES) rtl_device_interface.cpp
	
	@echo "📦 创建共享库..."
	g++ -shared -fPIC -o librtl_device.so \
		obj_dir/V$(TOP_MODULE)__ALL.a \
		-lverilated -lverilated_vcd -pthread

# 测试设备
test_device: librtl_device.so
	@echo "🧪 测试RTL设备..."
	python3 rtl_device_driver.py

# 清理
clean:
	rm -rf obj_dir/ *.so *.vcd

.PHONY: test_device clean
```

---

## 🔧 方法2: FPGA开发板作为设备

### 2.1 支持的FPGA开发板

在macOS上推荐使用以下FPGA开发板：

```bash
# 推荐的FPGA开发板
1. Xilinx Zynq UltraScale+ (ZCU102/ZCU104)
2. Intel/Altera Cyclone V (DE10-Nano)
3. Lattice ECP5 (ULX3S)
4. Xilinx Artix-7 (Arty A7)
```

### 2.2 FPGA设备驱动

```python
# fpga_device_driver.py
import usb.core
import usb.util
import struct
import time

class FPGADevice:
    """FPGA设备驱动"""
    
    def __init__(self, vendor_id=0x0403, product_id=0x6010):
        # 查找FPGA设备
        self.device = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        
        if self.device is None:
            raise RuntimeError("未找到FPGA设备")
        
        # 配置设备
        self.device.set_configuration()
        
        # 获取端点
        cfg = self.device.get_active_configuration()
        intf = cfg[(0, 0)]
        
        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        )
        
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )
        
        print("🔧 FPGA设备已连接")
    
    def write_register(self, addr, data):
        """写入FPGA寄存器"""
        cmd = struct.pack('<BII', 0x01, addr, data)  # 写命令
        self.ep_out.write(cmd)
    
    def read_register(self, addr):
        """读取FPGA寄存器"""
        cmd = struct.pack('<BI', 0x02, addr)  # 读命令
        self.ep_out.write(cmd)
        
        # 读取响应
        response = self.ep_in.read(4)
        return struct.unpack('<I', response)[0]
    
    def load_bitstream(self, bitstream_path):
        """加载比特流到FPGA"""
        with open(bitstream_path, 'rb') as f:
            bitstream = f.read()
        
        # 发送配置命令
        cmd = struct.pack('<BI', 0x03, len(bitstream))
        self.ep_out.write(cmd)
        
        # 发送比特流数据
        chunk_size = 1024
        for i in range(0, len(bitstream), chunk_size):
            chunk = bitstream[i:i+chunk_size]
            self.ep_out.write(chunk)
        
        print(f"✅ 比特流已加载: {bitstream_path}")
    
    def tpu_matrix_multiply(self, a, b):
        """使用FPGA TPU进行矩阵乘法"""
        # 配置TPU
        self.write_register(0x1000, 0x01)  # 启用TPU
        self.write_register(0x1004, a.shape[0])  # 矩阵大小
        
        # 传输数据到FPGA内存
        self._transfer_matrix(0x2000, a)
        self._transfer_matrix(0x3000, b)
        
        # 启动计算
        self.write_register(0x1008, 0x01)  # 开始计算
        
        # 等待完成
        while not (self.read_register(0x100C) & 0x01):
            time.sleep(0.001)
        
        # 读取结果
        result = self._read_matrix(0x4000, (a.shape[0], b.shape[1]))
        
        return result
    
    def _transfer_matrix(self, base_addr, matrix):
        """传输矩阵数据到FPGA"""
        flat = matrix.flatten().astype(np.float32)
        for i, val in enumerate(flat):
            val_bits = struct.unpack('<I', struct.pack('<f', val))[0]
            self.write_register(base_addr + i * 4, val_bits)
    
    def _read_matrix(self, base_addr, shape):
        """从FPGA读取矩阵数据"""
        size = shape[0] * shape[1]
        data = []
        
        for i in range(size):
            val_bits = self.read_register(base_addr + i * 4)
            val = struct.unpack('<f', struct.pack('<I', val_bits))[0]
            data.append(val)
        
        return np.array(data).reshape(shape)
```

---

## 🔧 方法3: 虚拟设备驱动

### 3.1 用户空间设备驱动

```python
# virtual_device_driver.py
import os
import mmap
import struct
import threading
from multiprocessing import shared_memory

class VirtualRTLDevice:
    """虚拟RTL设备 - 通过共享内存通信"""
    
    def __init__(self, device_name="rtl_chip_0"):
        self.device_name = device_name
        self.memory_size = 1024 * 1024  # 1MB共享内存
        
        # 创建共享内存
        try:
            self.shm = shared_memory.SharedMemory(
                name=device_name, 
                create=True, 
                size=self.memory_size
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=device_name)
        
        # 内存映射
        self.memory = memoryview(self.shm.buf)
        
        # 设备寄存器映射
        self.registers = {
            'DEVICE_ID': 0x0000,
            'STATUS': 0x0004,
            'CONTROL': 0x0008,
            'TPU_CTRL': 0x1000,
            'TPU_STATUS': 0x1004,
            'MATRIX_SIZE': 0x1008,
            'DATA_BASE': 0x2000,
        }
        
        # 初始化设备
        self._initialize_device()
        
        print(f"🔧 虚拟RTL设备已创建: {device_name}")
    
    def _initialize_device(self):
        """初始化设备寄存器"""
        self.write_register('DEVICE_ID', 0x12345678)
        self.write_register('STATUS', 0x00000001)  # 设备就绪
        self.write_register('CONTROL', 0x00000000)
    
    def read_register(self, reg_name):
        """读取寄存器"""
        if isinstance(reg_name, str):
            addr = self.registers[reg_name]
        else:
            addr = reg_name
        
        # 从共享内存读取
        data = struct.unpack('<I', self.memory[addr:addr+4])[0]
        return data
    
    def write_register(self, reg_name, value):
        """写入寄存器"""
        if isinstance(reg_name, str):
            addr = self.registers[reg_name]
        else:
            addr = reg_name
        
        # 写入共享内存
        struct.pack_into('<I', self.memory, addr, value)
    
    def get_device_path(self):
        """获取设备路径 (模拟)"""
        return f"/dev/rtl_{self.device_name}"
    
    def __del__(self):
        if hasattr(self, 'shm'):
            self.shm.close()

# 设备管理器
class VirtualDeviceManager:
    """虚拟设备管理器"""
    
    def __init__(self):
        self.devices = {}
    
    def create_device(self, device_name):
        """创建虚拟设备"""
        if device_name not in self.devices:
            self.devices[device_name] = VirtualRTLDevice(device_name)
            return True
        return False
    
    def list_devices(self):
        """列出所有虚拟设备"""
        return [
            {
                "name": name,
                "path": device.get_device_path(),
                "status": "active" if device else "inactive"
            }
            for name, device in self.devices.items()
        ]
    
    def get_device(self, device_name):
        """获取设备实例"""
        return self.devices.get(device_name)
```

### 3.2 系统集成脚本

```bash
#!/bin/bash
# setup_rtl_device.sh - 设置RTL设备环境

echo "🔧 设置RTL设备环境..."

# 创建设备目录
sudo mkdir -p /dev/rtl
sudo chmod 755 /dev/rtl

# 创建设备节点 (模拟)
for i in {0..3}; do
    device_path="/dev/rtl/chip$i"
    if [ ! -e "$device_path" ]; then
        sudo mknod "$device_path" c 240 $i
        sudo chmod 666 "$device_path"
        echo "✅ 创建设备节点: $device_path"
    fi
done

# 设置环境变量
export RTL_DEVICE_PATH="/dev/rtl"
export RTL_LIB_PATH="$(pwd)/librtl_device.so"

echo "🎉 RTL设备环境设置完成!"
echo "设备路径: $RTL_DEVICE_PATH"
echo "库路径: $RTL_LIB_PATH"
```

---

## 🔧 方法4: Docker容器化

### 4.1 Docker容器

```dockerfile
# Dockerfile.rtl
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    verilator \
    build-essential \
    python3 \
    python3-pip \
    git

# 安装Python包
RUN pip3 install numpy ctypes-sh

# 复制RTL代码
COPY rtl/ /app/rtl/
COPY *.cpp /app/
COPY *.py /app/
COPY Makefile.device /app/

WORKDIR /app

# 编译RTL设备
RUN make -f Makefile.device librtl_device.so

# 暴露端口
EXPOSE 8080

# 启动设备服务
CMD ["python3", "rtl_device_server.py"]
```

### 4.2 设备服务器

```python
# rtl_device_server.py
from flask import Flask, request, jsonify
import numpy as np
from rtl_device_driver import RTLDeviceDriver

app = Flask(__name__)

# 全局设备实例
device = None

@app.route('/device/init', methods=['POST'])
def init_device():
    global device
    try:
        device = RTLDeviceDriver()
        return jsonify({"status": "success", "message": "设备已初始化"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/device/info', methods=['GET'])
def get_device_info():
    if device is None:
        return jsonify({"status": "error", "message": "设备未初始化"})
    
    try:
        info = device.get_device_info()
        return jsonify({"status": "success", "data": info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/tpu/matmul', methods=['POST'])
def tpu_matrix_multiply():
    if device is None:
        return jsonify({"status": "error", "message": "设备未初始化"})
    
    try:
        data = request.json
        a = np.array(data['matrix_a'], dtype=np.float32)
        b = np.array(data['matrix_b'], dtype=np.float32)
        
        result = device.tpu_matrix_multiply(a, b)
        
        return jsonify({
            "status": "success",
            "result": result.tolist()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## 🚀 使用示例

### 完整使用流程

```python
# main_example.py
import numpy as np
from rtl_device_driver import RTLDeviceManager

def main():
    print("🔧 RTL设备使用示例")
    
    # 1. 创建设备管理器
    manager = RTLDeviceManager()
    
    # 2. 创建RTL设备
    device_name = manager.create_device("ai_chip_main")
    device = manager.get_device(device_name)
    
    if not device:
        print("❌ 设备创建失败")
        return
    
    # 3. 获取设备信息
    info = device.get_device_info()
    print(f"📊 设备信息: {info}")
    
    # 4. 执行矩阵乘法
    print("\n🧮 执行矩阵乘法...")
    a = np.random.randn(128, 128).astype(np.float32)
    b = np.random.randn(128, 128).astype(np.float32)
    
    result = device.tpu_matrix_multiply(a, b)
    
    # 5. 验证结果
    cpu_result = np.matmul(a, b)
    error = np.mean(np.abs(result - cpu_result))
    
    print(f"✅ 计算完成，误差: {error:.2e}")
    
    # 6. 性能测试
    print("\n🚀 性能基准测试...")
    benchmark_results = device.benchmark_performance([64, 128, 256])
    
    for size, result in benchmark_results.items():
        print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS")
    
    # 7. 系统状态
    status = manager.get_system_status()
    print(f"\n🖥️ 系统状态: {status}")

if __name__ == "__main__":
    main()
```

---

## 📋 总结

### 推荐方案排序

1. **软件仿真器** ⭐⭐⭐⭐⭐
   - 最容易实现
   - 完全兼容macOS
   - 开发调试方便

2. **FPGA开发板** ⭐⭐⭐⭐
   - 真实硬件性能
   - 需要额外硬件
   - 适合原型验证

3. **虚拟设备驱动** ⭐⭐⭐
   - 系统集成度高
   - 实现复杂度中等
   - 适合系统测试

4. **Docker容器** ⭐⭐
   - 部署简单
   - 性能有损失
   - 适合云端部署

### 快速开始

```bash
# 1. 编译RTL设备库
make -f Makefile.device librtl_device.so

# 2. 测试设备
python3 rtl_device_driver.py

# 3. 运行示例
python3 main_example.py
```

现在你的RTL代码就可以作为设备在macOS上运行了！🎉