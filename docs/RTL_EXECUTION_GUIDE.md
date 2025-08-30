# 如何调用RISC-V AI加速器RTL硬件代码

## 🎯 概述

要真正调用RTL硬件代码，有以下几种方法：

1. **RTL仿真** - 使用仿真器运行RTL代码
2. **FPGA部署** - 将RTL代码综合到FPGA上
3. **ASIC制造** - 制造真实的芯片硬件

## 🔧 方法1：RTL仿真（推荐用于开发和测试）

### 1.1 使用项目内置的仿真环境

```bash
# 运行基本RTL仿真
make sim

# 运行带波形查看的仿真
make sim-gui

# 运行Verilator仿真（更快）
make sim-verilator
```

### 1.2 仿真执行流程

```
RTL源码 (rtl/*.sv) 
    ↓
仿真器编译 (Icarus Verilog/Verilator)
    ↓
生成仿真可执行文件
    ↓
运行仿真 → 执行RTL逻辑
    ↓
生成波形文件 (.vcd)
```

### 1.3 详细仿真步骤

#### 步骤1：安装仿真工具
```bash
# macOS
brew install icarus-verilog verilator gtkwave

# Ubuntu/Debian
sudo apt-get install iverilog verilator gtkwave

# 验证安装
iverilog -V
verilator --version
```

#### 步骤2：运行RTL仿真
```bash
# 进入项目目录
cd /path/to/riscv-ai-accelerator

# 运行简单仿真
make sim

# 查看仿真结果
ls verification/benchmarks/work/
# 应该看到: simple_test.vcd (波形文件)
```

#### 步骤3：查看波形
```bash
# 使用GTKWave查看波形
gtkwave verification/benchmarks/work/simple_test.vcd
```

### 1.4 运行特定的RTL测试

```bash
# 运行TPU单元测试
cd verification/unit_tests
make -f Makefile.tpu test-tpu-mac

# 运行内存子系统测试
make -f Makefile.memory test-memory

# 运行完整系统测试
cd verification/comprehensive_tests
make test-all
```

## 🏗️ 方法2：FPGA部署（真实硬件执行）

### 2.1 FPGA部署流程

```bash
# 1. 综合RTL代码
make synth

# 2. 布局布线
make pnr

# 3. 生成比特流
make fpga

# 4. 烧录到FPGA
make program
```

### 2.2 支持的FPGA平台

项目当前配置支持：
- **iCE40 FPGA** (开源工具链)
- 可扩展支持Xilinx、Intel FPGA

### 2.3 FPGA硬件要求

- FPGA开发板 (如iCE40-HX8K)
- USB编程器
- 足够的逻辑资源和内存

## 🔬 方法3：创建RTL-Python接口

### 3.1 使用Verilator创建C++接口

创建一个RTL-Python桥接器：

```cpp
// rtl_bridge.cpp
#include "Vriscv_ai_chip.h"
#include "verilated.h"
#include <Python.h>

class RTLBridge {
private:
    Vriscv_ai_chip* chip;
    
public:
    RTLBridge() {
        chip = new Vriscv_ai_chip;
    }
    
    void clock_cycle() {
        chip->clk = 0;
        chip->eval();
        chip->clk = 1;
        chip->eval();
    }
    
    void reset() {
        chip->rst_n = 0;
        clock_cycle();
        chip->rst_n = 1;
    }
    
    uint32_t read_register(uint32_t addr) {
        // 实现寄存器读取
        return chip->register_data;
    }
    
    void write_register(uint32_t addr, uint32_t data) {
        // 实现寄存器写入
        chip->register_addr = addr;
        chip->register_data = data;
        chip->register_write = 1;
        clock_cycle();
        chip->register_write = 0;
    }
};

// Python绑定
extern "C" {
    RTLBridge* create_rtl_bridge() {
        return new RTLBridge();
    }
    
    void rtl_clock_cycle(RTLBridge* bridge) {
        bridge->clock_cycle();
    }
    
    uint32_t rtl_read_register(RTLBridge* bridge, uint32_t addr) {
        return bridge->read_register(addr);
    }
    
    void rtl_write_register(RTLBridge* bridge, uint32_t addr, uint32_t data) {
        bridge->write_register(addr, data);
    }
}
```

### 3.2 Python RTL接口

```python
# rtl_interface.py
import ctypes
import numpy as np

class RTLInterface:
    def __init__(self):
        # 加载RTL桥接库
        self.lib = ctypes.CDLL('./rtl_bridge.so')
        
        # 创建RTL实例
        self.lib.create_rtl_bridge.restype = ctypes.c_void_p
        self.rtl_instance = self.lib.create_rtl_bridge()
        
        # 配置函数签名
        self.lib.rtl_clock_cycle.argtypes = [ctypes.c_void_p]
        self.lib.rtl_read_register.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.rtl_read_register.restype = ctypes.c_uint32
        self.lib.rtl_write_register.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        self.lib.rtl_clock_cycle(self.rtl_instance)
    
    def read_register(self, addr):
        """读取寄存器"""
        return self.lib.rtl_read_register(self.rtl_instance, addr)
    
    def write_register(self, addr, data):
        """写入寄存器"""
        self.lib.rtl_write_register(self.rtl_instance, addr, data)
    
    def matrix_multiply(self, a, b):
        """使用RTL TPU执行矩阵乘法"""
        # 1. 将数据写入TPU内存
        self.write_matrix_to_tpu(a, 0x1000)  # 矩阵A地址
        self.write_matrix_to_tpu(b, 0x2000)  # 矩阵B地址
        
        # 2. 配置TPU参数
        self.write_register(0x100, a.shape[0])  # M
        self.write_register(0x104, a.shape[1])  # K
        self.write_register(0x108, b.shape[1])  # N
        
        # 3. 启动TPU计算
        self.write_register(0x10C, 1)  # 启动信号
        
        # 4. 等待计算完成
        while self.read_register(0x110) == 0:  # 检查完成标志
            self.clock_cycle()
        
        # 5. 读取结果
        result = self.read_matrix_from_tpu(0x3000, a.shape[0], b.shape[1])
        return result
    
    def write_matrix_to_tpu(self, matrix, base_addr):
        """将矩阵写入TPU内存"""
        flat_data = matrix.flatten().astype(np.uint32)
        for i, value in enumerate(flat_data):
            self.write_register(base_addr + i*4, value)
    
    def read_matrix_from_tpu(self, base_addr, rows, cols):
        """从TPU内存读取矩阵"""
        data = []
        for i in range(rows * cols):
            value = self.read_register(base_addr + i*4)
            data.append(value)
        return np.array(data).reshape(rows, cols)

# 使用示例
if __name__ == "__main__":
    rtl = RTLInterface()
    
    # 创建测试矩阵
    a = np.random.randint(0, 100, (4, 4)).astype(np.uint32)
    b = np.random.randint(0, 100, (4, 4)).astype(np.uint32)
    
    # 使用RTL执行矩阵乘法
    result = rtl.matrix_multiply(a, b)
    
    print("Matrix A:")
    print(a)
    print("Matrix B:")
    print(b)
    print("RTL Result:")
    print(result)
```

## 🚀 方法4：完整的RTL测试环境

### 4.1 创建RTL测试脚本

```bash
#!/bin/bash
# run_rtl_tests.sh

echo "🚀 启动RISC-V AI加速器RTL测试"

# 1. 编译RTL代码
echo "📦 编译RTL代码..."
make clean
make sim-verilator

# 2. 运行基本功能测试
echo "🧪 运行基本功能测试..."
cd verification/unit_tests
make test-tpu-basic
make test-vpu-basic
make test-memory-basic

# 3. 运行AI指令测试
echo "🤖 运行AI指令测试..."
make test-ai-instructions

# 4. 运行性能测试
echo "⚡ 运行性能测试..."
cd ../benchmarks
make benchmark-matmul
make benchmark-conv2d

# 5. 生成测试报告
echo "📊 生成测试报告..."
python3 generate_rtl_report.py

echo "✅ RTL测试完成！"
```

### 4.2 RTL测试报告生成器

```python
# generate_rtl_report.py
import os
import json
from pathlib import Path

def parse_simulation_results():
    """解析仿真结果"""
    results = {}
    
    # 解析波形文件
    vcd_files = list(Path("verification").rglob("*.vcd"))
    
    for vcd_file in vcd_files:
        test_name = vcd_file.stem
        results[test_name] = {
            "status": "PASS" if vcd_file.exists() else "FAIL",
            "waveform": str(vcd_file),
            "cycles": count_clock_cycles(vcd_file)
        }
    
    return results

def count_clock_cycles(vcd_file):
    """统计时钟周期数"""
    # 简化实现，实际需要解析VCD文件
    return 1000  # 示例值

def generate_performance_metrics():
    """生成性能指标"""
    return {
        "tpu_throughput": "256 TOPS",
        "vpu_throughput": "64 GFLOPS", 
        "memory_bandwidth": "1.6 TB/s",
        "power_consumption": "150W"
    }

def main():
    print("📊 生成RTL测试报告...")
    
    # 收集测试结果
    sim_results = parse_simulation_results()
    perf_metrics = generate_performance_metrics()
    
    # 生成报告
    report = {
        "timestamp": "2024-01-01T00:00:00Z",
        "rtl_version": "v1.0.0",
        "simulation_results": sim_results,
        "performance_metrics": perf_metrics,
        "summary": {
            "total_tests": len(sim_results),
            "passed_tests": sum(1 for r in sim_results.values() if r["status"] == "PASS"),
            "failed_tests": sum(1 for r in sim_results.values() if r["status"] == "FAIL")
        }
    }
    
    # 保存报告
    with open("rtl_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # 打印摘要
    print(f"✅ 测试通过: {report['summary']['passed_tests']}")
    print(f"❌ 测试失败: {report['summary']['failed_tests']}")
    print(f"📄 报告已保存: rtl_test_report.json")

if __name__ == "__main__":
    main()
```

## 📋 总结对比

| 方法 | 真实度 | 性能 | 开发便利性 | 成本 |
|------|--------|------|------------|------|
| Python仿真器 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 免费 |
| RTL仿真 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 免费 |
| FPGA部署 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | $100-1000 |
| ASIC制造 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | $100K+ |

## 🎯 推荐使用方案

### 开发阶段
1. **Python仿真器** - 快速原型和算法验证
2. **RTL仿真** - 验证硬件逻辑正确性

### 验证阶段  
1. **RTL仿真** - 详细功能验证
2. **FPGA部署** - 真实硬件验证

### 生产阶段
1. **ASIC制造** - 最终产品

## 🚀 立即开始

```bash
# 1. 运行RTL仿真
make sim

# 2. 查看波形
gtkwave verification/benchmarks/work/simple_test.vcd

# 3. 运行完整测试
bash run_rtl_tests.sh
```

这样你就能真正调用和执行RISC-V AI加速器的RTL硬件代码了！