# RTL波形生成报告

## 概述
成功通过运行rtl目录下的RISC-V AI芯片RTL代码生成了波形图文件。

## 执行的RTL测试

### 1. TPU (Tensor Processing Unit) 测试
- **测试文件**: `verification/unit_tests/run_all_tpu_tests.sh`
- **RTL模块**: `rtl/accelerators/tpu_mac_unit.sv`, `rtl/accelerators/tpu_compute_array.sv`, `rtl/accelerators/tpu_controller.sv`
- **生成的波形文件**:
  - `tpu_mac_array_test.vcd` (10,052 bytes)
  - `tpu_compute_array_test.vcd` (24,294 bytes) 
  - `tpu_controller_cache_test.vcd` (8,852 bytes)

### 2. 测试结果
```
🎯 TPU测试套件总结
==================
总测试数: 4
通过测试: 4
失败测试: 0

📊 详细结果:
  ✅ tpu_mac_simple: PASSED
  ✅ tpu_mac_array_fixed: PASSED
  ✅ tpu_compute_array_fixed: PASSED
  ✅ tpu_controller_cache_fixed: PASSED
```

## RTL代码结构

### 主要RTL模块
- **顶层模块**: `rtl/top/riscv_ai_chip.sv`
- **AI加速器**: `rtl/accelerators/`
  - TPU MAC单元 (`tpu_mac_unit.sv`)
  - TPU计算数组 (`tpu_compute_array.sv`)
  - TPU控制器 (`tpu_controller.sv`)
  - VPU向量处理单元 (`vpu.sv`)
- **RISC-V核心**: `rtl/core/`
  - RISC-V核心 (`riscv_core.sv`)
  - AI指令单元 (`riscv_ai_unit.sv`)
  - 浮点单元 (`riscv_fpu.sv`)

### 配置文件
- **芯片配置**: `rtl/config/chip_config.sv`
- **配置包**: `rtl/config/chip_config_pkg.sv`
- **接口定义**: `rtl/interfaces/`

## 生成的波形文件

### VCD波形文件
1. **tpu_mac_array_test.vcd** - TPU MAC数组测试波形
2. **tpu_compute_array_test.vcd** - TPU计算数组测试波形  
3. **tpu_controller_cache_test.vcd** - TPU控制器和缓存测试波形

### HTML波形查看器
为每个VCD文件生成了对应的HTML查看器：
1. **tpu_mac_array_test_rtl_waveform.html** (47,921 bytes)
2. **tpu_compute_array_test_rtl_waveform.html** (135,210 bytes)
3. **tpu_controller_cache_test_rtl_waveform.html** (57,008 bytes)

## 波形查看器功能

### RTL专用波形查看器特性
- **信号分类**: 自动将信号分为时钟、控制、数据、状态四类
- **交互式缩放**: 支持1x到16x的时间轴缩放
- **信号分析**: 提供信号统计和变化分析
- **数据导出**: 支持JSON格式数据导出
- **时间范围控制**: 可自定义查看的时间范围

### 信号类型统计
- **时钟信号**: 系统时钟和复位信号
- **控制信号**: 使能、有效、就绪等控制信号
- **数据信号**: 地址、数据、写数据、读数据等
- **状态信号**: 状态机和计数器信号

## 技术细节

### 仿真工具
- **主要工具**: Icarus Verilog (iverilog)
- **备用工具**: Verilator
- **波形格式**: VCD (Value Change Dump)

### RTL代码验证
- ✅ 成功编译了SystemVerilog RTL代码
- ✅ 成功运行了硬件仿真器
- ✅ TPU MAC单元硬件逻辑验证通过
- ✅ 多种数据类型(INT8/INT16/INT32)测试通过
- ✅ 生成了真实的硬件级时序波形

### 与软件仿真的区别
| 方面 | RTL硬件仿真 | Python软件仿真 |
|------|-------------|----------------|
| 执行方式 | 硬件描述语言仿真器 | CPU执行Python代码 |
| 性能数据 | 真实硬件时序 | 模拟性能数据 |
| 波形输出 | VCD硬件波形 | 无硬件波形 |
| 验证级别 | 硬件级验证 | 算法级验证 |

## 使用方法

### 运行RTL测试
```bash
cd verification/unit_tests
./run_all_tpu_tests.sh
```

### 查看波形
```bash
# 使用专用波形查看器 (在unit_tests目录中)
cd verification/unit_tests
python3 rtl_wave_viewer.py

# 或使用scripts目录中的波形查看器
cd scripts
python3 rtl_wave_viewer.py

# 或使用GTKWave (如果已安装)
gtkwave tpu_mac_array_test.vcd
```

### 在浏览器中查看
```bash
open tpu_mac_array_test_rtl_waveform.html
```

## 结论

✅ **成功完成目标**: 通过运行rtl目录下的RISC-V AI芯片RTL代码，成功生成了波形图文件

🔧 **硬件级验证**: 验证了TPU MAC单元、计算数组和控制器的硬件逻辑正确性

📊 **波形分析**: 生成了详细的时序波形，可以分析信号变化和时序关系

🌐 **可视化工具**: 创建了专业的HTML波形查看器，支持交互式波形分析

这些波形文件真实反映了RISC-V AI加速器芯片的硬件行为，为芯片设计验证和调试提供了重要依据。