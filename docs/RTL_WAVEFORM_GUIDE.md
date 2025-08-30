# RTL波形查看指南

## 🌊 如何查看RTL仿真波形

### 1. 快速查看波形

```bash
# 方法1: 使用快速脚本
./open_waveforms.sh

# 方法2: 直接使用GTKWave
cd verification/simple_rtl
gtkwave test_simple_tpu_mac.vcd

# 方法3: 使用波形查看器工具
python3 view_rtl_waveforms.py
```

### 2. 生成新的波形文件

```bash
# 运行RTL后端测试，会自动生成波形
python3 rtl_hardware_backend.py

# 或者运行完整的设备系统测试
python3 rtl_device_system.py
```

## 📊 波形文件说明

### 主要波形文件
- **`test_simple_tpu_mac.vcd`** - TPU MAC单元的完整测试波形
- **`simple_mac_test.vcd`** - 简单MAC测试波形

### 波形内容
```
🔬 Testing Simplified TPU MAC Unit RTL Code
==========================================
✅ Reset released, TPU MAC unit enabled
🧮 Test 1 (INT8): 10 * 20 + 5 = 205 ✅ PASSED
🧮 Test 2 (INT16): 7 * 8 + 100 = 156 ✅ PASSED  
🧮 Test 3 (INT32): 15 * 4 + 25 = 85 ✅ PASSED
🧮 Test 4 (Zero): 0 * 999 + 42 = 42 ✅ PASSED
```

## 🔍 GTKWave使用指南

### 启动GTKWave后的操作步骤：

1. **添加信号到波形窗口**
   - 在左侧的信号树中，展开模块 `test_simple_tpu_mac`
   - 展开子模块 `dut` (Device Under Test)
   - 选择要查看的信号，右键选择 "Append" 或直接拖拽到波形窗口

2. **重要信号说明**
   ```
   📡 时钟和控制信号:
   - clk          : 系统时钟 (100MHz)
   - rst_n        : 复位信号 (低电平有效)
   - enable       : 使能信号
   - valid_in     : 输入数据有效
   - valid_out    : 输出数据有效
   - ready        : 设备就绪
   
   📊 数据信号:
   - a_data[15:0] : MAC输入A (16位)
   - b_data[15:0] : MAC输入B (16位) 
   - c_data[31:0] : MAC输入C (32位)
   - result[31:0] : MAC输出结果 (32位)
   - data_type[2:0] : 数据类型选择
   
   🔧 内部信号:
   - mac_result[31:0] : 内部MAC结果
   - result_valid     : 内部结果有效标志
   ```

3. **查看测试序列**
   - 测试1: `10 * 20 + 5 = 205` (INT8模式)
   - 测试2: `7 * 8 + 100 = 156` (INT16模式)  
   - 测试3: `15 * 4 + 25 = 85` (INT32模式)
   - 测试4: `0 * 999 + 42 = 42` (零乘法测试)

### GTKWave快捷键：
- **Ctrl + A**: 全选所有信号
- **Ctrl + F**: 适应窗口显示所有时间
- **Ctrl + G**: 跳转到指定时间
- **鼠标滚轮**: 缩放时间轴
- **左右箭头**: 移动时间光标

## 📈 波形分析要点

### 1. 时序检查
- 确认时钟信号正常 (50% 占空比)
- 检查复位序列 (rst_n从0到1的转换)
- 验证建立时间和保持时间

### 2. 功能验证
- 输入数据变化时，输出是否正确响应
- valid_in和valid_out的握手时序
- MAC计算结果是否正确

### 3. 性能分析
- 计算延迟 (从valid_in到valid_out的时间)
- 吞吐率 (连续计算的间隔时间)
- 流水线效率

## 🛠️ 高级波形分析

### 1. 添加计算信号
在GTKWave中可以添加计算信号来验证MAC操作：
```
# 在GTKWave的"Insert"菜单中选择"Analog"
# 添加表达式: a_data * b_data + c_data
# 与result信号对比验证
```

### 2. 时间测量
- 使用时间光标测量关键路径延迟
- 分析时钟周期内的信号变化
- 验证时序约束

### 3. 导出波形数据
```bash
# 从GTKWave导出为其他格式
# File -> Write Save File (保存会话)
# File -> Print (导出为图片)
```

## 🔧 故障排除

### 如果GTKWave无法打开：
```bash
# 检查GTKWave安装
which gtkwave

# macOS安装GTKWave
brew install --cask gtkwave

# 检查VCD文件
ls -la verification/simple_rtl/*.vcd
```

### 如果波形文件为空：
```bash
# 重新运行RTL仿真
python3 rtl_hardware_backend.py

# 检查RTL编译
cd verification/simple_rtl
iverilog -o test_simple_tpu_mac -g2012 test_simple_tpu_mac.sv simple_tpu_mac.sv
vvp test_simple_tpu_mac
```

## 📝 波形文件信息

### 当前波形统计：
```
📊 test_simple_tpu_mac.vcd:
  - 文件大小: 2511 bytes
  - 仿真时间: 0 - 445000 ps (445 ns)
  - 信号数量: 25个信号
  - 模块: test_simple_tpu_mac, dut
  
📊 simple_mac_test.vcd:
  - 文件大小: 1278 bytes  
  - 仿真时间: 0 - 315000 ps (315 ns)
  - 信号数量: 8个信号
  - 模块: simple_mac_test
```

## 🎯 学习建议

1. **从简单信号开始**: 先查看时钟、复位等基本信号
2. **理解时序关系**: 观察输入输出的时序关系
3. **验证功能正确性**: 对比计算结果与期望值
4. **分析性能指标**: 测量延迟和吞吐率
5. **调试问题**: 使用波形定位时序或功能问题

## 🚀 下一步

- 尝试修改RTL代码，观察波形变化
- 添加更多测试用例
- 分析不同数据类型的性能差异
- 优化RTL设计以提高性能

---

**记住**: 波形是理解和调试RTL设计的最重要工具！🌊