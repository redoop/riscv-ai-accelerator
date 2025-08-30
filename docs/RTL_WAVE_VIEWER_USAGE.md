# RTL波形查看器使用指南

## 概述
RTL波形查看器是专门为RISC-V AI芯片项目开发的波形分析工具，用于查看和分析RTL仿真生成的VCD波形文件。

## 功能特性

### 🌊 专业波形分析
- **信号自动分类**: 自动将信号分为时钟、控制、数据、状态四大类
- **交互式缩放**: 支持1x到16x的时间轴缩放
- **时间范围控制**: 可自定义查看的时间范围(皮秒精度)
- **信号统计**: 提供详细的信号变化统计信息

### 📊 可视化界面
- **分类显示**: 按信号类型分组显示，便于分析
- **颜色编码**: 不同类型信号使用不同颜色标识
- **时间轴**: 精确的时间标尺显示
- **值标签**: 显示信号的具体数值变化

### 💾 数据处理
- **JSON导出**: 支持将波形数据导出为JSON格式
- **信号分析**: 提供信号变化次数和统计信息
- **十六进制显示**: 多bit信号自动转换为十六进制显示

## 使用方法

### 1. 基本使用
```bash
# 进入包含VCD文件的目录
cd verification/unit_tests

# 运行波形查看器
python3 rtl_wave_viewer.py
```

### 2. 从scripts目录运行
```bash
# 使用scripts目录中的查看器
cd scripts
python3 rtl_wave_viewer.py
```

### 3. 查看生成的HTML文件
波形查看器会为每个VCD文件生成对应的HTML查看器：
```bash
# 在浏览器中打开
open tpu_mac_array_test_rtl_waveform.html
```

## 支持的文件格式

### VCD文件
- **格式**: Value Change Dump (VCD)
- **生成工具**: Icarus Verilog, Verilator
- **文件扩展名**: `.vcd`

### 输出格式
- **HTML查看器**: `*_rtl_waveform.html`
- **JSON数据**: `*_rtl_waveform_data.json`

## 界面说明

### 控制面板
- **时间范围**: 设置查看的开始和结束时间
- **缩放级别**: 选择时间轴的缩放倍数
- **更新波形**: 应用新的时间范围和缩放设置
- **导出数据**: 将波形数据导出为JSON文件
- **信号分析**: 显示详细的信号统计信息

### 信号分类

#### 🕐 时钟信号 (红色)
- 系统时钟 (`clk`, `clock`)
- 复位信号 (`rst`, `reset`)

#### 🎛️ 控制信号 (橙色)
- 使能信号 (`enable`, `en`)
- 有效信号 (`valid`)
- 就绪信号 (`ready`)

#### 📊 数据信号 (蓝色)
- 地址信号 (`addr`)
- 数据信号 (`data`, `wdata`, `rdata`)

#### 📈 状态信号 (绿色)
- 状态机信号
- 计数器信号
- 其他状态指示信号

## 实际应用示例

### TPU MAC数组波形分析
```bash
# 运行TPU测试生成波形
cd verification/unit_tests
./run_all_tpu_tests.sh

# 查看生成的波形
python3 rtl_wave_viewer.py

# 在浏览器中查看
open tpu_mac_array_test_rtl_waveform.html
```

### 分析要点
1. **时钟域**: 检查时钟信号的周期和占空比
2. **控制流**: 观察使能和就绪信号的时序关系
3. **数据流**: 分析数据信号的变化和传输时序
4. **状态机**: 跟踪状态信号的转换过程

## 故障排除

### 常见问题

#### 1. 找不到VCD文件
```
❌ 没有找到RTL VCD文件
```
**解决方案**: 确保在包含VCD文件的目录中运行脚本

#### 2. 解析失败
```
❌ 解析RTL VCD文件失败
```
**解决方案**: 检查VCD文件是否完整，重新运行RTL仿真

#### 3. 浏览器无法打开HTML文件
**解决方案**: 
- 检查文件路径是否正确
- 尝试直接拖拽HTML文件到浏览器
- 使用`python3 -m http.server`启动本地服务器

### 性能优化
- 对于大型VCD文件，建议设置合适的时间范围
- 使用缩放功能聚焦于感兴趣的时间段
- 定期清理不需要的HTML文件以节省空间

## 技术细节

### 支持的信号类型
- **单bit信号**: 显示为高低电平波形
- **多bit信号**: 显示为十六进制数值
- **向量信号**: 自动解析位宽和数值

### 时间精度
- **基本单位**: 皮秒 (ps)
- **显示精度**: 根据时间范围自动调整
- **缩放范围**: 1x - 16x

### 浏览器兼容性
- Chrome/Chromium (推荐)
- Firefox
- Safari
- Edge

## 相关文档
- [RTL_WAVEFORM_GENERATION_REPORT.md](RTL_WAVEFORM_GENERATION_REPORT.md) - RTL波形生成报告
- [RTL_WAVEFORM_GUIDE.md](RTL_WAVEFORM_GUIDE.md) - RTL波形查看指南
- [JS_WAVEFORM_VIEWER_GUIDE.md](JS_WAVEFORM_VIEWER_GUIDE.md) - JavaScript波形查看器指南

---

*最后更新: 2025-08-31*