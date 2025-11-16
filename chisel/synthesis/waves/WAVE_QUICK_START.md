# 波形查看快速指南

## 最简单的方法：静态 HTML 页面（推荐）

### 一键生成并查看

```bash
cd chisel/synthesis

# 方法 1: 使用便捷脚本（推荐）
./view_wave.sh

# 方法 2: 手动指定参数
./view_wave.sh -f waves/post_syn.vcd -s 20 -p 3000

# 方法 3: 直接使用 Python 脚本
python3 generate_static_wave.py waves/post_syn.vcd
```

### 特点

✅ **无需服务器** - 生成独立的 HTML 文件，双击即可打开  
✅ **快速生成** - 332MB VCD 文件约 40 秒生成完成  
✅ **体积小** - 输出 HTML 仅 100KB 左右  
✅ **易分享** - 可以直接发送 HTML 文件给他人查看  
✅ **支持大文件** - 可处理任意大小的 VCD 文件  

### 生成的文件包含

- 📊 高质量波形图（PNG 格式，嵌入 HTML）
- 📋 信号列表（名称、路径、位宽）
- ℹ️ 文件信息（大小、时间范围等）
- 💾 下载波形图功能
- 🖨️ 打印功能

### 使用场景

- ✅ 日常波形查看
- ✅ 报告和文档
- ✅ 团队分享
- ✅ 远程查看（无需服务器）
- ✅ 离线查看

---

## 其他方法

### 方法 2: Web 服务器（适合交互式查看）

```bash
# 启动 Web 服务器
python3 wave_viewer.py --port 5000

# 在浏览器中打开
# http://localhost:5000
```

**三种模式**:
- `/` - 流式加载模式（默认，适合大文件）
- `/image` - 图片模式（一次性渲染）
- `/canvas` - Canvas 模式（适合小文件）

### 方法 3: 传统工具

```bash
# GTKWave（需要 X11）
gtkwave waves/post_syn.vcd

# Verdi（商业工具）
verdi -ssf waves/post_syn.vcd
```

---

## 完整工作流程

### 1. 运行仿真

```bash
cd chisel/synthesis
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

### 2. 查看波形

```bash
# 生成静态页面（推荐）
./view_wave.sh

# 或使用 Web 服务器
python3 wave_viewer.py
```

### 3. 查看报告

```bash
cat sim/post_syn_report.txt
```

---

## 参数说明

### view_wave.sh 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f, --file` | VCD 文件路径 | `waves/post_syn.vcd` |
| `-o, --output` | 输出 HTML 文件 | 自动生成 |
| `-s, --signals` | 最大信号数量 | 15 |
| `-p, --points` | 最大采样点数 | 2000 |

### 示例

```bash
# 显示更多信号
./view_wave.sh -s 30

# 提高采样率（更清晰，但生成更慢）
./view_wave.sh -p 5000

# 指定输出文件
./view_wave.sh -o my_waveform.html

# 组合使用
./view_wave.sh -f waves/post_syn.vcd -s 20 -p 3000 -o detailed_wave.html
```

---

## 性能对比

| 方法 | VCD 大小 | 生成时间 | 输出大小 | 优点 | 缺点 |
|------|---------|---------|---------|------|------|
| **静态 HTML** | 332 MB | ~40s | 100 KB | 快速、独立、易分享 | 信号数量有限 |
| Web 流式 | 任意 | 实时 | N/A | 支持大量信号 | 需要服务器 |
| Web 图片 | <500 MB | 实时 | N/A | 完整渲染 | 信号数量有限 |
| GTKWave | 任意 | N/A | N/A | 功能强大 | 需要 GUI |

---

## 故障排除

### 问题 1: matplotlib 未安装

```bash
python3 -m pip install matplotlib --user
```

### 问题 2: 生成时间过长

减少信号数量或采样点数：
```bash
./view_wave.sh -s 10 -p 1000
```

### 问题 3: HTML 文件过大

- 减少信号数量：`-s 10`
- 减少采样点：`-p 1000`
- 或使用 Web 服务器模式

### 问题 4: 无法打开 HTML

确保使用现代浏览器（Chrome、Firefox、Edge）

---

## 最佳实践

### 日常使用

```bash
# 快速查看关键信号（10 个）
./view_wave.sh -s 10

# 生成后自动打开
./view_wave.sh && xdg-open waveform_post_syn.html
```

### 详细分析

```bash
# 更多信号和更高采样率
./view_wave.sh -s 30 -p 5000 -o detailed_analysis.html
```

### 团队分享

```bash
# 生成标准报告
./view_wave.sh -s 15 -p 2000 -o team_review.html

# 发送 team_review.html 给团队成员
```

### 文档归档

```bash
# 为每次测试生成独立的波形页面
./view_wave.sh -o "waveform_$(date +%Y%m%d_%H%M%S).html"
```

---

## 提示

💡 **推荐配置**: `-s 15 -p 2000` 平衡了质量和速度  
💡 **快速预览**: `-s 5 -p 1000` 最快生成  
💡 **详细分析**: `-s 30 -p 5000` 最高质量  
💡 **自动选择**: 脚本会优先选择 clock、reset、trap 等关键信号  

---

## 总结

对于大多数使用场景，**静态 HTML 页面**是最佳选择：

```bash
./view_wave.sh
```

简单、快速、无需配置，生成的 HTML 文件可以直接在浏览器中打开！
