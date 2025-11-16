# 波形查看工具

本目录包含用于查看和分析 VCD 波形文件的工具集。

## 📁 目录结构

```
waves/
├── README.md                       # 本文件
├── wave_viewer.py                  # Web 波形查看器（Flask）
├── wave_renderer.py                # 波形渲染器（后端生成图片）
├── serve_wave.py                   # 简单 HTTP 服务器
├── generate_static_wave.py         # 生成静态 HTML 波形页面
├── view_wave.sh                    # 快速查看波形脚本
├── start_wave_viewer.sh            # 启动 Web 查看器
├── start_http.sh                   # 启动 HTTP 服务器
├── test_wave_viewer.py             # 测试脚本
├── test_image_render.py            # 图片渲染测试
├── WAVE_VIEWER_README.md           # Web 查看器详细文档
├── WAVE_VIEWER_USAGE.md            # 使用指南
├── WAVE_VIEWER_OPTIMIZATION.md     # 优化说明
└── WAVE_QUICK_START.md             # 快速开始指南
```

## 🚀 快速开始

### 方法 1: 生成静态 HTML（推荐）

最简单的方式，生成独立的 HTML 文件，无需服务器：

```bash
cd chisel/synthesis/waves

# 查看当前目录的 VCD 文件
./view_wave.sh

# 指定 VCD 文件
./view_wave.sh -f post_syn.vcd

# 自定义参数
./view_wave.sh -f post_syn.vcd -s 20 -p 3000 -o my_wave.html
```

生成的 HTML 文件可以直接在浏览器中打开。

### 方法 2: Web 波形查看器

交互式 Web 界面，支持实时缩放和信号选择：

```bash
cd chisel/synthesis/waves

# 启动 Web 服务器
./start_wave_viewer.sh

# 或手动启动
python3 wave_viewer.py --port 5000
```

然后在浏览器中访问 `http://localhost:5000`

### 方法 3: 简单 HTTP 服务器

查看已生成的静态 HTML 文件：

```bash
cd chisel/synthesis/waves

# 启动 HTTP 服务器
./start_http.sh

# 或手动启动
python3 serve_wave.py -p 8000
```

## 📋 工具说明

### 1. view_wave.sh - 快速查看脚本

生成静态波形 HTML 页面的便捷脚本。

**用法：**
```bash
./view_wave.sh [选项]

选项:
  -f, --file FILE      VCD 文件路径 (默认: post_syn.vcd)
  -o, --output FILE    输出 HTML 文件 (默认: 自动生成)
  -s, --signals NUM    最大信号数量 (默认: 15)
  -p, --points NUM     最大采样点数 (默认: 2000)
  -h, --help           显示帮助
```

**示例：**
```bash
# 使用默认参数
./view_wave.sh

# 指定 VCD 文件
./view_wave.sh -f ../sim/my_test.vcd

# 更多信号和采样点
./view_wave.sh -s 30 -p 5000

# 指定输出文件
./view_wave.sh -f post_syn.vcd -o waveform.html
```

### 2. generate_static_wave.py - 静态页面生成器

Python 脚本，生成包含波形图的独立 HTML 文件。

**用法：**
```bash
python3 generate_static_wave.py <vcd_file> [选项]

选项:
  -o, --output FILE       输出 HTML 文件
  --max-signals NUM       最大信号数量 (默认: 20)
  --max-points NUM        最大采样点数 (默认: 3000)
```

**示例：**
```bash
# 基本用法
python3 generate_static_wave.py post_syn.vcd

# 自定义输出
python3 generate_static_wave.py post_syn.vcd -o my_wave.html

# 更多信号
python3 generate_static_wave.py post_syn.vcd --max-signals 50
```

### 3. wave_viewer.py - Web 波形查看器

基于 Flask 的交互式 Web 波形查看器。

**特点：**
- 实时加载 VCD 文件
- 交互式信号选择
- 时间范围缩放
- 支持大文件（智能抽样）
- 后端渲染（服务器生成图片）

**用法：**
```bash
python3 wave_viewer.py [选项]

选项:
  --port PORT          Web 服务器端口 (默认: 5000)
  --host HOST          服务器地址 (默认: 0.0.0.0)
  --wave-dir DIR       波形文件目录 (默认: .)
```

**示例：**
```bash
# 默认配置
python3 wave_viewer.py

# 自定义端口
python3 wave_viewer.py --port 8080

# 指定波形目录
python3 wave_viewer.py --wave-dir ../sim
```

### 4. serve_wave.py - HTTP 服务器

简单的 HTTP 服务器，用于查看静态 HTML 文件。

**用法：**
```bash
python3 serve_wave.py [选项]

选项:
  -p, --port PORT      HTTP 端口 (默认: 8000)
  -d, --directory DIR  服务目录 (默认: .)
  --no-browser         不自动打开浏览器
```

**示例：**
```bash
# 默认配置
python3 serve_wave.py

# 自定义端口
python3 serve_wave.py -p 8080

# 指定目录
python3 serve_wave.py -d ../sim
```

## 🔧 依赖安装

### 必需依赖

```bash
# matplotlib - 用于波形渲染
pip3 install matplotlib --user

# Flask - 用于 Web 查看器
pip3 install flask --user
```

### 自动安装

脚本会自动检查并安装缺失的依赖：

```bash
# 运行任何脚本时会自动安装
./view_wave.sh
./start_wave_viewer.sh
```

## 📊 使用场景

### 场景 1: 快速查看仿真结果

```bash
# 运行仿真
cd ../
python3 run_post_syn_sim.py

# 查看波形
cd waves
./view_wave.sh -f post_syn.vcd
```

### 场景 2: 分析特定信号

```bash
# 生成包含更多信号的波形
./view_wave.sh -s 50 -p 5000

# 或使用 Web 查看器交互式选择
./start_wave_viewer.sh
```

### 场景 3: 分享波形结果

```bash
# 生成独立 HTML 文件
./view_wave.sh -f post_syn.vcd -o report_wave.html

# 将 report_wave.html 发送给他人
# 接收者无需任何工具，直接在浏览器中打开即可
```

### 场景 4: 调试大型设计

```bash
# 使用 Web 查看器的交互功能
./start_wave_viewer.sh

# 在浏览器中:
# 1. 选择感兴趣的信号
# 2. 缩放到特定时间范围
# 3. 导出为图片
```

## 🎯 最佳实践

### 1. 选择合适的工具

- **快速查看**: 使用 `view_wave.sh`
- **详细分析**: 使用 `wave_viewer.py`
- **分享结果**: 生成静态 HTML
- **大文件**: 使用 Web 查看器（支持抽样）

### 2. 优化性能

```bash
# 对于大文件，限制信号数量
./view_wave.sh -s 10 -p 1000

# 或使用 Web 查看器的智能抽样
python3 wave_viewer.py
```

### 3. 自定义信号选择

编辑 `generate_static_wave.py` 中的优先级关键字：

```python
priority_keywords = ['clk', 'clock', 'reset', 'trap', 'valid', 'ready', 'irq']
```

## 📚 详细文档

- [WAVE_VIEWER_README.md](WAVE_VIEWER_README.md) - Web 查看器详细说明
- [WAVE_VIEWER_USAGE.md](WAVE_VIEWER_USAGE.md) - 使用指南
- [WAVE_VIEWER_OPTIMIZATION.md](WAVE_VIEWER_OPTIMIZATION.md) - 性能优化
- [WAVE_QUICK_START.md](WAVE_QUICK_START.md) - 快速开始

## 🐛 故障排除

### 问题 1: 找不到 VCD 文件

```bash
# 检查文件是否存在
ls -la *.vcd

# 或指定完整路径
./view_wave.sh -f ../sim/post_syn.vcd
```

### 问题 2: 依赖未安装

```bash
# 手动安装所有依赖
pip3 install matplotlib flask --user
```

### 问题 3: 端口被占用

```bash
# 使用其他端口
python3 wave_viewer.py --port 5001
python3 serve_wave.py -p 8001
```

### 问题 4: 波形图为空

```bash
# 检查 VCD 文件是否有效
head -20 post_syn.vcd

# 增加信号数量
./view_wave.sh -s 50
```

## 💡 提示

1. **VCD 文件位置**: 仿真生成的 VCD 文件通常在当前目录或 `../sim/` 目录
2. **输出文件**: 生成的 HTML 文件默认保存在当前目录
3. **浏览器兼容**: 支持所有现代浏览器（Chrome、Firefox、Safari、Edge）
4. **文件大小**: 静态 HTML 文件通常比原始 VCD 文件小很多
5. **性能**: 对于超大文件（>100MB），建议使用 Web 查看器的抽样功能

## 🔗 相关链接

- [VCD 格式规范](https://en.wikipedia.org/wiki/Value_change_dump)
- [GTKWave](http://gtkwave.sourceforge.net/) - 传统波形查看器
- [Matplotlib](https://matplotlib.org/) - Python 绘图库
- [Flask](https://flask.palletsprojects.com/) - Python Web 框架

---

**快速命令参考:**

```bash
# 最常用的命令
./view_wave.sh                          # 快速查看
./start_wave_viewer.sh                  # Web 查看器
./start_http.sh                         # HTTP 服务器

# 从 synthesis 目录运行
cd chisel/synthesis
./waves/view_wave.sh -f waves/post_syn.vcd
```
