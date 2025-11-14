# 逻辑综合后网表仿真 - 快速开始

## ⚡ 5 分钟快速开始

### 步骤 1: 准备网表

```bash
# 确保网表文件存在
ls synthesis/netlist/SimpleEdgeAiSoC_syn.v
```

### 步骤 2: 运行仿真

```bash
cd chisel/synthesis

# 方法 A: 使用 Python 脚本（推荐）
python run_post_syn_sim.py

# 方法 B: 使用 Makefile
make full
```

### 步骤 3: 查看结果

```bash
# 查看报告
cat sim/detailed_report.txt

# 查看波形
make wave
```

## 📋 常用命令

### Python 脚本

```bash
# 完整仿真
python run_post_syn_sim.py

# 使用 Verilator
python run_post_syn_sim.py --simulator verilator

# 基本测试
python run_post_syn_sim.py --testbench basic

# 查看波形
python run_post_syn_sim.py --wave

# 生成报告
python run_post_syn_sim.py --report

# 帮助
python run_post_syn_sim.py --help
```

### Makefile

```bash
# 查看帮助
make help

# 编译
make compile_advanced

# 仿真
make sim_advanced

# 波形
make wave

# 报告
make report

# 完整流程
make full

# 清理
make clean
```

## 📊 测试内容

✓ 复位功能  
✓ 基本操作  
✓ GPIO 测试  
✓ 中断响应  
✓ UART 接口  
✓ 压力测试  
✓ 性能分析  

## 📁 输出文件

- `sim/compile_advanced.log` - 编译日志
- `sim/sim_advanced.log` - 仿真日志
- `sim/detailed_report.txt` - 测试报告
- `waves/advanced_post_syn.vcd` - 波形文件

## 🛠️ 支持的工具

- **VCS** (Synopsys) - 默认
- **Verilator** - 开源
- **ModelSim** - 需要配置
- **Verdi** - 波形查看
- **GTKWave** - 波形查看

## ⚠️ 注意事项

1. 确保网表文件存在
2. 检查工具是否安装
3. 仿真时间较长（比 RTL 慢）
4. 需要标准单元库（如果使用特定工艺）

## 🔍 调试

```bash
# 查看日志
tail -f sim/sim_advanced.log

# 检查错误
grep -i error sim/*.log

# 查看波形
verdi -ssf waves/advanced_post_syn.vcd
```

## 📞 获取帮助

```bash
# 详细文档
cat README.md

# 工具信息
make info

# Python 帮助
python run_post_syn_sim.py --help
```

---

**快速命令:**
```bash
cd chisel/synthesis && python run_post_syn_sim.py
```
