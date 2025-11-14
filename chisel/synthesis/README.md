# 逻辑综合后网表仿真

本目录包含逻辑综合后网表的仿真测试平台和脚本。

## 📁 目录结构

```
synthesis/
├── README.md                    # 本文件
├── Makefile                     # Make 构建文件
├── run_post_syn_sim.py         # Python 自动化脚本
├── netlist/                     # 综合后的网表文件
│   └── SimpleEdgeAiSoC_syn.v   # 综合网表（需要生成）
├── testbench/                   # 测试平台
│   ├── post_syn_tb.sv          # 基本测试平台
│   ├── advanced_post_syn_tb.sv # 高级测试平台
│   ├── test_utils.sv           # 测试工具
│   └── filelist.f              # 文件列表
├── sim/                         # 仿真输出目录
│   ├── compile.log             # 编译日志
│   ├── sim.log                 # 仿真日志
│   └── *_report.txt            # 测试报告
└── waves/                       # 波形文件目录
    └── *.vcd                    # VCD 波形文件
```

## 🚀 快速开始

### 前提条件

1. **综合网表**: 需要先运行逻辑综合生成网表文件
2. **仿真工具**: VCS、Verilator 或其他 Verilog 仿真器
3. **波形查看器**: Verdi、GTKWave 等

### 方法 1: 使用 Python 脚本（推荐）

```bash
# 运行完整仿真流程
python run_post_syn_sim.py

# 使用 Verilator
python run_post_syn_sim.py --simulator verilator

# 使用基本测试平台
python run_post_syn_sim.py --testbench basic

# 查看波形
python run_post_syn_sim.py --wave

# 生成报告
python run_post_syn_sim.py --report

# 查看帮助
python run_post_syn_sim.py --help
```

### 方法 2: 使用 Makefile

```bash
# 查看帮助
make help

# 编译高级测试平台
make compile_advanced

# 运行仿真
make sim_advanced

# 查看波形
make wave

# 生成报告
make report

# 完整流程
make full

# 清理
make clean
```

### 方法 3: 手动运行

```bash
# 使用 VCS
vcs -full64 -sverilog \
    -timescale=1ns/1ps \
    -debug_all \
    -f testbench/filelist.f \
    netlist/SimpleEdgeAiSoC_syn.v \
    testbench/advanced_post_syn_tb.sv \
    -o sim/simv

# 运行仿真
./sim/simv -l sim/sim.log

# 查看波形
verdi -ssf waves/advanced_post_syn.vcd
```

## 📋 测试平台说明

### 基本测试平台 (post_syn_tb.sv)

**特点:**
- 简单的功能验证
- 快速运行
- 基本的信号监控

**测试内容:**
1. 系统启动测试
2. GPIO 功能测试
3. 中断信号测试
4. 长时间运行稳定性测试

### 高级测试平台 (advanced_post_syn_tb.sv)

**特点:**
- 详细的功能测试
- 性能分析
- 完整的测试报告

**测试内容:**
1. 复位功能测试
2. 基本操作测试
3. GPIO 模式测试
4. 中断响应测试
5. UART 接口测试
6. 压力测试
7. 性能分析

## 📊 测试报告

仿真完成后会生成详细的测试报告：

```
synthesis/sim/detailed_report.txt
```

报告内容包括：
- 设计信息
- 测试结果
- 统计信息
- 性能分析
- 结论

## 🔍 调试方法

### 查看仿真日志

```bash
# 编译日志
cat sim/compile_advanced.log

# 仿真日志
cat sim/sim_advanced.log

# 测试报告
cat sim/detailed_report.txt
```

### 查看波形

```bash
# 使用 Verdi
make wave

# 使用 GTKWave
make wave_gtk

# 或直接使用工具
verdi -ssf waves/advanced_post_syn.vcd
gtkwave waves/advanced_post_syn.vcd
```

### 关键信号

在波形中重点关注：
- `clock` - 时钟信号
- `reset` - 复位信号
- `trap` - 异常信号
- `compact_irq` - CompactAccel 中断
- `bitnet_irq` - BitNetAccel 中断
- `gpio_out` - GPIO 输出
- `uart_tx` - UART 发送

## 🛠️ 工具支持

### VCS (Synopsys)

```bash
# 编译
vcs -full64 -sverilog -debug_all ...

# 仿真
./simv -l sim.log

# 波形
verdi -ssf waves/*.fsdb
```

### Verilator (开源)

```bash
# 编译和仿真
verilator --cc --exe --build --trace ...

# 运行
./obj_dir/Vsim

# 波形
gtkwave waves/*.vcd
```

### ModelSim/QuestaSim (Mentor)

```bash
# 编译
vlog -sv testbench/*.sv netlist/*.v

# 仿真
vsim -c -do "run -all" top

# 波形
vsim -view waves/*.wlf
```

## 📈 性能指标

仿真会统计以下性能指标：

- **仿真时间**: 总仿真时间（ns）
- **时钟周期**: 总时钟周期数
- **中断次数**: CompactAccel 和 BitNetAccel 中断次数
- **Trap 次数**: 异常发生次数
- **IPC**: 指令每周期（如果可测量）

## ⚠️ 注意事项

### 1. 网表文件

确保网表文件存在且正确：
```bash
ls -l netlist/SimpleEdgeAiSoC_syn.v
```

### 2. 标准单元库

如果使用特定工艺库，需要在 `filelist.f` 中添加：
```
+libext+.v
/path/to/stdcell/library.v
```

### 3. 时序

综合后网表包含时序信息，仿真时间会比 RTL 仿真长。

### 4. 功耗

可以使用 VCS 的功耗分析功能：
```bash
vcs -full64 -sverilog -debug_all -power ...
```

## 🔧 自定义测试

### 添加新的测试用例

1. 在测试平台中添加新的 task：

```systemverilog
task test_custom();
  $display("自定义测试...");
  // 测试代码
  $display("✓ 自定义测试完成");
endtask
```

2. 在主测试序列中调用：

```systemverilog
initial begin
  test_reset();
  test_custom();  // 添加这里
  test_basic_operation();
  // ...
end
```

### 修改测试参数

在测试平台顶部修改参数：

```systemverilog
parameter CLK_PERIOD = 10;  // 时钟周期 (ns)
parameter TEST_CYCLES = 1000;  // 测试周期数
```

## 📚 参考资料

### 综合相关
- [Synopsys Design Compiler User Guide](https://www.synopsys.com/)
- [Cadence Genus User Guide](https://www.cadence.com/)

### 仿真相关
- [VCS User Guide](https://www.synopsys.com/verification/simulation/vcs.html)
- [Verilator Manual](https://verilator.org/guide/latest/)
- [ModelSim User Manual](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/model-sim.html)

### 波形查看
- [Verdi User Guide](https://www.synopsys.com/verification/debug/verdi.html)
- [GTKWave Documentation](http://gtkwave.sourceforge.net/)

## 🤝 贡献

如果发现问题或有改进建议，请：
1. 检查现有的测试用例
2. 添加新的测试场景
3. 更新文档

## 📞 获取帮助

```bash
# 查看 Makefile 帮助
make help

# 查看 Python 脚本帮助
python run_post_syn_sim.py --help

# 查看工具信息
make info
```

---

**快速命令参考:**

```bash
# 完整流程
make full

# 或
python run_post_syn_sim.py

# 查看结果
make report
make wave
```
