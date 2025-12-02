# Iverilog 仿真环境

## 📋 目录说明

这个目录包含用于 VCS/Verdi 仿真的测试环境和工具。虽然目录名为 `iverilog`，但实际上是为 Synopsys VCS 仿真器设计的。

## 📁 目录结构

```
iverilog/
├── asic_top.sv              # ASIC 顶层模块
├── filelist/                # 文件列表
│   ├── asic_tblist.f        # Testbench 文件列表
│   ├── asic_top.f           # 顶层文件列表
│   ├── ip.f                 # IP 文件列表
│   ├── lib.f                # 库文件列表
│   └── soc.f                # SoC 文件列表
├── lib/                     # 库文件
│   ├── icsIOA_N55_3P3.v     # ICS55 I/O 库
│   └── tc_clk.sv            # 时钟模型
├── rcu/                     # RCU 模块
│   └── rcu.sv               # 复位控制单元
├── run/                     # 运行目录
│   └── Makefile             # VCS 仿真 Makefile
├── tb/                      # Testbench
│   ├── include/             # 头文件
│   ├── N25Qxxx.v            # Flash 模型
│   ├── soc_tb.sv            # SoC testbench
│   └── tty.v                # UART 终端模型
└── utils/                   # 工具模块
    ├── clk_int_div.sv       # 时钟分频器
    ├── config.svh           # 配置文件
    ├── register.sv          # 寄存器模块
    ├── rst_sync.sv          # 复位同步器
    ├── stdcell.sv           # 标准单元
    └── xchecker.sv          # 交叉检查器
```

## 🔧 使用方法

### 方法 1: 使用 Icarus Verilog (开源) ⭐ 推荐

如果没有 VCS，使用开源的 Icarus Verilog：

```bash
cd run

# 使用 Icarus Verilog Makefile
make -f Makefile.iverilog compile  # 编译
make -f Makefile.iverilog sim      # 仿真
make -f Makefile.iverilog wave     # 查看波形
make -f Makefile.iverilog clean    # 清理
```

### 方法 2: 使用项目主脚本 ⭐⭐ 最推荐

使用项目提供的完整仿真脚本：

```bash
cd chisel/synthesis

# 运行后综合仿真
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

### 方法 3: 使用 VCS (Synopsys)

如果有 VCS 商业许可：

```bash
cd run

# 编译
make comp

# 仿真
make sim

# 查看波形
make wave

# 清理
make clean
```

## 📝 注意事项

1. **工具要求**：
   - **VCS**: Synopsys VCS 仿真器（商业软件，需要许可证）
   - **Verdi**: Synopsys Verdi 波形查看器（商业软件）
   - **Icarus Verilog**: 开源仿真器（免费，推荐）✅
   - **GTKWave**: 开源波形查看器（免费）✅

2. **VCS 错误处理**：
   如果看到 `make: vcs: No such file or directory` 错误：
   - 使用 `make -f Makefile.iverilog` 代替
   - 或使用项目主仿真脚本
   - 详见 [IVERILOG_USAGE.md](IVERILOG_USAGE.md)

3. **文件路径**：
   - 原始 Makefile 中的某些路径可能需要调整
   - `Makefile.iverilog` 已修复路径问题
   - 特别是 `SIM_INC` 中的绝对路径

4. **仿真选项**：
   - VCS: 全功能仿真，包含时序检查和 SVA 断言
   - Icarus Verilog: 功能仿真，部分 SystemVerilog 支持

## 🔄 与项目集成

这个目录是独立的仿真环境，与项目主要的仿真流程分离。

**推荐使用项目主流程：**

```bash
# 1. RTL 仿真（Chisel + Verilator）
cd chisel
sbt test

# 2. 后综合仿真（Icarus Verilog）
cd chisel/synthesis
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# 3. 查看波形
cd waves
gtkwave post_syn.vcd
```

## 📚 相关文档

- [综合指南](../README.md)
- [ICS55 综合结果](../ICS55_SYNTHESIS_RESULTS.md)
- [后综合仿真脚本](../run_post_syn_sim.py)

## ⚠️ 免责声明

此目录中的某些文件可能包含特定项目的路径和配置，使用前请根据实际环境调整。

---

**创建日期**: 2025-11-21  
**用途**: VCS/Verdi 仿真环境  
**状态**: 参考用途
