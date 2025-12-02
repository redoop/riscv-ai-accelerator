# 使用 Icarus Verilog 进行仿真

## 📋 问题说明

原始的 `Makefile` 是为 Synopsys VCS（商业仿真器）设计的。如果你看到以下错误：

```
make: vcs: No such file or directory
make: *** [Makefile:58: comp] Error 127
```

这是因为系统中没有安装 VCS。

## 🔧 解决方案

### 方案 1: 使用 Icarus Verilog Makefile（推荐）

我们提供了一个支持开源 Icarus Verilog 的 Makefile：

```bash
cd chisel/synthesis/iverilog/run

# 使用 Icarus Verilog Makefile
make -f Makefile.iverilog compile  # 编译
make -f Makefile.iverilog sim      # 仿真
make -f Makefile.iverilog wave     # 查看波形
make -f Makefile.iverilog clean    # 清理
```

### 方案 2: 使用项目主仿真脚本（最推荐）

项目已经提供了完整的后综合仿真脚本：

```bash
cd chisel/synthesis

# 运行后综合仿真
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# 查看波形
cd waves
gtkwave post_syn.vcd
```

### 方案 3: 手动使用 Icarus Verilog

```bash
cd chisel/synthesis/iverilog/run

# 编译
iverilog -g2005-sv \
  -I../top -I../utils -I../tb -I../tb/include \
  -o soc_tb.vvp \
  -s soc_tb \
  $(cat ../filelist/*.f | grep -v "^#")

# 仿真
vvp soc_tb.vvp

# 查看波形
gtkwave soc_tb.vcd
```

## 🔄 工具对比

| 特性 | VCS (商业) | Icarus Verilog (开源) |
|------|-----------|---------------------|
| **成本** | 昂贵 | 免费 |
| **速度** | 非常快 | 中等 |
| **SystemVerilog** | 完整支持 | 部分支持 |
| **调试功能** | 强大 | 基本 |
| **适用场景** | 商业项目 | 开源项目、学习 |

## 📦 安装 Icarus Verilog

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install iverilog gtkwave
```

### macOS
```bash
brew install icarus-verilog gtkwave
```

### 使用 OSS CAD Suite（推荐）
```bash
# 已安装在 /opt/tools/oss-cad/oss-cad-suite/
# 包含 Icarus Verilog, GTKWave, Yosys 等所有工具
```

## 🎯 推荐工作流程

### 1. RTL 仿真（Chisel + Verilator）
```bash
cd chisel
sbt test
```

### 2. 逻辑综合（Yosys + ICS55 PDK）
```bash
cd chisel/synthesis
bash run_ics55_synthesis.sh
```

### 3. 后综合仿真（Icarus Verilog）
```bash
cd chisel/synthesis
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

### 4. 查看波形（GTKWave）
```bash
cd chisel/synthesis/waves
gtkwave post_syn.vcd
```

## 📝 注意事项

### 文件路径问题

原始 Makefile 中包含一些绝对路径，可能需要调整：

```makefile
# 这些路径可能不存在
+incdir+../../project_1854/source
+incdir+/nfs/share/home/qiming/soc/ysyxSoC/new_soc
```

**解决方法：**
1. 使用 `Makefile.iverilog`（已修复路径问题）
2. 或使用项目主仿真脚本

### SystemVerilog 支持

Icarus Verilog 对 SystemVerilog 的支持有限：
- ✅ 支持：基本语法、interface、always_comb
- ⚠️ 部分支持：class、constraint
- ❌ 不支持：某些高级特性

如果遇到语法错误，可能需要：
1. 简化 SystemVerilog 代码
2. 使用 Verilator（更好的 SV 支持）
3. 使用商业工具（VCS, Questa）

## 🔍 故障排除

### 问题 1: 找不到文件

```
Error: Cannot find file: xxx.v
```

**解决：**
- 检查 filelist 中的路径是否正确
- 确保所有源文件都存在
- 使用 `make -f Makefile.iverilog list-files` 查看文件列表

### 问题 2: 语法错误

```
Error: syntax error
```

**解决：**
- 检查是否使用了 Icarus Verilog 不支持的 SystemVerilog 特性
- 尝试使用 `-g2012` 标志启用更多 SV 特性
- 考虑使用 Verilator

### 问题 3: 没有波形输出

```
Warning: No waveform file generated
```

**解决：**
- 确保 testbench 中有 `$dumpfile` 和 `$dumpvars`
- 检查仿真是否正常完成
- 查看 sim.log 日志

## 📚 相关文档

- [Icarus Verilog 官方文档](http://iverilog.icarus.com/)
- [GTKWave 用户手册](http://gtkwave.sourceforge.net/)
- [项目综合指南](../README.md)
- [后综合仿真脚本](../run_post_syn_sim.py)

## 💡 最佳实践

1. **优先使用项目主脚本**
   ```bash
   cd chisel/synthesis
   python run_post_syn_sim.py --simulator iverilog --netlist ics55
   ```

2. **如果需要自定义仿真**
   ```bash
   cd chisel/synthesis/iverilog/run
   make -f Makefile.iverilog sim
   ```

3. **查看波形**
   ```bash
   # 使用 GTKWave
   gtkwave soc_tb.vcd
   
   # 或使用项目的 Web 波形查看器
   cd chisel/synthesis/waves
   ./start_wave_viewer.sh
   ```

## ✅ 总结

- ❌ 原始 Makefile 需要 VCS（商业软件）
- ✅ 使用 `Makefile.iverilog` 支持开源工具
- ✅ 或使用项目主仿真脚本（最推荐）
- ✅ Icarus Verilog 完全免费且功能足够

---

**更新日期**: 2025-11-21  
**工具**: Icarus Verilog + GTKWave  
**状态**: ✅ 可用
