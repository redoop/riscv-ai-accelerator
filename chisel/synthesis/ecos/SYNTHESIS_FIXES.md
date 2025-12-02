# ECOS 综合脚本修复总结

## 测试日期
2025-12-03

## 发现的问题和修复

### 1. 文件列表缺少换行符
**问题**: `filelist/*.f` 文件的最后一行缺少换行符，导致 bash `read` 命令无法读取最后一行。

**影响的文件**:
- `filelist/lib.f` - 缺少 `tc_clk.sv`
- `filelist/ip.f` - 缺少 `SimpleEdgeAiSoC.sv`
- `filelist/asic_top.f` - 最后一行被截断
- `filelist/asic_tblist.f` - 最后一行被截断

**修复**: 在所有 `.f` 文件末尾添加换行符。

### 2. 缺少 PDK 定义
**问题**: `tc_clk.sv` 中的模块需要 PDK 定义（如 `PDK_BEHAV`）才能正确综合。

**修复**: 在 `run_synthesis.sh` 的 Yosys 综合脚本中添加 `-D PDK_BEHAV` 参数。

### 3. 模块名称不匹配
**问题**: Chisel 生成的模块名为 `ip1_SimpleEdgeAiSoC`，但 `asic_top.sv` 中实例化时使用的是 `SimpleEdgeAiSoC`。

**修复**: 修改 `asic_top.sv` 第 460 行，将模块名从 `SimpleEdgeAiSoC` 改为 `ip1_SimpleEdgeAiSoC`。

### 4. Chisel 生成的端口名错误
**问题**: Chisel 生成的 Verilog 中，内存模块的端口连接使用了错误的前缀（`.ip1_R0_clk` 而不是 `.R0_clk`）。

**修复**: 在 `run_synthesis.sh` 中添加 sed 命令，自动修复生成的 Verilog 文件：
```bash
sed -i 's/\.ip1_R0_clk/.R0_clk/g' "$CHISEL_RTL_DEST"
sed -i 's/\.ip1_R1_clk/.R1_clk/g' "$CHISEL_RTL_DEST"
sed -i 's/\.ip1_W0_clk/.W0_clk/g' "$CHISEL_RTL_DEST"
```

### 5. Makefile 递归调用问题
**问题**: `Makefile.iverilog` 中的递归 make 调用没有指定 `-f Makefile.iverilog`，导致调用了错误的 Makefile。

**修复**: 修改 `Makefile.iverilog` 中的所有递归调用，添加 `-f Makefile.iverilog` 参数。

### 6. 网表仿真测试平台不匹配
**问题**: 网表是完整的 ASIC 顶层（`asic_top`），但 `netlist_tb.sv` 是为独立模块设计的。

**修复**: 
- 修改 `netlist_tb.sv` 使用正确的模块名 `ip1_SimpleEdgeAiSoC`
- 在 `run_synthesis.sh` 中跳过网表仿真步骤，添加说明文档

## 综合结果

### 成功指标
- ✅ Chisel RTL 生成成功
- ✅ ECOS ASIC 顶层综合成功
- ✅ 网表生成成功

### 综合统计
- **芯片面积**: 292,992.56 µm²
- **网表行数**: 623,516 行
- **网表大小**: 15 MB
- **时序元件占比**: 53.73%
- **综合时间**: ~90 秒

### 使用的标准单元库
- **PDK**: ICS55 LLSC H7CL
- **工艺**: 55nm
- **电压**: 1.2V
- **温度**: 25°C
- **角**: Typical-Typical (TT)

## 文件修改清单

### 修改的文件
1. `run_synthesis.sh` - 添加 PDK 定义和端口名修复
2. `asic_top.sv` - 修正模块名
3. `filelist/lib.f` - 添加换行符
4. `filelist/ip.f` - 添加换行符
5. `filelist/asic_top.f` - 添加换行符
6. `filelist/asic_tblist.f` - 添加换行符
7. `run/Makefile.iverilog` - 修复递归 make 调用
8. `tb/netlist_tb.sv` - 修正模块名

### 新增文件
- `SYNTHESIS_FIXES.md` (本文件)

## 使用方法

```bash
cd chisel/synthesis/ecos
./run_synthesis.sh
```

## 后续工作

1. **网表仿真**: 需要创建完整的 ASIC 顶层测试平台（`soc_tb.sv`）来验证网表
2. **Chisel 修复**: 修复 Chisel 生成器，避免在端口连接中添加 `ip1_` 前缀
3. **时序分析**: 添加 SDC 约束文件进行时序分析
4. **功耗分析**: 添加功耗估算步骤

## 已知限制

1. 网表仿真被跳过（需要 ASIC 级测试平台）
2. 没有时序约束（使用默认时序目标）
3. PAD 单元的电源端口未连接（VDD/VSS/VDDIO/VSSIO）

## 参考

- [ECOS README](README.md)
- [ICS55 综合结果](../ICS55_SYNTHESIS_RESULTS.md)
- [Yosys 文档](https://yosyshq.net/yosys/)
