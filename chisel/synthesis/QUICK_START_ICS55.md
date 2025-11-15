# ICS55 PDK 快速开始指南

本指南帮助你快速使用 ICS55 PDK 进行逻辑综合和后端仿真。

## 🎯 目标

使用 ICS55 PDK 对 SimpleEdgeAiSoC 进行逻辑综合，并验证综合后的网表功能正确性。

## 📋 前提条件

- ✅ 已安装 Yosys（OSS CAD Suite）
- ✅ 已安装 Icarus Verilog
- ✅ 已生成 Chisel RTL
- ✅ 有 GitHub 访问权限（用于克隆 PDK）

## 🚀 5 步完成综合和仿真

### 步骤 1: 安装 ICS55 PDK

```bash
cd chisel/synthesis/pdk
python get_ics55_pdk.py
```

**预期输出:**
```
Cloning into 'icsprout55-pdk'...
...
```

**验证安装:**
```bash
# 检查 Liberty 文件
ls -la icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 检查 Verilog 模型
ls -la icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v
```

### 步骤 2: 生成 RTL（如果还没有）

```bash
cd ../..  # 回到 chisel 目录
make verilog
```

**预期输出:**
```
[success] Total time: XX s
```

**验证 RTL:**
```bash
ls -la generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

### 步骤 3: 运行逻辑综合

```bash
cd synthesis
./run_ics55_synthesis.sh
```

**预期输出:**
```
==========================================
ICS55 PDK 逻辑综合
==========================================
PDK: ICS55
Liberty: .../ics55_stdcell_typ.lib
Verilog: .../ics55_stdcell.v
RTL: ../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
输出: netlist/SimpleEdgeAiSoC_ics55.v

运行 Yosys 综合...
...
✓ 综合成功！
网表文件: netlist/SimpleEdgeAiSoC_ics55.v
```

**验证网表:**
```bash
ls -la netlist/SimpleEdgeAiSoC_ics55.v
ls -la netlist/ics55_LLSC_H7CL.v
```

### 步骤 4: 查看综合统计

```bash
cat netlist/synthesis_stats_ics55.txt
```

**示例输出:**
```
=== SimpleEdgeAiSoC ===

   Number of wires:               XXXX
   Number of wire bits:           XXXX
   Number of cells:               XXXX
     ICS55_AND2X1                 XXX
     ICS55_NAND2X1                XXX
     ICS55_DFFX1                  XXX
     ...
```

### 步骤 5: 运行后端仿真

```bash
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

**预期输出:**
```
============================================================
逻辑综合后网表仿真
============================================================
设计: SimpleEdgeAiSoC
时间: 2024-XX-XX XX:XX:XX
============================================================

使用 Icarus Verilog 进行仿真...
------------------------------------------------------------
使用 ICS55 PDK 网表
✓ 找到 Icarus Verilog: /opt/tools/oss-cad/oss-cad-suite/bin/iverilog
1. 编译...
  包含标准单元库: .../ics55_stdcell.v
✓ 编译成功
2. 运行仿真...
[仿真输出]
✓ 仿真成功
```

## ✅ 验证结果

### 检查综合结果

```bash
# 查看网表文件大小
wc -l netlist/SimpleEdgeAiSoC_ics55.v

# 查看使用的标准单元类型
grep "ICS55_" netlist/SimpleEdgeAiSoC_ics55.v | cut -d' ' -f1 | sort | uniq -c

# 查看综合日志
less netlist/synthesis_ics55.log
```

### 检查仿真结果

仿真输出应该显示：
- ✓ 测试通过
- ✓ 无错误信息
- ✓ 功能验证成功

## 🔧 使用 Makefile（可选）

如果你喜欢使用 Makefile：

```bash
# 综合
make synth_ics55

# 仿真
make sim_ics55

# 查看信息
make info
```

## 📊 对比不同 PDK

你可以对比不同 PDK 的综合结果：

```bash
# IHP SG13G2 (130nm)
./run_ihp_synthesis.sh
cat netlist/synthesis_stats_ihp.txt

# ICS55 (55nm)
./run_ics55_synthesis.sh
cat netlist/synthesis_stats_ics55.txt

# 通用综合
./run_generic_synthesis.sh
cat netlist/synthesis_stats.txt
```

## 🐛 常见问题

### 问题 1: PDK 克隆失败

**错误:**
```
Permission denied (publickey)
```

**解决:**
1. 检查 SSH 密钥配置
2. 或使用 HTTPS 克隆：
   ```bash
   git clone --recursive https://github.com/IDE-Platform/icsprout55-pdk.git
   ```

### 问题 2: RTL 文件未找到

**错误:**
```
错误: 未找到 RTL 文件
```

**解决:**
```bash
cd chisel
make verilog
```

### 问题 3: Yosys 未找到

**错误:**
```
/opt/tools/oss-cad/oss-cad-suite/bin/yosys: No such file or directory
```

**解决:**
1. 安装 OSS CAD Suite
2. 或修改 `run_ics55_synthesis.sh` 中的 `YOSYS_BIN` 路径

### 问题 4: 综合失败

**解决步骤:**
1. 查看日志：
   ```bash
   cat netlist/synthesis_ics55.log
   ```
2. 检查 PDK 文件完整性：
   ```bash
   ls -la pdk/icsprout55-pdk/lib/
   ls -la pdk/icsprout55-pdk/verilog/
   ```
3. 尝试通用综合验证 RTL：
   ```bash
   ./run_generic_synthesis.sh
   ```

### 问题 5: 仿真失败

**解决步骤:**
1. 确认网表存在：
   ```bash
   ls -la netlist/SimpleEdgeAiSoC_ics55.v
   ```
2. 确认标准单元库存在：
   ```bash
   ls -la netlist/ics55_LLSC_H7CL.v
   ```
3. 检查 Icarus Verilog：
   ```bash
   iverilog -v
   ```

## 📈 下一步

完成基本流程后，你可以：

1. **查看详细文档**: [ICS55_PDK_GUIDE.md](ICS55_PDK_GUIDE.md)
2. **对比 IHP PDK**: [IHP_PDK_GUIDE.md](IHP_PDK_GUIDE.md)
3. **自定义综合参数**: 编辑 `run_ics55_synthesis.sh`
4. **添加测试用例**: 修改 `testbench/post_syn_tb.sv`
5. **查看波形**: 使用 GTKWave 或 Verdi

## 🎓 学习资源

### ICS55 PDK
- GitHub: https://github.com/IDE-Platform/icsprout55-pdk
- 文档: 查看 PDK 仓库中的文档

### Yosys
- 官网: https://yosyshq.net/yosys/
- 教程: https://yosyshq.readthedocs.io/

### Icarus Verilog
- 官网: http://iverilog.icarus.com/
- Wiki: http://iverilog.wikia.com/

## 💡 提示

1. **首次使用**: 建议先使用通用综合验证 RTL 正确性
2. **PDK 选择**: ICS55 (55nm) 比 IHP (130nm) 更先进，但可能需要更多调试
3. **仿真时间**: 后端仿真比 RTL 仿真慢，这是正常的
4. **日志文件**: 保存综合和仿真日志以便调试

## 📞 获取帮助

```bash
# 查看脚本帮助
python run_post_syn_sim.py --help

# 查看 Makefile 帮助
make help

# 查看系统信息
make info
```

---

**快速命令总结:**

```bash
# 完整流程（一次性执行）
cd chisel/synthesis/pdk && python get_ics55_pdk.py && cd .. && \
./run_ics55_synthesis.sh && \
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# 或使用 Makefile
make synth_ics55 && make sim_ics55
```

祝你综合顺利！🎉
