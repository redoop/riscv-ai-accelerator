# 综合脚本修改说明

## 修改日期
2025-12-03

## 修改内容

### 1. `run_synthesis.sh` - 综合脚本重构

#### 主要改进

**原始功能:**
- 仅综合单个 SimpleEdgeAiSoC 模块
- 需要手动生成 Chisel RTL
- 不包含网表仿真

**新增功能:**
1. **自动化 Chisel RTL 生成**
   - 自动检测 RTL 是否存在
   - 如果不存在，自动调用 sbt 生成
   - 生成后复制到 `project/verilog/` 目录

2. **完整 ECOS ASIC 顶层综合**
   - 综合目标从单个 IP 改为完整 ASIC 顶层 (`asic_top`)
   - 包含所有支持模块：IO PAD、时钟、复位等
   - 使用 filelist 管理文件依赖

3. **自动网表仿真**
   - 综合完成后自动运行网表仿真
   - 使用 Icarus Verilog 验证功能
   - 生成波形文件供分析

4. **改进的文件组织**
   - 网表输出到 `project/netlist/`
   - RTL 复制到 `project/verilog/`
   - 符合 ECOS 项目结构

#### 详细变更

**路径配置:**
```bash
# 旧配置
RTL_FILE="../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv"
OUTPUT_DIR="netlist"
NETLIST_FILE="$OUTPUT_DIR/SimpleEdgeAiSoC_ics55.v"

# 新配置
CHISEL_RTL_SRC="../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv"
PROJECT_VERILOG_DIR="$SCRIPT_DIR/project/verilog"
CHISEL_RTL_DEST="$PROJECT_VERILOG_DIR/SimpleEdgeAiSoC.sv"
OUTPUT_DIR="$SCRIPT_DIR/project/netlist"
NETLIST_FILE="$OUTPUT_DIR/asic_top_ics55.v"
```

**综合目标:**
```bash
# 旧目标
hierarchy -top ip1_SimpleEdgeAiSoC

# 新目标
hierarchy -top asic_top
```

**文件收集:**
```bash
# 新增：从 filelist 收集所有文件
for flist in asic_top.f ip.f lib.f soc.f; do
    # 读取并处理文件列表
    # 替换 $RTL_PATH 变量
    # 检查文件存在性
done
```

**流程步骤:**
1. 生成 Chisel RTL
2. 复制 RTL 到项目目录
3. 检查工具和 PDK
4. 综合 ASIC 顶层
5. 检查综合结果
6. 运行网表仿真

### 2. `Makefile.iverilog` - 路径配置修正

#### 问题分析

原始 Makefile 中的路径配置与更新后的综合脚本不一致，导致网表仿真无法找到正确的文件。

#### 修正内容

**PDK 路径修正:**
```makefile
# 修正前
PDK_ROOT := ../../pdk/icsprout55-pdk

# 修正后
PDK_ROOT := ../pdk/icsprout55-pdk
```

**网表路径修正:**
```makefile
# 修正前
NETLIST_DIR := ../../netlist
NETLIST_FILE := $(NETLIST_DIR)/SimpleEdgeAiSoC_ics55.v

# 修正后
NETLIST_DIR := ../project/netlist
NETLIST_FILE := $(NETLIST_DIR)/asic_top_ics55.v
```

**Chisel RTL 路径修正:**
```makefile
# 修正前
CHISEL_RTL := ../../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 修正后
CHISEL_RTL := ../project/verilog/SimpleEdgeAiSoC.sv
```

#### 修正原因

1. **路径层级错误**: 从 `run/` 目录到 `ecos/` 只需要 `../` 而不是 `../../`
2. **输出目录不一致**: 网表输出到 `project/netlist/` 而不是 `netlist/`
3. **文件名不匹配**: 综合生成的是 `asic_top_ics55.v` 而不是 `SimpleEdgeAiSoC_ics55.v`
4. **RTL 源位置**: 应该使用 `project/verilog/` 中的副本，保持项目自包含

### 3. `asic_top.sv` - IP 选择配置

#### 修改内容

**IP 定义更新:**
```systemverilog
// 旧配置
//`define ip_1 3'd1 // project_1884
`define ip_3 3'd3   // ysyxSoCASIC

// 新配置
`define ip_1 3'd1 // SimpleEdgeAiSoC (RISC-V AI Accelerator)
//`define ip_3 3'd3   // ysyxSoCASIC
```

**原因:**
- SimpleEdgeAiSoC 应该使用 ip_1 槽位
- 与 filelist/ip.f 中的配置一致
- 避免与其他项目冲突

### 4. 新增文档

#### `PATH_VERIFICATION.md`
- 详细的路径配置验证文档
- 目录结构说明
- 路径关系图
- 验证命令和测试方法
- 修正前后对比

#### `SYNTHESIS_GUIDE.md`
- 完整的综合流程说明
- 文件结构说明
- ASIC 顶层配置详解
- IO 分配表
- 故障排除指南

#### `CHANGES.md` (本文件)
- 详细的修改记录
- 变更原因说明
- 使用示例

## 使用示例

### 基本使用

```bash
cd chisel/synthesis/ecos
./run_synthesis.sh
```

脚本会自动完成：
1. 生成 Chisel RTL (如果需要)
2. 综合 ASIC 顶层
3. 运行网表仿真
4. 生成报告和波形

### 查看结果

```bash
# 综合统计
cat project/netlist/synthesis_stats.txt

# 综合日志
cat project/netlist/synthesis.log

# 仿真日志
cat run/sim_netlist.log

# 查看波形
cd run
make -f Makefile.iverilog netlist-wave
```

### 单独运行各步骤

```bash
# 1. 仅生成 Chisel RTL
cd chisel
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 2. 仅综合 (需要先有 RTL)
cd synthesis/ecos
# 修改脚本，注释掉仿真部分

# 3. 仅运行网表仿真
cd synthesis/ecos/run
make -f Makefile.iverilog netlist
```

## 兼容性

### 保持兼容
- 所有原有文件路径保持不变
- PDK 配置不变
- 仿真脚本不变

### 新增依赖
- 需要 sbt (Chisel 构建工具)
- 需要 Yosys (综合工具)
- 需要 Icarus Verilog (仿真工具)

## 测试建议

### 测试步骤

1. **清理环境**
   ```bash
   cd chisel
   sbt clean
   rm -rf generated/
   cd synthesis/ecos
   rm -rf project/verilog/* project/netlist/*
   ```

2. **运行完整流程**
   ```bash
   ./run_synthesis.sh
   ```

3. **验证输出**
   - 检查 `project/verilog/SimpleEdgeAiSoC.sv` 存在
   - 检查 `project/netlist/asic_top_ics55.v` 存在
   - 检查仿真日志无错误
   - 检查波形文件生成

### 预期结果

```
✓ Chisel RTL 生成成功
✓ RTL 复制到 project/verilog/
✓ 综合成功，生成网表
✓ 网表仿真通过
✓ 波形文件生成
```

## 已知问题

### 潜在问题

1. **Chisel 编译时间**
   - 首次编译可能需要较长时间
   - 建议预先编译 Chisel 项目

2. **综合时间**
   - ASIC 顶层综合比单个 IP 慢
   - 预计 5-15 分钟（取决于机器性能）

3. **仿真时间**
   - 网表仿真比 RTL 仿真慢
   - 可能需要调整仿真时长

### 解决方案

1. **加速 Chisel 编译**
   ```bash
   cd chisel
   sbt compile  # 预编译
   ```

2. **跳过仿真**
   - 修改脚本，注释掉步骤 6
   - 手动运行仿真

3. **调整仿真参数**
   - 修改 `run/Makefile.iverilog`
   - 调整测试向量

## 后续改进

### 计划改进

1. **并行化**
   - 支持多核并行综合
   - 加速编译和仿真

2. **增量综合**
   - 检测文件变化
   - 仅重新综合修改的模块

3. **报告生成**
   - 自动生成 HTML 报告
   - 包含面积、时序、功耗分析

4. **CI/CD 集成**
   - 添加自动化测试
   - 集成到持续集成流程

## 参考

- [SYNTHESIS_GUIDE.md](./SYNTHESIS_GUIDE.md) - 综合指南
- [IVERILOG_USAGE.md](./IVERILOG_USAGE.md) - 仿真指南
- [run/README_NETLIST_SIM.md](./run/README_NETLIST_SIM.md) - 网表仿真说明
