# ECOS 项目路径验证文档

## 目录结构

```
chisel/synthesis/ecos/
├── run_synthesis.sh          # 综合脚本 (工作目录: ecos/)
├── asic_top.sv               # ASIC 顶层
├── filelist/                 # 文件列表
│   ├── asic_top.f
│   ├── ip.f
│   ├── lib.f
│   └── soc.f
├── project/                  # 项目输出目录
│   ├── verilog/              # Chisel RTL 复制目标
│   │   └── SimpleEdgeAiSoC.sv
│   └── netlist/              # 综合网表输出
│       ├── asic_top_ics55.v
│       └── ics55_LLSC_H7CL.v
├── pdk/                      # PDK 目录
│   └── icsprout55-pdk/
├── run/                      # 仿真运行目录
│   ├── Makefile.iverilog     # Icarus Verilog Makefile
│   └── run_sim.py            # Python 仿真脚本
├── tb/                       # 测试平台
├── utils/                    # 工具模块
├── rcu/                      # 复位时钟单元
└── lib/                      # IO PAD 库
```

## 路径配置验证

### 1. run_synthesis.sh (工作目录: ecos/)

| 变量 | 路径 | 绝对路径 | 状态 |
|------|------|----------|------|
| `SCRIPT_DIR` | `.` | `/opt/.../chisel/synthesis/ecos` | ✓ |
| `CHISEL_RTL_SRC` | `../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` | `/opt/.../chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` | ✓ |
| `PROJECT_VERILOG_DIR` | `$SCRIPT_DIR/project/verilog` | `/opt/.../chisel/synthesis/ecos/project/verilog` | ✓ |
| `CHISEL_RTL_DEST` | `$PROJECT_VERILOG_DIR/SimpleEdgeAiSoC.sv` | `/opt/.../chisel/synthesis/ecos/project/verilog/SimpleEdgeAiSoC.sv` | ✓ |
| `OUTPUT_DIR` | `$SCRIPT_DIR/project/netlist` | `/opt/.../chisel/synthesis/ecos/project/netlist` | ✓ |
| `NETLIST_FILE` | `$OUTPUT_DIR/asic_top_ics55.v` | `/opt/.../chisel/synthesis/ecos/project/netlist/asic_top_ics55.v` | ✓ |
| `PDK_ROOT` | `$SCRIPT_DIR/pdk/icsprout55-pdk` | `/opt/.../chisel/synthesis/ecos/pdk/icsprout55-pdk` | ✓ |

### 2. Makefile.iverilog (工作目录: ecos/run/)

**修正前的问题：**
- ❌ `NETLIST_DIR := ../../netlist` (错误，应该是 `../project/netlist`)
- ❌ `NETLIST_FILE := $(NETLIST_DIR)/SimpleEdgeAiSoC_ics55.v` (错误，应该是 `asic_top_ics55.v`)
- ❌ `PDK_ROOT := ../../pdk/icsprout55-pdk` (错误，应该是 `../pdk/icsprout55-pdk`)
- ❌ `CHISEL_RTL := ../../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` (不一致，应该用 project/verilog 中的副本)

**修正后的配置：**

| 变量 | 路径 | 绝对路径 | 状态 |
|------|------|----------|------|
| `PDK_ROOT` | `../pdk/icsprout55-pdk` | `/opt/.../chisel/synthesis/ecos/pdk/icsprout55-pdk` | ✓ 已修正 |
| `PDK_VERILOG` | `$(PDK_ROOT)/IP/STD_cell/.../ics55_LLSC_H7CL.v` | `/opt/.../chisel/synthesis/ecos/pdk/.../ics55_LLSC_H7CL.v` | ✓ 已修正 |
| `NETLIST_DIR` | `../project/netlist` | `/opt/.../chisel/synthesis/ecos/project/netlist` | ✓ 已修正 |
| `NETLIST_FILE` | `$(NETLIST_DIR)/asic_top_ics55.v` | `/opt/.../chisel/synthesis/ecos/project/netlist/asic_top_ics55.v` | ✓ 已修正 |
| `CHISEL_RTL` | `../project/verilog/SimpleEdgeAiSoC.sv` | `/opt/.../chisel/synthesis/ecos/project/verilog/SimpleEdgeAiSoC.sv` | ✓ 已修正 |

### 3. filelist/ip.f (引用路径)

**文件内容：**
```
$RTL_PATH/project/verilog/SimpleEdgeAiSoC.sv
```

**路径解析：**
- `$RTL_PATH` 在 `run_synthesis.sh` 中设置为 `$SCRIPT_DIR` (即 `ecos/`)
- 完整路径: `ecos/project/verilog/SimpleEdgeAiSoC.sv` ✓

## 路径关系图

```
从 run/ 目录看：
run/
├── ../ (ecos/)
│   ├── project/
│   │   ├── verilog/
│   │   │   └── SimpleEdgeAiSoC.sv    ← ../project/verilog/SimpleEdgeAiSoC.sv
│   │   └── netlist/
│   │       └── asic_top_ics55.v      ← ../project/netlist/asic_top_ics55.v
│   ├── pdk/
│   │   └── icsprout55-pdk/           ← ../pdk/icsprout55-pdk/
│   ├── filelist/
│   ├── tb/
│   └── ...
└── ../../ (chisel/)
    └── generated/
        └── simple_edgeaisoc/
            └── SimpleEdgeAiSoC.sv    ← ../../generated/... (原始生成位置)
```

## 验证命令

### 从 ecos/ 目录验证

```bash
cd chisel/synthesis/ecos

# 验证 Chisel RTL 源路径
ls -l ../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 验证项目目录
ls -ld project/verilog project/netlist

# 验证 PDK 路径
ls -ld pdk/icsprout55-pdk
```

### 从 run/ 目录验证

```bash
cd chisel/synthesis/ecos/run

# 验证网表路径
ls -l ../project/netlist/asic_top_ics55.v

# 验证 PDK 路径
ls -l ../pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v

# 验证 Chisel RTL 副本
ls -l ../project/verilog/SimpleEdgeAiSoC.sv

# 使用 Makefile 验证配置
make -f Makefile.iverilog config
```

## 修正总结

### 修正的文件

1. **Makefile.iverilog**
   - ✓ 修正 `PDK_ROOT` 从 `../../pdk` 到 `../pdk`
   - ✓ 修正 `NETLIST_DIR` 从 `../../netlist` 到 `../project/netlist`
   - ✓ 修正 `NETLIST_FILE` 从 `SimpleEdgeAiSoC_ics55.v` 到 `asic_top_ics55.v`
   - ✓ 修正 `CHISEL_RTL` 从 `../../../generated/...` 到 `../project/verilog/SimpleEdgeAiSoC.sv`

### 为什么需要修正

1. **一致性**: 所有路径应该指向 `run_synthesis.sh` 生成的输出位置
2. **可维护性**: 使用统一的项目结构，避免混淆
3. **正确性**: 网表文件名应该反映实际的顶层模块名 (`asic_top`)
4. **独立性**: 仿真应该使用 `project/verilog/` 中的 RTL 副本，而不是原始生成位置

## 测试验证

### 完整流程测试

```bash
# 1. 清理环境
cd chisel/synthesis/ecos
rm -rf project/verilog/* project/netlist/*

# 2. 运行综合脚本
./run_synthesis.sh

# 3. 验证输出
ls -l project/verilog/SimpleEdgeAiSoC.sv
ls -l project/netlist/asic_top_ics55.v

# 4. 单独运行网表仿真
cd run
make -f Makefile.iverilog check-netlist
make -f Makefile.iverilog netlist
```

### 预期结果

```
✓ project/verilog/SimpleEdgeAiSoC.sv 存在
✓ project/netlist/asic_top_ics55.v 存在
✓ project/netlist/ics55_LLSC_H7CL.v 存在
✓ Makefile 能正确找到所有文件
✓ 网表仿真成功运行
```

## 注意事项

1. **相对路径**: 所有路径都是相对于各自脚本的工作目录
2. **环境变量**: `$RTL_PATH` 在 `run_synthesis.sh` 中设置，供 filelist 使用
3. **文件复制**: Chisel RTL 会被复制到 `project/verilog/`，这是设计决策，确保项目自包含
4. **网表命名**: 网表文件名 `asic_top_ics55.v` 反映了顶层模块名和目标 PDK

## 相关文档

- [SYNTHESIS_GUIDE.md](./SYNTHESIS_GUIDE.md) - 综合流程指南
- [CHANGES.md](./CHANGES.md) - 修改记录
- [IVERILOG_USAGE.md](./IVERILOG_USAGE.md) - Icarus Verilog 使用说明
