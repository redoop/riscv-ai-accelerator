# ECOS 项目综合指南

## 概述

本指南说明如何使用 `run_synthesis.sh` 脚本完成从 Chisel RTL 生成到网表综合和仿真的完整流程。

## 综合流程

### 自动化流程

`run_synthesis.sh` 脚本自动完成以下步骤：

1. **生成 Chisel RTL**
   - 调用 `sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"`
   - 生成 `SimpleEdgeAiSoC.sv` 文件

2. **复制 RTL 到项目目录**
   - 从 `chisel/generated/simple_edgeaisoc/` 复制到 `ecos/project/verilog/`
   - 供后续综合使用

3. **综合 ASIC 顶层**
   - 使用 Yosys 综合工具
   - 目标 PDK: ICS55 LLSC H7CL
   - 顶层模块: `asic_top`
   - 包含所有支持文件（IO PAD、时钟、复位等）

4. **生成网表**
   - 输出到 `ecos/project/netlist/asic_top_ics55.v`
   - 包含综合统计和日志

5. **运行网表仿真**
   - 使用 Icarus Verilog
   - 验证综合后的网表功能正确性

### 使用方法

```bash
cd chisel/synthesis/ecos
./run_synthesis.sh
```

## 文件结构

### 输入文件

```
ecos/
├── asic_top.sv              # ASIC 顶层模块
├── filelist/
│   ├── asic_top.f          # 顶层和工具模块列表
│   ├── ip.f                # IP 核列表 (SimpleEdgeAiSoC)
│   ├── lib.f               # 库文件列表 (IO PAD, 时钟)
│   └── soc.f               # SoC 模块列表
├── utils/                   # 工具模块 (时钟分频、复位同步等)
├── rcu/                     # 复位和时钟单元
└── lib/                     # IO PAD 库
```

### 输出文件

```
ecos/project/
├── verilog/
│   └── SimpleEdgeAiSoC.sv  # Chisel 生成的 RTL (复制)
└── netlist/
    ├── asic_top_ics55.v    # 综合后的网表
    ├── ics55_LLSC_H7CL.v   # PDK Verilog 模型
    ├── synthesis_stats.txt  # 综合统计
    └── synthesis.log        # 综合日志
```

## ASIC 顶层配置

### IP 选择

`asic_top.sv` 支持多个 IP 核，通过 `ip_sel_pad[2:0]` 选择：

- `ip_0` (3'd0): project_1854 (未启用)
- **`ip_1` (3'd1): SimpleEdgeAiSoC** ✓ 当前启用
- `ip_2` (3'd2): project_1839 (未启用)
- `ip_3` (3'd3): ysyxSoCASIC (未启用)
- `ip_4` (3'd4): project_1988 (未启用)
- `ip_5` (3'd5): project_1993 (未启用)

### 时钟和复位

- **系统时钟**: `sys_clk_i_pad` (100MHz 输入)
- **复位信号**: `rst_n_pad` (低电平有效)
- **时钟输出**: `sys_clk_o_pad` (时钟回环输出)

内部时钟分频：
- `clk_100m`: 100MHz (主时钟)
- `clk_50m`: 50MHz
- `clk_25m`: 25MHz

### IO 分配 (ip_1 - SimpleEdgeAiSoC)

根据 `asic_top.sv` 中的配置：

```systemverilog
io_pad[31:0]   -> io_gpio_in[31:0]   (输入)
io_pad[63:32]  -> io_gpio_out[31:0]  (输出)
io_pad[64]     -> io_uart_tx         (输出)
io_pad[65]     -> io_uart_rx         (输入)
io_pad[66]     -> io_uart_tx_irq     (输出)
io_pad[67]     -> io_uart_rx_irq     (输出)
io_pad[68]     -> io_lcd_spi_clk     (输出)
io_pad[69]     -> io_lcd_spi_mosi    (输出)
io_pad[70]     -> io_lcd_spi_cs      (输出)
io_pad[71]     -> io_lcd_spi_dc      (输出)
io_pad[72]     -> io_lcd_spi_rst     (输出)
io_pad[73]     -> io_lcd_backlight   (输出)
io_pad[74]     -> io_trap            (输出)
io_pad[75]     -> io_compact_irq     (输出)
io_pad[76]     -> io_bitnet_irq      (输出)
```

## 网表仿真

### 自动仿真

脚本会自动运行网表仿真，使用：

```bash
make -f Makefile.iverilog netlist
```

### 手动仿真

如果需要单独运行仿真：

```bash
cd ecos/run

# 方法 1: 使用 Makefile
make -f Makefile.iverilog netlist

# 方法 2: 使用 Python 脚本
python run_sim.py --simulator iverilog --netlist

# 查看波形
make -f Makefile.iverilog netlist-wave
# 或
python run_sim.py --wave --mode netlist
```

## 综合统计

综合完成后，查看统计信息：

```bash
cat ecos/project/netlist/synthesis_stats.txt
```

统计包括：
- 使用的标准单元数量
- 面积估算
- 时序信息
- 功耗估算

## 故障排除

### Chisel RTL 生成失败

```bash
cd chisel
sbt clean
sbt compile
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
```

### 综合失败

检查日志：
```bash
cat ecos/project/netlist/synthesis.log
```

常见问题：
- PDK 未安装或路径错误
- Yosys 版本不兼容
- RTL 文件缺失或语法错误

### 网表仿真失败

检查日志：
```bash
cat ecos/run/compile_netlist.log
cat ecos/run/sim_netlist.log
```

常见问题：
- PDK Verilog 模型缺失
- 测试平台配置错误
- 时序违例

## 下一步

1. **查看综合结果**
   ```bash
   cat ecos/project/netlist/synthesis_stats.txt
   ```

2. **分析波形**
   ```bash
   cd ecos/run
   make -f Makefile.iverilog netlist-wave
   ```

3. **优化设计**
   - 根据综合报告调整时序约束
   - 优化关键路径
   - 减少面积或功耗

4. **物理设计**
   - 布局规划
   - 布线
   - 时序收敛

## 参考文档

- [Chisel 文档](../../../README.md)
- [ECOS 仿真指南](./IVERILOG_USAGE.md)
- [网表仿真说明](./run/README_NETLIST_SIM.md)
- [ICS55 PDK 文档](./pdk/README.md)
