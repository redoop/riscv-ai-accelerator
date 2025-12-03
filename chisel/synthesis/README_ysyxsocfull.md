# ysyxSoCFull 逻辑综合指南

## 概述

本目录包含 ysyxSoCFull 的逻辑综合脚本，用于将 RTL 设计映射到 ICS55 PDK 标准单元库。

## 设计信息

- **顶层模块**: `ysyxSoCFull`
- **时钟信号**: `clock`
- **复位信号**: `reset`
- **源文件**: `../ecos/ysyxSoC/build/ysyxSoCFull.v`

### 包含的主要模块

1. **处理器核心**
   - CPU (RISC-V)

2. **总线互连**
   - AXI4Xbar (AXI4 交叉开关)
   - AXI4Buffer (AXI4 缓冲)
   - AXI4Fragmenter (AXI4 分片器)
   - AXI4UserYanker (AXI4 用户信号处理)
   - APBFanout (APB 扇出)
   - AXI4ToAPB (总线桥接)

3. **存储器**
   - AXI4MROM (Mask ROM)
   - AXI4RAM (SRAM)
   - APBSDRAM (SDRAM 控制器)

4. **外设控制器**
   - APBUart16550 (UART)
   - APBGPIO (GPIO)
   - APBKeyboard (PS/2 键盘)
   - APBVGA (VGA 显示)
   - APBSPI (SPI/Flash)
   - APBPSRAM (PSRAM)

5. **仿真模型**
   - flash (Flash 存储器模型)
   - psram (PSRAM 模型)
   - sdram (SDRAM 模型)
   - bitrev (位反转测试模块)

## 前置条件

### 1. 生成 RTL

首先需要生成 ysyxSoCFull.v：

```bash
cd ../ecos/ysyxSoC
make verilog
```

这将在 `ecos/ysyxSoC/build/ysyxSoCFull.v` 生成完整的 Verilog 文件。

### 2. 安装 PDK

确保已安装 ICS55 PDK：

```bash
python pdk/get_ics55_pdk.py
```

### 3. 安装工具

需要安装 OSS CAD Suite (包含 Yosys):

```bash
# 确认 Yosys 可用
/opt/tools/oss-cad/oss-cad-suite/bin/yosys --version
```

## 使用方法

### 运行综合

```bash
cd chisel/synthesis
./run_ysyxsocfull_synthesis.sh
```

### 输出文件

综合成功后，将在 `netlist/` 目录生成以下文件：

- `ysyxSoCFull_ics55.v` - 综合后的网表
- `synthesis_stats_ysyxsocfull.txt` - 综合统计信息
- `synthesis_ysyxsocfull.log` - 完整的综合日志
- `ics55_LLSC_H7CL.v` - 标准单元 Verilog 模型
- `timing_constraints_ysyxsocfull.sdc` - 时序约束文件

## 时序约束

默认时序约束文件位于 `fpga/constraints/timing_ysyxsocfull.sdc`。

### 默认配置

- **时钟周期**: 20ns (50MHz)
- **时钟不确定性**: 0.5ns
- **输入延迟**: 6ns (30% 时钟周期)
- **输出延迟**: 6ns (30% 时钟周期)

### 修改时钟频率

编辑 `fpga/constraints/timing_ysyxsocfull.sdc`：

```tcl
# 修改时钟周期
set CLOCK_PERIOD 10.0  # 100MHz
# 或
set CLOCK_PERIOD 40.0  # 25MHz
```

## 综合统计

查看综合后的资源使用情况：

```bash
cat netlist/synthesis_stats_ysyxsocfull.txt
```

典型输出包括：
- 标准单元数量
- 触发器数量
- 组合逻辑单元数量
- 存储器位数
- 面积估算

## 后续步骤

### 1. 静态时序分析 (STA)

如果安装了 OpenSTA：

```bash
sta -f netlist/timing_constraints_ysyxsocfull.sdc netlist/ysyxSoCFull_ics55.v
```

### 2. 后综合仿真

使用 Icarus Verilog 或 Verilator 进行后综合仿真：

```bash
# 使用 Icarus Verilog
iverilog -o sim netlist/ysyxSoCFull_ics55.v netlist/ics55_LLSC_H7CL.v testbench.v
vvp sim
```

### 3. 布局布线

将网表导入到 OpenROAD 或其他 P&R 工具进行物理设计。

## 故障排除

### 综合失败

1. **检查 RTL 是否生成**
   ```bash
   ls -lh ../ecos/ysyxSoC/build/ysyxSoCFull.v
   ```

2. **检查 PDK 是否安装**
   ```bash
   ls -lh pdk/icsprout55-pdk/IP/STD_cell/
   ```

3. **查看详细日志**
   ```bash
   less netlist/synthesis_ysyxsocfull.log
   ```

### 时序违例

如果出现时序违例：

1. **降低时钟频率** - 增加 SDC 文件中的 `CLOCK_PERIOD`
2. **优化关键路径** - 检查日志中的关键路径
3. **调整约束** - 修改输入/输出延迟约束

### 内存不足

ysyxSoCFull 是一个大型设计，综合可能需要大量内存：

- 确保至少有 8GB 可用内存
- 考虑分模块综合
- 使用更强大的服务器

## 文件说明

- `run_ysyxsocfull_synthesis.sh` - 综合脚本
- `filelist_ysyxsocfull.f` - 源文件列表
- `fpga/constraints/timing_ysyxsocfull.sdc` - 时序约束
- `README_ysyxsocfull.md` - 本文档

## 参考资料

- [Yosys 文档](https://yosyshq.net/yosys/)
- [ICS55 PDK 文档](pdk/icsprout55-pdk/docs/)
- [ysyxSoC 项目](../ecos/ysyxSoC/)
