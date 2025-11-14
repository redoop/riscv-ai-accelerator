# EdgeAiSoC 构建指南

## 前置要求

### 软件依赖
- **Scala**: 2.12.x 或 2.13.x
- **SBT**: 1.5.0+
- **Java**: JDK 8 或 11
- **Chisel3**: 3.5.0+

### 可选工具
- **Verilator**: 用于仿真
- **Vivado/Quartus**: 用于 FPGA 综合
- **RISC-V GCC**: 用于软件开发

## 构建步骤

### 1. 生成 Verilog

```bash
cd chisel
sbt "runMain riscv.ai.EdgeAiSoCMain"
```

输出文件:
- `generated/edgeaisoc/EdgeAiSoC.v` - 主 SoC 模块
- `generated/edgeaisoc/*.v` - 其他生成的模块

### 2. 验证生成的 Verilog

```bash
# 使用 Verilator 检查语法
verilator --lint-only generated/edgeaisoc/EdgeAiSoC.v \
    chisel/src/main/resources/rtl/picorv32.v
```

### 3. 仿真测试

```bash
# 使用 Chisel 测试框架
sbt "testOnly riscv.ai.EdgeAiSoCTest"
```

## 项目结构

```
chisel/
├── src/
│   ├── main/
│   │   ├── scala/
│   │   │   └── EdgeAiSoC.scala          # 主设计文件
│   │   └── resources/
│   │       └── rtl/
│   │           └── picorv32.v           # PicoRV32 核心
│   └── test/
│       └── scala/
│           └── EdgeAiSoCTest.scala      # 测试文件
├── docs/
│   ├── EdgeAiSoC_README.md              # 项目文档
│   ├── EdgeAiSoC_BUILD.md               # 构建指南
│   └── RISCV_INTEGRATION_PLAN.md        # 集成方案
└── generated/
    └── edgeaisoc/
        └── EdgeAiSoC.v                  # 生成的 Verilog
```

## FPGA 综合

### Xilinx Vivado

1. 创建新项目
```tcl
create_project edgeaisoc ./vivado_project -part xc7z020clg400-1
```

2. 添加源文件
```tcl
add_files {
    generated/edgeaisoc/EdgeAiSoC.v
    chisel/src/main/resources/rtl/picorv32.v
}
```

3. 设置顶层模块
```tcl
set_property top EdgeAiSoC [current_fileset]
```

4. 添加约束文件
```tcl
add_files -fileset constrs_1 constraints/edgeaisoc.xdc
```

5. 运行综合
```tcl
launch_runs synth_1
wait_on_run synth_1
```

6. 运行实现
```tcl
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```

### Intel Quartus

1. 创建项目
```tcl
project_new edgeaisoc -overwrite
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
```

2. 添加源文件
```tcl
set_global_assignment -name VERILOG_FILE generated/edgeaisoc/EdgeAiSoC.v
set_global_assignment -name VERILOG_FILE chisel/src/main/resources/rtl/picorv32.v
```

3. 设置顶层
```tcl
set_global_assignment -name TOP_LEVEL_ENTITY EdgeAiSoC
```

4. 编译
```tcl
execute_flow -compile
```

## 资源使用估算

### Xilinx Zynq-7020

| 资源 | 使用量 | 可用量 | 利用率 |
|------|--------|--------|--------|
| LUT | ~15,000 | 53,200 | ~28% |
| FF | ~12,000 | 106,400 | ~11% |
| BRAM | ~30 | 140 | ~21% |
| DSP | ~20 | 220 | ~9% |

### Intel Cyclone V

| 资源 | 使用量 | 可用量 | 利用率 |
|------|--------|--------|--------|
| ALM | ~8,000 | 32,070 | ~25% |
| Register | ~12,000 | 128,280 | ~9% |
| M10K | ~60 | 397 | ~15% |
| DSP | ~20 | 87 | ~23% |

## 时序约束

### 时钟约束 (XDC)

```tcl
# 主时钟 100 MHz
create_clock -period 10.000 -name clk [get_ports clk]

# 输入延迟
set_input_delay -clock clk -max 2.0 [get_ports {uart_rx gpio_in[*]}]
set_input_delay -clock clk -min 0.5 [get_ports {uart_rx gpio_in[*]}]

# 输出延迟
set_output_delay -clock clk -max 2.0 [get_ports {uart_tx gpio_out[*]}]
set_output_delay -clock clk -min 0.5 [get_ports {uart_tx gpio_out[*]}]

# 虚假路径
set_false_path -from [get_ports reset]
```

### 时钟约束 (SDC)

```tcl
# 主时钟 100 MHz
create_clock -name clk -period 10.000 [get_ports {clk}]

# 输入延迟
set_input_delay -clock clk -max 2.0 [get_ports {uart_rx gpio_in[*]}]
set_input_delay -clock clk -min 0.5 [get_ports {uart_rx gpio_in[*]}]

# 输出延迟
set_output_delay -clock clk -max 2.0 [get_ports {uart_tx gpio_out[*]}]
set_output_delay -clock clk -min 0.5 [get_ports {uart_tx gpio_out[*]}]
```

## 仿真

### Verilator 仿真

```bash
# 编译仿真器
verilator --cc --exe --build \
    -Wall \
    --top-module EdgeAiSoC \
    generated/edgeaisoc/EdgeAiSoC.v \
    chisel/src/main/resources/rtl/picorv32.v \
    sim/sim_main.cpp

# 运行仿真
./obj_dir/VEdgeAiSoC
```

### ModelSim/QuestaSim 仿真

```tcl
# 编译
vlog generated/edgeaisoc/EdgeAiSoC.v
vlog chisel/src/main/resources/rtl/picorv32.v
vlog sim/testbench.v

# 仿真
vsim -c testbench
run -all
```

## 软件开发

### 编译 RISC-V 程序

```bash
# 安装 RISC-V 工具链
# Ubuntu/Debian:
sudo apt-get install gcc-riscv64-unknown-elf

# 编译示例程序
riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 \
    -nostdlib -nostartfiles \
    -T linker.ld \
    -o firmware.elf \
    startup.S main.c

# 生成二进制文件
riscv64-unknown-elf-objcopy -O binary firmware.elf firmware.bin

# 生成十六进制文件
riscv64-unknown-elf-objcopy -O verilog firmware.elf firmware.hex
```

### 链接脚本示例 (linker.ld)

```ld
MEMORY
{
    ROM (rx)  : ORIGIN = 0x80000000, LENGTH = 256M
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 256M
}

SECTIONS
{
    .text : {
        *(.text.start)
        *(.text*)
    } > ROM

    .data : {
        *(.data*)
    } > RAM

    .bss : {
        *(.bss*)
    } > RAM
}
```

## 调试

### OpenOCD 配置

```tcl
# openocd.cfg
interface ftdi
ftdi_vid_pid 0x0403 0x6010

adapter_khz 1000

# RISC-V target
set _CHIPNAME riscv
jtag newtap $_CHIPNAME cpu -irlen 5 -expected-id 0x10001001

set _TARGETNAME $_CHIPNAME.cpu
target create $_TARGETNAME riscv -chain-position $_TARGETNAME

init
halt
```

### GDB 调试

```bash
# 启动 OpenOCD
openocd -f openocd.cfg

# 在另一个终端启动 GDB
riscv64-unknown-elf-gdb firmware.elf

# GDB 命令
(gdb) target remote localhost:3333
(gdb) load
(gdb) break main
(gdb) continue
```

## 常见问题

### Q: 综合失败，时序不满足
A: 尝试降低时钟频率或添加流水线寄存器

### Q: 仿真时 PicoRV32 无法启动
A: 检查复位信号和初始 PC 地址配置

### Q: 加速器计算结果不正确
A: 验证矩阵数据格式和地址映射

### Q: 中断无法触发
A: 检查中断使能寄存器和中断控制器配置

## 性能优化

### 1. 时钟频率优化
- 添加流水线寄存器
- 优化关键路径
- 使用时钟门控

### 2. 面积优化
- 共享资源
- 减少寄存器数量
- 使用 ROM 替代 RAM

### 3. 功耗优化
- 时钟门控
- 电源门控
- 降低工作频率

## 参考资料

- [Chisel 官方文档](https://www.chisel-lang.org/)
- [PicoRV32 文档](https://github.com/YosysHQ/picorv32)
- [RISC-V 规范](https://riscv.org/specifications/)
- [AXI4 协议规范](https://developer.arm.com/documentation/ihi0022/latest/)
