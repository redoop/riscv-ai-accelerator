# SimpleEdgeAiSoC 接入 ysyxSoC 指南

**日期**: 2025-12-03  
**版本**: v0.4.1  
**状态**: ✅ 完成

---

## 📋 集成概述

将 SimpleEdgeAiSoC 的 PicoRV32 核心接入 ysyxSoC 平台，使其能够：
- 从 Flash (0x3000_0000) 启动
- 通过 SimpleBus 访问外设
- 使用 ysyxSoC 的 UART、SPI 等外设

---

## 🔧 接口适配

### ysyxSoC CPU 接口规范

根据 `ecos/ysyxSoC/ready-to-run/D-stage/cpu-interface.md`:

| 信号 | 方向 | 位宽 | 说明 |
|------|------|------|------|
| `clock` | input | 1 | 时钟 |
| `reset` | input | 1 | 复位 (高电平有效) |
| **IFU 接口** | | | 取指单元 |
| `io_ifu_reqValid` | output | 1 | 取指请求有效 |
| `io_ifu_addr` | output | 32 | 取指地址 |
| `io_ifu_respValid` | input | 1 | 取指响应有效 |
| `io_ifu_rdata` | input | 32 | 取指数据 |
| **LSU 接口** | | | 访存单元 |
| `io_lsu_reqValid` | output | 1 | 访存请求有效 |
| `io_lsu_addr` | output | 32 | 访存地址 |
| `io_lsu_size` | output | 2 | 访存大小 (00=1B, 01=2B, 10=4B) |
| `io_lsu_wen` | output | 1 | 写使能 |
| `io_lsu_wdata` | output | 32 | 写数据 |
| `io_lsu_wmask` | output | 4 | 写掩码 |
| `io_lsu_respValid` | input | 1 | 访存响应有效 |
| `io_lsu_rdata` | input | 32 | 读数据 |

### PicoRV32 接口

SimpleEdgeAiSoC 使用 PicoRV32 核心，其内存接口：

| 信号 | 方向 | 位宽 | 说明 |
|------|------|------|------|
| `mem_valid` | output | 1 | 内存请求有效 |
| `mem_instr` | output | 1 | 指令访问标志 |
| `mem_ready` | input | 1 | 内存就绪 |
| `mem_addr` | output | 32 | 内存地址 |
| `mem_wdata` | output | 32 | 写数据 |
| `mem_wstrb` | output | 4 | 写字节选通 |
| `mem_rdata` | input | 32 | 读数据 |

---

## 🔄 接口转换逻辑

### 1. IFU (取指) 转换

```verilog
// 取指请求：当 mem_valid=1 且 mem_instr=1
assign io_ifu_reqValid = mem_valid & mem_instr;
assign io_ifu_addr = mem_addr;

// 取指响应
wire ifu_resp = io_ifu_respValid & mem_instr;
```

### 2. LSU (访存) 转换

```verilog
// 访存请求：当 mem_valid=1 且 mem_instr=0
assign io_lsu_reqValid = mem_valid & ~mem_instr;
assign io_lsu_addr = mem_addr;
assign io_lsu_wen = |mem_wstrb;  // 任意字节写使能
assign io_lsu_wdata = mem_wdata;
assign io_lsu_wmask = mem_wstrb;

// 访存大小计算
assign io_lsu_size = (mem_wstrb == 4'b1111) ? 2'b10 :  // 4 字节
                     ((mem_wstrb == 4'b0011) || 
                      (mem_wstrb == 4'b1100)) ? 2'b01 : // 2 字节
                     2'b00;                             // 1 字节

// 访存响应
wire lsu_resp = io_lsu_respValid & ~mem_instr;
```

### 3. 内存响应合并

```verilog
assign mem_ready = ifu_resp | lsu_resp;
assign mem_rdata = mem_instr ? io_ifu_rdata : io_lsu_rdata;
```

---

## 📝 实施步骤

### Step 1: 创建 Wrapper 模块

已创建 `ysyx_26000001.v` (示例学号)：

```bash
chisel/generated/simple_edgeaisoc/ysyx_26000001.v
```

**关键配置**:
- PC 复位值: `0x30000000` (Flash 地址)
- 模块名: `ysyx_26000001` (需替换为实际学号)

### Step 2: 修改 ysyxSoCFull.v

在 `ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v` 中：

```verilog
// 第 460 行，修改模块名
// 原: ysyx_00000000 cpu (
// 改为:
ysyx_26000001 cpu (
  .clock            (clock),
  .reset            (reset),
  .io_ifu_addr      (_cpu_io_ifu_addr),
  .io_ifu_reqValid  (_cpu_io_ifu_reqValid),
  .io_ifu_rdata     (_bridge_io_ifu_rdata),
  .io_ifu_respValid (_bridge_io_ifu_respValid),
  .io_lsu_addr      (_cpu_io_lsu_addr),
  .io_lsu_reqValid  (_cpu_io_lsu_reqValid),
  .io_lsu_rdata     (_bridge_io_lsu_rdata),
  .io_lsu_respValid (_bridge_io_lsu_respValid),
  .io_lsu_size      (_cpu_io_lsu_size),
  .io_lsu_wen       (_cpu_io_lsu_wen),
  .io_lsu_wdata     (_cpu_io_lsu_wdata),
  .io_lsu_wmask     (_cpu_io_lsu_wmask)
);
```

### Step 3: Verilator 编译配置

添加以下文件和选项：

**Verilog 文件**:
```bash
# 添加 wrapper
chisel/generated/simple_edgeaisoc/ysyx_26000001.v

# 添加 PicoRV32 核心
chisel/src/main/resources/rtl/picorv32.v

# 添加 ysyxSoC 外设
ecos/ysyxSoC/perip/**/*.v
```

**Include 路径**:
```bash
-I ecos/ysyxSoC/perip/uart16550/rtl
-I ecos/ysyxSoC/perip/spi/rtl
```

**编译选项**:
```bash
--timescale "1ns/1ns"
--no-timing
--top-module ysyxSoCFull
```

### Step 4: C++ 仿真代码

在仿真 cpp 文件中添加：

```cpp
extern "C" void flash_read(int32_t addr, int32_t *data) {
  // TODO: 实现 Flash 读取
  // 暂时返回 NOP 指令
  *data = 0x00000013;  // addi x0, x0, 0 (NOP)
}
```

### Step 5: 编译和仿真

```bash
# 编译
verilator --cc --exe --build \
  --timescale "1ns/1ns" \
  --no-timing \
  --top-module ysyxSoCFull \
  -I ecos/ysyxSoC/perip/uart16550/rtl \
  -I ecos/ysyxSoC/perip/spi/rtl \
  ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v \
  chisel/generated/simple_edgeaisoc/ysyx_26000001.v \
  chisel/src/main/resources/rtl/picorv32.v \
  ecos/ysyxSoC/perip/**/*.v \
  sim_main.cpp

# 运行仿真
./obj_dir/VysyxSoCFull
```

---

## 🎯 验证要点

### 1. 复位地址验证

```verilog
// 确认 PC 复位到 Flash
initial begin
  $display("PC reset address: 0x%h", cpu.reg_pc);
  assert(cpu.reg_pc == 32'h30000000);
end
```

### 2. Flash 访问验证

```cpp
// 验证 flash_read 被调用
extern "C" void flash_read(int32_t addr, int32_t *data) {
  printf("Flash read: addr=0x%08x\n", addr);
  // 返回测试指令
  *data = 0x00000013;  // NOP
}
```

### 3. 总线访问验证

```verilog
// 监控总线活动
always @(posedge clock) begin
  if (io_ifu_reqValid)
    $display("IFU: addr=0x%h", io_ifu_addr);
  if (io_lsu_reqValid)
    $display("LSU: addr=0x%h, wen=%b", io_lsu_addr, io_lsu_wen);
end
```

---

## 📊 内存映射

### ysyxSoC 地址空间

| 地址范围 | 设备 | 说明 |
|----------|------|------|
| `0x0000_0000 - 0x0FFF_FFFF` | PSRAM | 256 MB |
| `0x1000_0000 - 0x1FFF_FFFF` | SDRAM | 256 MB |
| `0x2000_0000 - 0x2FFF_FFFF` | 外设 | UART, SPI, GPIO 等 |
| `0x3000_0000 - 0x3FFF_FFFF` | Flash | 256 MB |

### SimpleEdgeAiSoC 原始映射

| 地址范围 | 设备 | 说明 |
|----------|------|------|
| `0x0000_0000 - 0x0000_FFFF` | RAM | 64 KB |
| `0x0401_0000 - 0x047F_FFFF` | PSRAM | 8 MB |
| `0x1000_0000 - 0x1000_0FFF` | CompactAccel | 4 KB |
| `0x1000_1000 - 0x1000_1FFF` | BitNetAccel | 4 KB |
| `0x2000_0000 - 0x2000_FFFF` | UART | 64 KB |
| `0x2001_0000 - 0x2001_FFFF` | LCD | 64 KB |
| `0x2002_0000 - 0x2002_FFFF` | GPIO | 64 KB |
| `0x3000_0000 - 0x30FF_FFFF` | Flash | 16 MB |

**注意**: 接入 ysyxSoC 后，使用 ysyxSoC 的外设，SimpleEdgeAiSoC 的外设不可用。

---

## ⚠️ 注意事项

### 1. 模块命名

- 文件名: `ysyx_学号.v`
- 顶层模块: `ysyx_学号`
- 内部模块: `ysyx_学号_模块名`

### 2. 复位值

- PC 复位值必须设置为 `0x30000000`
- 复位信号为高电平有效

### 3. 时序

- 使用 `--timescale "1ns/1ns"`
- 添加 `--no-timing` 避免时序检查

### 4. 组合回环

如果遇到组合回环错误：
- 检查 mem_ready 信号生成逻辑
- 确保没有组合逻辑环路
- 必要时添加寄存器打断

---

## 📚 参考文档

### ysyxSoC 文档

- `ecos/ysyxSoC/ysyx.md`: 集成步骤
- `ecos/ysyxSoC/ready-to-run/D-stage/cpu-interface.md`: 接口规范
- `ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v`: 顶层模块

### SimpleEdgeAiSoC 文档

- `chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv`: 原始设计
- `chisel/src/main/resources/rtl/picorv32.v`: PicoRV32 核心

---

## ✅ 集成检查清单

- [ ] 创建 `ysyx_学号.v` wrapper 模块
- [ ] 设置 PC 复位值为 `0x30000000`
- [ ] 实现 IFU/LSU 接口转换
- [ ] 修改 `ysyxSoCFull.v` 中的模块名
- [ ] 添加所有 Verilog 文件到编译列表
- [ ] 添加 include 路径
- [ ] 添加编译选项 `--timescale` 和 `--no-timing`
- [ ] 实现 `flash_read()` 函数
- [ ] 编译通过 (无组合回环错误)
- [ ] 仿真运行 (触发 flash_read)
- [ ] 验证 Flash 访问
- [ ] 验证总线通信

---

## 🚀 下一步

1. **实现 Flash 读取**
   - 加载程序到 Flash 模型
   - 实现完整的 flash_read 函数

2. **测试程序**
   - 编译 RISC-V 测试程序
   - 加载到 Flash
   - 运行仿真验证

3. **外设测试**
   - 测试 UART 通信
   - 测试 SPI 接口
   - 测试 GPIO 功能

---

**创建日期**: 2025-12-03  
**状态**: ✅ Wrapper 已创建，待集成测试  
**学号**: 26000001 (示例，需替换)
