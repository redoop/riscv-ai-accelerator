# SimpleEdgeAiSoC 完整集成到 ysyxSoC 总结

**日期**: 2025-12-03  
**版本**: v0.4.1-full  
**状态**: ✅ 完整设计

---

## ✅ 集成的完整组件

### 核心组件

| 组件 | 功能 | 性能/容量 | 地址范围 |
|------|------|----------|---------|
| **PicoRV32** | RISC-V CPU | 50-100 MHz | - |
| **CompactAccel** | 8x8 矩阵加速器 | 1.6 GOPS @ 100MHz | 0x2000_0000 - 0x2000_0FFF |
| **BitNetAccel** | 16x16 BitNet 加速器 | 4.8 GOPS @ 100MHz | 0x2000_1000 - 0x2000_1FFF |
| **Flash Controller** | SPI Flash 控制器 | 16 MB @ 25 MHz | 0x2000_2000 - 0x2000_2FFF |
| **PSRAM Controller** | PSRAM 控制器 | 8 MB @ 50 MHz | 0x2000_3000 - 0x2000_3FFF |
| **Flash Memory** | Flash 存储空间 | 16 MB | 0x3000_0000 - 0x30FF_FFFF |
| **PSRAM Memory** | PSRAM 存储空间 | 8 MB | 0x0400_0000 - 0x047F_FFFF |

### 总性能

- **AI 计算**: 6.4 GOPS (1.6 + 4.8)
- **存储容量**: 24 MB (16 MB Flash + 8 MB PSRAM)
- **存储带宽**: 
  - Flash: 3 MB/s (SPI)
  - PSRAM: 6.25 MB/s (SPI), 25 MB/s (Quad SPI)

---

## 🏗️ 完整架构

```
SimpleEdgeAiSoC 集成到 ysyxSoC
├── CPU: PicoRV32 (RV32I)
├── AI 加速器
│   ├── CompactAccel (8x8, 1.6 GOPS)
│   └── BitNetAccel (16x16, 4.8 GOPS)
├── 存储控制器
│   ├── Flash Controller (16 MB SPI)
│   └── PSRAM Controller (8 MB Quad SPI)
└── 存储空间
    ├── Flash Memory (16 MB)
    └── PSRAM Memory (8 MB)
```

---

## 📍 完整内存映射

| 地址范围 | 设备 | 类型 | 说明 |
|----------|------|------|------|
| `0x0000_0000 - 0x0FFF_FFFF` | PSRAM (ysyxSoC) | 外部 | 256 MB 大容量 |
| `0x0400_0000 - 0x047F_FFFF` | **PSRAM (SimpleEdgeAiSoC)** | **内置** | **8 MB 专用** |
| `0x1000_0000 - 0x1FFF_FFFF` | SDRAM (ysyxSoC) | 外部 | 256 MB |
| `0x2000_0000 - 0x2000_0FFF` | **CompactAccel** | **内置** | **4 KB 寄存器** |
| `0x2000_1000 - 0x2000_1FFF` | **BitNetAccel** | **内置** | **4 KB 寄存器** |
| `0x2000_2000 - 0x2000_2FFF` | **Flash Controller** | **内置** | **4 KB 寄存器** |
| `0x2000_3000 - 0x2000_3FFF` | **PSRAM Controller** | **内置** | **4 KB 寄存器** |
| `0x2000_4000 - 0x2FFF_FFFF` | Peripherals (ysyxSoC) | 外部 | UART, SPI, GPIO |
| `0x3000_0000 - 0x30FF_FFFF` | **Flash Memory** | **内置** | **16 MB 存储** |

**粗体** = SimpleEdgeAiSoC 提供的组件

---

## 🔧 关键设计特性

### 1. 混合内存架构

**优势**:
- ✅ SimpleEdgeAiSoC 的 Flash/PSRAM 用于 AI 模型和数据
- ✅ ysyxSoC 的 SDRAM 用于大容量通用存储
- ✅ 两套存储系统互不干扰

**使用场景**:
```c
// AI 模型存储在 SimpleEdgeAiSoC Flash
uint8_t *model = (uint8_t*)0x30000000;  // 16 MB

// AI 数据缓存在 SimpleEdgeAiSoC PSRAM
uint8_t *data = (uint8_t*)0x04000000;   // 8 MB

// 通用数据在 ysyxSoC SDRAM
uint8_t *buffer = (uint8_t*)0x10000000; // 256 MB
```

### 2. 本地 vs 总线访问

| 访问类型 | 路由 | 延迟 |
|---------|------|------|
| AI 加速器 | 本地（不经过总线） | 低 |
| Flash/PSRAM 控制器 | 本地 | 低 |
| Flash/PSRAM 存储 | 本地 | 中 |
| ysyxSoC 外设 | SimpleBus | 中 |
| ysyxSoC SDRAM | SimpleBus | 高 |

### 3. 地址解码逻辑

```verilog
// 本地设备（SimpleEdgeAiSoC）
wire compact_sel    = addr in [0x20000000, 0x20000FFF];
wire bitnet_sel     = addr in [0x20001000, 0x20001FFF];
wire flash_ctrl_sel = addr in [0x20002000, 0x20002FFF];
wire psram_ctrl_sel = addr in [0x20003000, 0x20003FFF];
wire flash_mem_sel  = addr in [0x30000000, 0x30FFFFFF];
wire psram_mem_sel  = addr in [0x04000000, 0x047FFFFF];

wire local_sel = compact_sel | bitnet_sel | flash_ctrl_sel | 
                 psram_ctrl_sel | flash_mem_sel | psram_mem_sel;

// 总线设备（ysyxSoC）
wire bus_sel = ~local_sel & ~mem_instr;
```

---

## 📊 资源占用估算

### 逻辑资源

| 模块 | 标准单元 | 面积 (µm²) | 占比 |
|------|---------|-----------|------|
| PicoRV32 | ~15,000 | ~50,000 | 38% |
| CompactAccel | ~5,000 | ~15,000 | 11% |
| BitNetAccel | ~8,000 | ~25,000 | 19% |
| Flash Controller | ~3,000 | ~10,000 | 8% |
| PSRAM Controller | ~4,000 | ~15,000 | 11% |
| Wrapper 逻辑 | ~2,000 | ~5,000 | 4% |
| 其他 | ~3,000 | ~10,000 | 8% |
| **总计** | **~40,000** | **~130,000** | **100%** |

### 存储资源

| 类型 | 容量 | 用途 |
|------|------|------|
| Flash | 16 MB | AI 模型、程序代码 |
| PSRAM | 8 MB | AI 数据、临时缓存 |
| 内部 RAM | 64 KB | 栈、堆、变量 |
| **总计** | **24.06 MB** | - |

---

## 🎯 应用场景

### 1. AI 推理

```c
// 1. 从 Flash 加载模型
uint8_t *model = (uint8_t*)0x30000000;

// 2. 输入数据到 PSRAM
uint8_t *input = (uint8_t*)0x04000000;
memcpy(input, sensor_data, input_size);

// 3. 配置 BitNetAccel
write_reg(BITNET_BASE, REG_MATRIX_SIZE, 16);
// ... 写入激活值和权重

// 4. 启动计算
write_reg(BITNET_BASE, REG_CTRL, 1);

// 5. 等待中断
while (!bitnet_done);

// 6. 读取结果
uint32_t *result = (uint32_t*)(BITNET_BASE + REG_RESULT);
```

### 2. 大规模数据处理

```c
// SimpleEdgeAiSoC PSRAM: 快速 AI 数据
uint8_t *ai_data = (uint8_t*)0x04000000;  // 8 MB

// ysyxSoC SDRAM: 大容量通用数据
uint8_t *big_data = (uint8_t*)0x10000000; // 256 MB

// 数据流水线
for (int i = 0; i < total_batches; i++) {
  // 1. 从 SDRAM 加载批次数据
  memcpy(ai_data, big_data + i * batch_size, batch_size);
  
  // 2. AI 加速器处理
  process_with_accelerator(ai_data);
  
  // 3. 结果写回 SDRAM
  memcpy(big_data + i * batch_size, ai_data, batch_size);
}
```

### 3. 实时系统

```c
// 高优先级: AI 推理（本地，低延迟）
void ai_task() {
  // 访问 CompactAccel/BitNetAccel
  // 延迟: ~10 cycles
}

// 中优先级: 数据缓存（本地 PSRAM）
void cache_task() {
  // 访问 SimpleEdgeAiSoC PSRAM
  // 延迟: ~50 cycles
}

// 低优先级: 通用 I/O（总线）
void io_task() {
  // 访问 ysyxSoC UART/SPI
  // 延迟: ~100 cycles
}
```

---

## ⚠️ 注意事项

### 1. 地址空间重叠

**PSRAM 地址重叠**:
- ysyxSoC PSRAM: `0x0000_0000 - 0x0FFF_FFFF` (256 MB)
- SimpleEdgeAiSoC PSRAM: `0x0400_0000 - 0x047F_FFFF` (8 MB)

**解决方案**:
- SimpleEdgeAiSoC PSRAM 优先（本地访问）
- 访问 `0x0400_0000 - 0x047F_FFFF` 时，路由到 SimpleEdgeAiSoC
- 其他地址路由到 ysyxSoC

### 2. Flash 启动

**PC 复位值**: `0x30000000` (Flash 起始地址)

**启动流程**:
1. CPU 从 `0x30000000` 取指
2. 访问 SimpleEdgeAiSoC Flash Memory
3. 执行 bootloader
4. 加载程序到 RAM/PSRAM

### 3. 中断优先级

```c
#define IRQ_COMPACT  16  // CompactAccel
#define IRQ_BITNET   17  // BitNetAccel

// 中断处理
void irq_handler() {
  uint32_t irq = read_csr(mip);
  if (irq & (1 << IRQ_BITNET)) {
    handle_bitnet_done();
  }
  if (irq & (1 << IRQ_COMPACT)) {
    handle_compact_done();
  }
}
```

---

## 📚 文件清单

### 源文件

| 文件 | 说明 |
|------|------|
| `ysyx_26000001_with_ai.v` | 完整 wrapper（CPU + AI + 存储） |
| `SimpleCompactAccel.v` | CompactAccel 模块 |
| `SimpleBitNetAccel.v` | BitNetAccel 模块 |
| `SPIFlash.v` | Flash 控制器 |
| `PSRAM.v` | PSRAM 控制器 |
| `picorv32.v` | PicoRV32 核心 |

### 脚本

| 文件 | 说明 |
|------|------|
| `build_sim_with_ai.sh` | 完整编译脚本 |

### 文档

| 文件 | 说明 |
|------|------|
| `YSYXSOC_AI_INTEGRATION.md` | 详细集成指南 |
| `YSYXSOC_FULL_INTEGRATION_SUMMARY.md` | 本文档 |
| `YSYXSOC_INTEGRATION.md` | 基础集成指南 |
| `YSYXSOC_SIMULATION_REPORT.md` | 仿真报告 |

---

## ✅ 验证清单

### 编译验证
- [ ] 生成所有 Verilog 模块
- [ ] Verilator 编译通过
- [ ] 无致命警告

### 功能验证
- [ ] CPU 正常运行
- [ ] CompactAccel 读写测试
- [ ] BitNetAccel 读写测试
- [ ] Flash 控制器测试
- [ ] PSRAM 控制器测试
- [ ] Flash 存储访问测试
- [ ] PSRAM 存储访问测试
- [ ] 中断功能测试

### 性能验证
- [ ] AI 加速器性能测试
- [ ] 存储带宽测试
- [ ] 延迟测试
- [ ] 时序收敛

---

## 🚀 快速开始

```bash
# 1. 编译
cd /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage
./build_sim_with_ai.sh

# 2. 运行
./obj_dir/VysyxSoCTop

# 3. 测试（需要编写 C 程序）
# - CompactAccel: 0x20000000
# - BitNetAccel:  0x20001000
# - Flash Ctrl:   0x20002000
# - PSRAM Ctrl:   0x20003000
# - Flash Memory: 0x30000000
# - PSRAM Memory: 0x04000000
```

---

## 📈 性能对比

| 配置 | AI 性能 | 存储容量 | 存储带宽 |
|------|---------|---------|---------|
| **仅 PicoRV32** | 0 GOPS | 0 MB | 0 MB/s |
| **+ AI 加速器** | 6.4 GOPS | 0 MB | 0 MB/s |
| **+ 存储（完整）** | 6.4 GOPS | 24 MB | 29 MB/s |

**完整集成提升**:
- ✅ AI 性能: ∞ (从无到有)
- ✅ 存储容量: 24 MB
- ✅ 存储带宽: 29 MB/s (3 + 6.25 + 25 Quad)

---

**创建日期**: 2025-12-03  
**状态**: ✅ 完整设计  
**推荐**: 用于 AI 边缘推理应用  
**版本**: v0.4.1-full
