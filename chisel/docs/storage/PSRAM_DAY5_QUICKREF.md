# PSRAM Day 5 快速参考 - Quad SPI

## ✅ 完成状态

**日期**: 2025-12-03  
**任务**: Quad SPI 支持  
**状态**: ✅ 完成  
**测试**: 12/12 通过 (100%)  
**性能**: 4× 带宽提升

---

## 🚀 新增功能

### Quad SPI 命令

| 命令 | 代码 | 功能 | 带宽 |
|------|------|------|------|
| QUAD_READ | 0xEB | 4-bit 并行读取 | 25 MB/s |
| QUAD_WRITE | 0x38 | 4-bit 并行写入 | 25 MB/s |
| ENTER_QPI | 0x35 | 进入 QPI 模式 | - |
| EXIT_QPI | 0xF5 | 退出 QPI 模式 | - |

### 新增 IO 信号

```scala
val spi_sio2_out = Output(Bool())  // SIO2 输出
val spi_sio2_oe = Output(Bool())   // SIO2 输出使能
val spi_sio2_in = Input(Bool())    // SIO2 输入
val spi_sio3_out = Output(Bool())  // SIO3 输出
val spi_sio3_oe = Output(Bool())   // SIO3 输出使能
val spi_sio3_in = Input(Bool())    // SIO3 输入
```

---

## 📊 性能对比

### 带宽

| 模式 | 带宽 | 提升 |
|------|------|------|
| 标准 SPI | 6.25 MB/s | 1× |
| **Quad SPI** | **25 MB/s** | **4×** |

### 延迟

| 操作 | 标准 SPI | Quad SPI | 改进 |
|------|----------|----------|------|
| 32-bit 传输 | 640 ns | 160 ns | -75% |
| 完整读取 | 1.44 us | 0.80 us | -44% |

---

## 🔧 使用示例

### Quad 读取

```scala
// 1. 设置命令
poke(dut.io.reg_addr, 0x00.U)
poke(dut.io.reg_wdata, 0xEB.U)  // QUAD_READ
poke(dut.io.reg_wen, true.B)
step(1)

// 2. 设置地址
poke(dut.io.reg_addr, 0x04.U)
poke(dut.io.reg_wdata, 0x004000.U)
step(1)

// 3. 启动
poke(dut.io.reg_addr, 0x0C.U)
poke(dut.io.reg_wdata, 0x01.U)
step(1)

// 4. 等待完成 (更快!)
// ... (同标准 SPI)
```

### QPI 模式切换

```scala
// 进入 QPI 模式
poke(dut.io.reg_addr, 0x00.U)
poke(dut.io.reg_wdata, 0x35.U)  // ENTER_QPI
poke(dut.io.reg_wen, true.B)
step(1)

poke(dut.io.reg_addr, 0x0C.U)
poke(dut.io.reg_wdata, 0x01.U)
step(1)

// 检查模式标志
poke(dut.io.reg_addr, 0x14.U)  // CONFIG
poke(dut.io.reg_ren, true.B)
step(1)
// configReg bit 0 = 1 表示 QPI 模式

// 退出 QPI 模式
poke(dut.io.reg_addr, 0x00.U)
poke(dut.io.reg_wdata, 0xF5.U)  // EXIT_QPI
step(1)
```

---

## 🧪 测试命令

```bash
# 运行所有测试 (包括 Quad SPI)
cd chisel
sbt "testOnly riscv.ai.PSRAMTest"

# 运行 Quad SPI 测试
sbt "testOnly riscv.ai.PSRAMTest -- -z 'quad'"
sbt "testOnly riscv.ai.PSRAMTest -- -z 'QPI'"
```

---

## 📈 测试结果

```
✅ 12/12 测试通过 (100%)

新增测试:
- should support quad read operation ✓
- should support quad write operation ✓
- should enter and exit QPI mode ✓
- should verify quad mode output enables ✓

运行时间: 5.3 秒
```

---

## 🎯 技术要点

### 4-bit 并行传输

- **读取**: 从 SIO[3:0] 同时读取 4 bits
- **写入**: 向 SIO[3:0] 同时发送 4 bits
- **周期**: 32 bits / 4 = 8 SPI 周期

### 输出使能控制

- **读模式**: OE=0 (SIO2/3 为输入)
- **写模式**: OE=1 (SIO2/3 为输出)
- **自动切换**: 根据命令类型

### QPI 模式

- **标志位**: configReg bit 0
- **即时切换**: 无需状态机
- **持久化**: 软件可查询

---

## 🚀 下一步

**Day 6: SoC 集成**
- 集成到 EdgeAiSoCSimple
- 地址解码 (0x04000000-0x047FFFFF)
- SoC 级别测试

---

**创建日期**: 2025-12-03  
**版本**: v1.0
