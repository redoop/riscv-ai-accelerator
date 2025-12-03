# PSRAM Day 4 快速参考

## ✅ 完成状态

**日期**: 2025-12-03  
**任务**: PSRAM 控制器开发 (基础)  
**状态**: ✅ 完成  
**测试**: 8/8 通过 (100%)

---

## 📁 新增文件

```
chisel/src/main/scala/peripherals/PSRAM.scala    (200 行)
chisel/src/test/scala/PSRAMTest.scala            (250 行)
```

---

## 🎯 核心功能

### 寄存器映射

| 地址 | 名称 | 功能 |
|------|------|------|
| 0x00 | CMD | 命令寄存器 (READ/FAST_READ/WRITE) |
| 0x04 | ADDR | 24-bit 地址 (8 MB) |
| 0x08 | DATA | 32-bit 数据 |
| 0x0C | CTRL | 控制 (start bit) |
| 0x10 | STATUS | 状态 (done flag) |
| 0x14 | CONFIG | 配置 (预留) |

### 支持命令

| 命令 | 代码 | 功能 |
|------|------|------|
| READ | 0x03 | 标准读取 |
| FAST_READ | 0x0B | 快速读取 (带 dummy cycles) |
| WRITE | 0x02 | 写入 |

---

## 🔧 使用示例

### 读取操作

```scala
// 1. 设置命令
poke(dut.io.reg_addr, 0x00.U)
poke(dut.io.reg_wdata, 0x03.U)  // READ
poke(dut.io.reg_wen, true.B)
step(1)

// 2. 设置地址
poke(dut.io.reg_addr, 0x04.U)
poke(dut.io.reg_wdata, 0x001000.U)
step(1)

// 3. 启动操作
poke(dut.io.reg_addr, 0x0C.U)
poke(dut.io.reg_wdata, 0x01.U)  // start
step(1)

// 4. 等待完成
while((peek(dut.io.reg_rdata) & 2) == 0) {
  poke(dut.io.reg_addr, 0x10.U)  // STATUS
  poke(dut.io.reg_ren, true.B)
  step(1)
}

// 5. 读取数据
poke(dut.io.reg_addr, 0x08.U)  // DATA
step(1)
val data = peek(dut.io.reg_rdata)
```

### 写入操作

```scala
// 1. 设置命令
poke(dut.io.reg_addr, 0x00.U)
poke(dut.io.reg_wdata, 0x02.U)  // WRITE
step(1)

// 2. 设置地址
poke(dut.io.reg_addr, 0x04.U)
poke(dut.io.reg_wdata, 0x002000.U)
step(1)

// 3. 设置数据
poke(dut.io.reg_addr, 0x08.U)
poke(dut.io.reg_wdata, "hDEADBEEF".U)
step(1)

// 4. 启动操作
poke(dut.io.reg_addr, 0x0C.U)
poke(dut.io.reg_wdata, 0x01.U)
step(1)

// 5. 等待完成
// (同读取操作)
```

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| SPI 频率 | 50 MHz |
| 地址空间 | 8 MB |
| 数据宽度 | 32-bit |
| 读取延迟 | ~5 us |
| 写入延迟 | ~5 us |
| 带宽 (理论) | 6.25 MB/s |

---

## 🧪 测试命令

```bash
# 运行所有 PSRAM 测试
cd chisel
sbt "testOnly riscv.ai.PSRAMTest"

# 运行单个测试
sbt "testOnly riscv.ai.PSRAMTest -- -z 'initialize'"
sbt "testOnly riscv.ai.PSRAMTest -- -z 'read operation'"
sbt "testOnly riscv.ai.PSRAMTest -- -z 'write operation'"
```

---

## 🚀 下一步

**Day 5: Quad SPI 支持**
- 实现 4-bit 并行传输
- 性能提升 4 倍 (6.25 → 25 MB/s)
- 支持 QUAD_READ/QUAD_WRITE 命令

---

**创建日期**: 2025-12-03  
**版本**: v1.0
