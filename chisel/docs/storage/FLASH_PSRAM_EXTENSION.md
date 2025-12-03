# Flash 和 PSRAM 扩展评估

## 需求概述

添加外部存储器支持：
- **SPI Flash**: 16 MB @ 0x30000000-0x30FFFFFF
- **PSRAM**: 8 MB @ 0x04000000-0x047FFFFF

---

## 改动规模评估

### 📊 总体评估

| 项目 | 工作量 | 复杂度 | 风险 |
|------|--------|--------|------|
| **SPI Flash 控制器** | 🟡 中 (2-3天) | 🟢 低 | 🟢 低 |
| **PSRAM 控制器** | 🟡 中 (2-3天) | 🟡 中 | 🟡 中 |
| **地址解码扩展** | 🟢 小 (0.5天) | 🟢 低 | 🟢 低 |
| **测试验证** | 🟡 中 (2天) | 🟡 中 | 🟡 中 |
| **文档更新** | 🟢 小 (0.5天) | 🟢 低 | 🟢 低 |
| **总计** | **🟡 中 (7-9天)** | **🟡 中** | **🟡 中** |

---

## 详细改动分析

### 1. SPI Flash 控制器

#### 新增代码量: ~300 行 Chisel

```scala
// chisel/src/main/scala/peripherals/SPIFlash.scala
class SPIFlash extends Module {
  val io = IO(new Bundle {
    // 寄存器接口
    val reg = new SimpleRegIO()
    
    // SPI 接口
    val spi_clk = Output(Bool())
    val spi_mosi = Output(Bool())
    val spi_miso = Input(Bool())
    val spi_cs = Output(Bool())
  })
  
  // 状态机
  val sIdle :: sCommand :: sAddress :: sData :: sDone :: Nil = Enum(5)
  val state = RegInit(sIdle)
  
  // 命令寄存器
  val cmdReg = RegInit(0.U(8.W))    // 0x00: 命令 (READ=0x03, FAST_READ=0x0B)
  val addrReg = RegInit(0.U(24.W))  // 0x04: 地址
  val dataReg = RegInit(0.U(32.W))  // 0x08: 数据
  val ctrlReg = RegInit(0.U(32.W))  // 0x0C: 控制 (start, busy, done)
  
  // SPI 时钟分频 (100MHz -> 25MHz)
  val clkDiv = RegInit(0.U(2.W))
  val spiClk = RegInit(false.B)
  when(clkDiv === 1.U) {
    spiClk := ~spiClk
    clkDiv := 0.U
  }.otherwise {
    clkDiv := clkDiv + 1.U
  }
  
  // 状态机逻辑
  switch(state) {
    is(sIdle) {
      when(ctrlReg(0)) { // start bit
        state := sCommand
      }
    }
    is(sCommand) {
      // 发送命令字节
      state := sAddress
    }
    is(sAddress) {
      // 发送24位地址
      state := sData
    }
    is(sData) {
      // 读取数据
      state := sDone
    }
    is(sDone) {
      ctrlReg := ctrlReg & ~1.U // clear start bit
      state := sIdle
    }
  }
  
  // 寄存器读写
  when(io.reg.wen) {
    switch(io.reg.addr(3, 0)) {
      is(0x0.U) { cmdReg := io.reg.wdata(7, 0) }
      is(0x4.U) { addrReg := io.reg.wdata(23, 0) }
      is(0xC.U) { ctrlReg := io.reg.wdata }
    }
  }
  
  when(io.reg.ren) {
    io.reg.rdata := MuxLookup(io.reg.addr(3, 0), 0.U)(Seq(
      0x0.U -> cmdReg,
      0x4.U -> addrReg,
      0x8.U -> dataReg,
      0xC.U -> ctrlReg
    ))
  }
  
  io.spi_clk := spiClk
  io.spi_cs := (state =/= sIdle)
}
```

**改动点**:
1. ✅ 新增文件: `peripherals/SPIFlash.scala` (~300 行)
2. ✅ 修改 SoC: 添加 Flash 模块实例 (~20 行)
3. ✅ 修改地址解码: 添加 0x30000000 范围 (~10 行)
4. ✅ 添加测试: `SPIFlashTest.scala` (~200 行)

**复杂度**: 🟢 低
- SPI 协议简单
- 已有 LCD SPI 参考
- 状态机清晰

---

### 2. PSRAM 控制器

#### 新增代码量: ~400 行 Chisel

```scala
// chisel/src/main/scala/peripherals/PSRAM.scala
class PSRAM extends Module {
  val io = IO(new Bundle {
    // 寄存器接口
    val reg = new SimpleRegIO()
    
    // SPI/QPI 接口
    val spi_clk = Output(Bool())
    val spi_cs = Output(Bool())
    val spi_sio = Analog(4.W)  // SIO0-SIO3 (支持 Quad SPI)
  })
  
  // 状态机
  val sIdle :: sCommand :: sAddress :: sWait :: sData :: sDone :: Nil = Enum(6)
  val state = RegInit(sIdle)
  
  // 寄存器
  val cmdReg = RegInit(0.U(8.W))     // 0x00: 命令
  val addrReg = RegInit(0.U(24.W))   // 0x04: 地址
  val dataReg = RegInit(0.U(32.W))   // 0x08: 数据
  val ctrlReg = RegInit(0.U(32.W))   // 0x0C: 控制
  val statusReg = RegInit(0.U(32.W)) // 0x10: 状态
  
  // PSRAM 命令
  val CMD_READ = 0x03.U
  val CMD_FAST_READ = 0x0B.U
  val CMD_WRITE = 0x02.U
  val CMD_QUAD_READ = 0xEB.U
  val CMD_QUAD_WRITE = 0x38.U
  
  // SPI 时钟分频 (100MHz -> 50MHz for PSRAM)
  val clkDiv = RegInit(0.U(1.W))
  val spiClk = RegInit(false.B)
  when(clkDiv === 0.U) {
    spiClk := ~spiClk
    clkDiv := 1.U
  }.otherwise {
    clkDiv := 0.U
  }
  
  // 位计数器
  val bitCnt = RegInit(0.U(6.W))
  val byteCnt = RegInit(0.U(3.W))
  
  // 状态机
  switch(state) {
    is(sIdle) {
      when(ctrlReg(0)) { // start
        state := sCommand
        bitCnt := 0.U
      }
    }
    is(sCommand) {
      when(bitCnt === 7.U) {
        state := sAddress
        bitCnt := 0.U
      }.otherwise {
        bitCnt := bitCnt + 1.U
      }
    }
    is(sAddress) {
      when(bitCnt === 23.U) {
        state := Mux(cmdReg === CMD_FAST_READ || cmdReg === CMD_QUAD_READ,
                     sWait, sData)
        bitCnt := 0.U
      }.otherwise {
        bitCnt := bitCnt + 1.U
      }
    }
    is(sWait) {
      // Dummy cycles for fast read
      when(bitCnt === 7.U) {
        state := sData
        bitCnt := 0.U
      }.otherwise {
        bitCnt := bitCnt + 1.U
      }
    }
    is(sData) {
      when(bitCnt === 31.U) {
        state := sDone
      }.otherwise {
        bitCnt := bitCnt + 1.U
      }
    }
    is(sDone) {
      ctrlReg := ctrlReg & ~1.U
      statusReg := statusReg | 2.U // done flag
      state := sIdle
    }
  }
  
  // 寄存器接口
  when(io.reg.wen) {
    switch(io.reg.addr(4, 0)) {
      is(0x00.U) { cmdReg := io.reg.wdata(7, 0) }
      is(0x04.U) { addrReg := io.reg.wdata(23, 0) }
      is(0x08.U) { dataReg := io.reg.wdata }
      is(0x0C.U) { ctrlReg := io.reg.wdata }
    }
  }
  
  when(io.reg.ren) {
    io.reg.rdata := MuxLookup(io.reg.addr(4, 0), 0.U)(Seq(
      0x00.U -> cmdReg,
      0x04.U -> addrReg,
      0x08.U -> dataReg,
      0x0C.U -> ctrlReg,
      0x10.U -> statusReg
    ))
  }
  
  io.spi_clk := spiClk
  io.spi_cs := (state =/= sIdle)
}
```

**改动点**:
1. ✅ 新增文件: `peripherals/PSRAM.scala` (~400 行)
2. ✅ 修改 SoC: 添加 PSRAM 模块实例 (~20 行)
3. ✅ 修改地址解码: 添加 0x04000000 范围 (~10 行)
4. ✅ 添加测试: `PSRAMTest.scala` (~250 行)

**复杂度**: 🟡 中
- 支持 Quad SPI 模式
- 需要处理 dummy cycles
- 时序要求较高

---

### 3. SoC 集成改动

#### 修改文件: `EdgeAiSoCSimple.scala`

```scala
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // 现有接口
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    val lcd_spi_clk = Output(Bool())
    // ... 其他现有信号
    
    // ✅ 新增: SPI Flash 接口
    val flash_spi_clk = Output(Bool())
    val flash_spi_mosi = Output(Bool())
    val flash_spi_miso = Input(Bool())
    val flash_spi_cs = Output(Bool())
    
    // ✅ 新增: PSRAM 接口
    val psram_spi_clk = Output(Bool())
    val psram_spi_cs = Output(Bool())
    val psram_spi_sio = Analog(4.W)  // Quad SPI
  })
  
  // 现有模块
  val riscv = Module(new SimplePicoRV32())
  val uart = Module(new RealUART(clockFreq, baudRate))
  val lcd = Module(new TFTLCD())
  val compactAccel = Module(new SimpleCompactAccel())
  val bitnetAccel = Module(new SimpleBitNetAccel())
  
  // ✅ 新增: Flash 控制器
  val flash = Module(new SPIFlash())
  flash.io.reg <> /* 连接到地址解码器 */
  io.flash_spi_clk := flash.io.spi_clk
  io.flash_spi_mosi := flash.io.spi_mosi
  flash.io.spi_miso := io.flash_spi_miso
  io.flash_spi_cs := flash.io.spi_cs
  
  // ✅ 新增: PSRAM 控制器
  val psram = Module(new PSRAM())
  psram.io.reg <> /* 连接到地址解码器 */
  io.psram_spi_clk := psram.io.spi_clk
  io.psram_spi_cs := psram.io.spi_cs
  io.psram_spi_sio <> psram.io.spi_sio
  
  // ✅ 修改: 地址解码器
  val regAddr = io.reg.addr
  when(regAddr >= 0x30000000.U && regAddr < 0x31000000.U) {
    // SPI Flash: 16 MB
    flash.io.reg <> io.reg
  }.elsewhen(regAddr >= 0x04000000.U && regAddr < 0x04800000.U) {
    // PSRAM: 8 MB
    psram.io.reg <> io.reg
  }.elsewhen(/* 现有地址范围 */) {
    // 现有外设
  }
}
```

**改动量**: ~50 行

---

### 4. 内存映射更新

#### 新的内存映射

| 地址范围 | 大小 | 设备 | 说明 |
|---------|------|------|------|
| 0x00000000-0x0000FFFF | 64 KB | RAM | 内部 SRAM |
| **0x04000000-0x047FFFFF** | **8 MB** | **PSRAM** | **外部 PSRAM (新增)** |
| 0x00010000-0x000101FF | 512 B | CompactAccel | 矩阵加速器 |
| 0x00010200-0x000103FF | 512 B | BitNetAccel | BitNet 加速器 |
| 0x00010400-0x0001041F | 32 B | UART | 串口 |
| 0x00010420-0x0001941F | 32 KB | LCD | 帧缓冲 |
| 0x00019420-0x0001943F | 32 B | GPIO | 通用 I/O |
| **0x30000000-0x30FFFFFF** | **16 MB** | **SPI Flash** | **外部 Flash (新增)** |

---

### 5. 软件驱动

#### HAL 层扩展: ~200 行 C

```c
// chisel/software/lib/hal.h

// SPI Flash 寄存器
#define FLASH_BASE      0x30000000
#define FLASH_CMD       (FLASH_BASE + 0x00)
#define FLASH_ADDR      (FLASH_BASE + 0x04)
#define FLASH_DATA      (FLASH_BASE + 0x08)
#define FLASH_CTRL      (FLASH_BASE + 0x0C)

// PSRAM 寄存器
#define PSRAM_BASE      0x04000000
#define PSRAM_CMD       (PSRAM_BASE + 0x00)
#define PSRAM_ADDR      (PSRAM_BASE + 0x04)
#define PSRAM_DATA      (PSRAM_BASE + 0x08)
#define PSRAM_CTRL      (PSRAM_BASE + 0x0C)
#define PSRAM_STATUS    (PSRAM_BASE + 0x10)

// Flash 操作
void flash_init(void);
uint32_t flash_read(uint32_t addr);
void flash_read_block(uint32_t addr, uint8_t *buf, uint32_t len);

// PSRAM 操作
void psram_init(void);
uint32_t psram_read(uint32_t addr);
void psram_write(uint32_t addr, uint32_t data);
void psram_read_block(uint32_t addr, uint8_t *buf, uint32_t len);
void psram_write_block(uint32_t addr, const uint8_t *buf, uint32_t len);
```

```c
// chisel/software/lib/hal.c

void flash_init(void) {
    // 初始化 Flash 控制器
    REG32(FLASH_CTRL) = 0;
}

uint32_t flash_read(uint32_t addr) {
    REG32(FLASH_CMD) = 0x03;  // READ command
    REG32(FLASH_ADDR) = addr;
    REG32(FLASH_CTRL) = 1;    // Start
    
    // 等待完成
    while(REG32(FLASH_CTRL) & 1);
    
    return REG32(FLASH_DATA);
}

void psram_init(void) {
    // 初始化 PSRAM 控制器
    REG32(PSRAM_CTRL) = 0;
}

uint32_t psram_read(uint32_t addr) {
    REG32(PSRAM_CMD) = 0x03;   // READ command
    REG32(PSRAM_ADDR) = addr;
    REG32(PSRAM_CTRL) = 1;     // Start
    
    while(REG32(PSRAM_STATUS) & 1);  // Wait busy
    
    return REG32(PSRAM_DATA);
}

void psram_write(uint32_t addr, uint32_t data) {
    REG32(PSRAM_CMD) = 0x02;   // WRITE command
    REG32(PSRAM_ADDR) = addr;
    REG32(PSRAM_DATA) = data;
    REG32(PSRAM_CTRL) = 1;     // Start
    
    while(REG32(PSRAM_STATUS) & 1);  // Wait busy
}
```

---

### 6. 测试代码

#### Chisel 测试: ~450 行

```scala
// chisel/src/test/scala/SPIFlashTest.scala
class SPIFlashTest extends AnyFlatSpec with ChiselScalatestTester {
  "SPIFlash" should "read data correctly" in {
    test(new SPIFlash()) { dut =>
      // 测试读取命令
      dut.io.reg.wen.poke(true.B)
      dut.io.reg.addr.poke(0x00.U)
      dut.io.reg.wdata.poke(0x03.U)  // READ command
      dut.clock.step(1)
      
      dut.io.reg.addr.poke(0x04.U)
      dut.io.reg.wdata.poke(0x123456.U)  // Address
      dut.clock.step(1)
      
      dut.io.reg.addr.poke(0x0C.U)
      dut.io.reg.wdata.poke(0x01.U)  // Start
      dut.clock.step(1)
      
      // 等待完成
      var cycles = 0
      while(cycles < 100) {
        dut.io.reg.ren.poke(true.B)
        dut.io.reg.addr.poke(0x0C.U)
        dut.clock.step(1)
        if((dut.io.reg.rdata.peek().litValue & 1) == 0) {
          cycles = 100
        }
        cycles += 1
      }
      
      // 验证 SPI 信号
      assert(dut.io.spi_cs.peek().litToBoolean == false)
    }
  }
}

// chisel/src/test/scala/PSRAMTest.scala
class PSRAMTest extends AnyFlatSpec with ChiselScalatestTester {
  "PSRAM" should "write and read data" in {
    test(new PSRAM()) { dut =>
      // 测试写入
      // 测试读取
      // 验证数据一致性
    }
  }
}
```

---

## 改动总结

### 代码量统计

| 文件 | 类型 | 行数 | 复杂度 |
|------|------|------|--------|
| `peripherals/SPIFlash.scala` | 新增 | ~300 | 🟢 低 |
| `peripherals/PSRAM.scala` | 新增 | ~400 | 🟡 中 |
| `EdgeAiSoCSimple.scala` | 修改 | +50 | 🟢 低 |
| `SPIFlashTest.scala` | 新增 | ~200 | 🟢 低 |
| `PSRAMTest.scala` | 新增 | ~250 | 🟡 中 |
| `hal.h` | 修改 | +30 | 🟢 低 |
| `hal.c` | 修改 | +200 | 🟢 低 |
| **总计** | | **~1,430 行** | **🟡 中** |

### 工作量估算

| 任务 | 时间 | 人员 |
|------|------|------|
| SPI Flash 控制器开发 | 2 天 | 1 人 |
| PSRAM 控制器开发 | 2-3 天 | 1 人 |
| SoC 集成 | 0.5 天 | 1 人 |
| 软件驱动开发 | 1 天 | 1 人 |
| 测试验证 | 2 天 | 1 人 |
| 文档更新 | 0.5 天 | 1 人 |
| **总计** | **8-9 天** | **1 人** |

---

## 风险评估

### 技术风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| SPI 时序问题 | 🟡 中 | 参考 LCD SPI 实现 |
| PSRAM Quad SPI | 🟡 中 | 先实现标准 SPI，再扩展 |
| 地址冲突 | 🟢 低 | 清晰的内存映射 |
| 测试覆盖 | 🟡 中 | 完整的单元测试 |

### 集成风险

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 引脚冲突 | 🟢 低 | 独立的 SPI 接口 |
| 时钟域 | 🟢 低 | 使用同一时钟 |
| 资源占用 | 🟢 低 | 增加 < 5% 面积 |

---

## 实施建议

### Phase 1: SPI Flash (3 天)

```bash
# Day 1: 控制器开发
cd chisel/src/main/scala/peripherals
# 创建 SPIFlash.scala
# 实现基本读取功能

# Day 2: 集成和测试
# 修改 EdgeAiSoCSimple.scala
# 创建 SPIFlashTest.scala
# 运行测试

# Day 3: 软件驱动
cd chisel/software/lib
# 扩展 hal.h 和 hal.c
# 创建示例程序
```

### Phase 2: PSRAM (4 天)

```bash
# Day 1-2: 控制器开发
cd chisel/src/main/scala/peripherals
# 创建 PSRAM.scala
# 实现读写功能
# 支持 Quad SPI

# Day 3: 集成和测试
# 修改 EdgeAiSoCSimple.scala
# 创建 PSRAMTest.scala
# 运行测试

# Day 4: 软件驱动和验证
# 扩展 HAL
# 创建测试程序
# 性能测试
```

### Phase 3: 文档和优化 (1 天)

```bash
# 更新 README
# 更新内存映射文档
# 性能优化
# 代码审查
```

---

## 性能预期

### 读取性能

| 设备 | 时钟 | 带宽 | 延迟 |
|------|------|------|------|
| **SPI Flash** | 25 MHz | ~3 MB/s | ~10 us |
| **PSRAM (SPI)** | 50 MHz | ~6 MB/s | ~5 us |
| **PSRAM (Quad)** | 50 MHz | ~24 MB/s | ~3 us |
| 内部 RAM | 100 MHz | ~400 MB/s | ~10 ns |

### 容量提升

```
之前: 64 KB RAM
之后: 64 KB RAM + 8 MB PSRAM + 16 MB Flash
提升: 375× 容量增加
```

---

## 结论

### 改动规模: 🟡 中等

**代码量**: ~1,430 行
**工作量**: 8-9 天 (1 人)
**复杂度**: 🟡 中等
**风险**: 🟡 中等

### 可行性: ✅ 高

- 技术成熟
- 有参考实现 (LCD SPI)
- 清晰的接口定义
- 可分阶段实施

### 建议: ✅ 推荐实施

**优势**:
- ✅ 大幅提升存储容量 (375×)
- ✅ 支持更大模型
- ✅ 实现复杂度可控
- ✅ 性能提升明显

**注意事项**:
- ⚠️ 需要额外引脚 (Flash: 4, PSRAM: 6)
- ⚠️ 增加功耗 (~10-20 mW)
- ⚠️ 需要完整测试验证

---

**评估日期**: 2025年12月3日  
**评估人**: 项目团队  
**状态**: 可行，建议实施
