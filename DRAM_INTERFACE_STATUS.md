# 外部 DRAM 接口状态报告

## 当前状态: ❌ 不支持

### 现有内存架构

当前 SimpleEdgeAiSoC 的内存架构：

```
┌─────────────────────────────────────────────────────────┐
│ SimpleEdgeAiSoC                                         │
│                                                         │
│  ┌──────────────┐                                      │
│  │  PicoRV32    │                                      │
│  │   (RV32I)    │                                      │
│  └──────┬───────┘                                      │
│         │ mem_valid, mem_addr, mem_wdata, mem_rdata   │
│         ▼                                              │
│  ┌──────────────┐                                      │
│  │MemAdapter    │                                      │
│  └──────┬───────┘                                      │
│         │ reg interface                                │
│         ▼                                              │
│  ┌──────────────────────────────────────────┐         │
│  │  内部寄存器和 SRAM (64 KB)               │         │
│  │  - 0x00000000: RAM (64 KB)               │         │
│  │  - 0x00010000: CompactAccel              │         │
│  │  - 0x00010200: BitNetAccel               │         │
│  │  - 0x00010400: UART                      │         │
│  │  - 0x00010420: LCD (32 KB framebuffer)   │         │
│  │  - 0x00019420: GPIO                      │         │
│  └──────────────────────────────────────────┘         │
│                                                         │
│  ❌ 无外部 DRAM 接口                                   │
└─────────────────────────────────────────────────────────┘
```

### 代码验证

**文件**: `chisel/src/main/scala/EdgeAiSoCSimple.scala`

```scala
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    val lcd_spi_clk = Output(Bool())
    val lcd_spi_mosi = Output(Bool())
    val lcd_spi_cs = Output(Bool())
    val lcd_spi_dc = Output(Bool())
    val lcd_spi_rst = Output(Bool())
    val lcd_backlight = Output(Bool())
    val gpio_out = Output(UInt(32.W))
    val gpio_in = Input(UInt(32.W))
    // ... 其他信号
    
    // ❌ 没有 DRAM 接口信号
    // 缺少: dram_addr, dram_data, dram_we, dram_oe, etc.
  })
}
```

### 内存限制

| 项目 | 大小 | 用途 |
|------|------|------|
| **总 RAM** | 64 KB | 程序和数据 |
| LCD 帧缓冲 | 32 KB | 显示 (128×128×16bit) |
| **可用内存** | ~32 KB | 程序、栈、堆 |

### 影响分析

#### 1. 模型大小限制
```
可用内存: ~32 KB
典型 BitNet 模型:
- 小型: 10-20 KB    ✅ 可运行
- 中型: 50-100 KB   ❌ 无法运行
- 大型: 1-10 MB     ❌ 无法运行
```

#### 2. 批处理限制
```
单样本推理: ✅ 支持
批处理 (2-4): ⚠️  内存紧张
批处理 (8+):  ❌ 内存不足
```

#### 3. 应用场景限制
```
✅ 可行场景:
- 简单分类 (MNIST, CIFAR-10)
- 关键词识别
- 手势识别
- 简单目标检测

❌ 不可行场景:
- 大型语言模型
- 高分辨率图像处理
- 视频处理
- 复杂多任务推理
```

---

## 为什么不支持外部 DRAM？

### 设计考虑

1. **简化设计**
   - 降低复杂度
   - 减少验证工作量
   - 加快开发周期

2. **目标应用**
   - 边缘 AI 推理
   - 低功耗场景
   - 简单模型部署

3. **成本考虑**
   - 减少芯片面积
   - 降低功耗
   - 简化 PCB 设计

### 技术挑战

如果要添加外部 DRAM 支持，需要：

1. **DRAM 控制器**
   - DDR3/DDR4 PHY
   - 刷新逻辑
   - 时序控制
   - 训练序列

2. **AXI 总线**
   - AXI4 主接口
   - 总线仲裁
   - 缓存一致性

3. **时序约束**
   - 多时钟域
   - 时钟生成 (PLL)
   - 时序收敛

4. **验证复杂度**
   - DRAM 模型
   - 时序仿真
   - 信号完整性

---

## 解决方案

### 方案 1: 优化内存使用 (短期)

**优先级**: 🔴 高

```c
// 1. 模型量化
- 使用 2-bit 权重 (已实现)
- 激活值量化 (8-bit)
- 稀疏性优化 (已实现)

// 2. 内存复用
- 层间缓冲复用
- 原地操作
- 流式处理

// 3. 模型压缩
- 剪枝
- 知识蒸馏
- 低秩分解
```

**预期效果**: 可运行模型大小提升 2-3×

### 方案 2: 添加外部 SRAM (中期)

**优先级**: 🟡 中

```verilog
// 添加 SRAM 接口 (简单，低功耗)
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // ... 现有接口
    
    // 外部 SRAM 接口
    val sram_addr = Output(UInt(20.W))  // 1 MB
    val sram_data = Analog(16.W)
    val sram_we_n = Output(Bool())
    val sram_oe_n = Output(Bool())
    val sram_ce_n = Output(Bool())
  })
}
```

**优势**:
- ✅ 简单实现
- ✅ 低功耗
- ✅ 无需复杂控制器
- ✅ 可扩展到 1-8 MB

**劣势**:
- ⚠️ 速度较慢
- ⚠️ 容量有限

### 方案 3: 添加外部 DRAM (长期)

**优先级**: 🟢 低

```verilog
// 添加 DDR3 接口 (复杂，高性能)
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // ... 现有接口
    
    // DDR3 接口
    val ddr3_addr = Output(UInt(14.W))
    val ddr3_ba = Output(UInt(3.W))
    val ddr3_cas_n = Output(Bool())
    val ddr3_ck_p = Output(Bool())
    val ddr3_ck_n = Output(Bool())
    val ddr3_cke = Output(Bool())
    val ddr3_cs_n = Output(Bool())
    val ddr3_dm = Output(UInt(2.W))
    val ddr3_dq = Analog(16.W)
    val ddr3_dqs_p = Analog(2.W)
    val ddr3_dqs_n = Analog(2.W)
    val ddr3_odt = Output(Bool())
    val ddr3_ras_n = Output(Bool())
    val ddr3_reset_n = Output(Bool())
    val ddr3_we_n = Output(Bool())
  })
}
```

**优势**:
- ✅ 大容量 (256 MB - 2 GB)
- ✅ 高带宽
- ✅ 支持大型模型

**劣势**:
- ❌ 实现复杂
- ❌ 功耗高
- ❌ 验证困难
- ❌ 成本高

---

## 当前建议

### 立即行动

1. **优化内存使用**
   ```bash
   # 实现内存分析工具
   cd chisel/software
   ./tools/analyze_memory.sh
   ```

2. **模型压缩**
   ```python
   # 训练更小的模型
   model = BitNetModel(
       layers=8,        # 减少层数
       hidden=128,      # 减少隐藏层
       sparsity=0.5     # 增加稀疏性
   )
   ```

3. **文档化限制**
   - 更新 README
   - 添加内存使用指南
   - 提供模型大小建议

### 未来考虑

如果需要外部 DRAM，建议：

1. **评估需求**
   - 目标应用场景
   - 模型大小要求
   - 性能要求

2. **选择方案**
   - SRAM: 简单应用 (< 8 MB)
   - DRAM: 复杂应用 (> 8 MB)

3. **分阶段实现**
   - Phase 1: SRAM 接口
   - Phase 2: DDR3 控制器
   - Phase 3: 高级特性

---

## 结论

**当前状态**: ❌ SimpleEdgeAiSoC **不支持**外部 DRAM 接口

**原因**:
- 设计简化
- 降低复杂度
- 针对边缘 AI 场景

**影响**:
- 内存限制 ~32 KB
- 仅支持小型模型
- 批处理受限

**建议**:
1. 短期: 优化内存使用
2. 中期: 考虑外部 SRAM
3. 长期: 评估 DRAM 需求

**风险等级**: 🟡 中等
- 对小型模型: ✅ 可接受
- 对大型模型: ❌ 限制明显

---

**更新时间**: 2025年12月3日  
**状态**: 已确认  
**下次审查**: 需求评估后
