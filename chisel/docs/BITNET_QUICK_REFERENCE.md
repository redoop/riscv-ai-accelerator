# BitNetScaleAiChip 快速参考手册

## 模块清单

| 模块名称 | 数量 | 功能 | 关键参数 |
|---------|------|------|---------|
| `BitNetScaleAiChip` | 1 | 顶层芯片 | 10-bit 地址, AXI-Lite |
| `BitNetMatrixMultiplier` | 2 | 16×16 矩阵乘法器 | 4096 周期延迟 |
| `BitNetComputeUnit` | 2 | 无乘法器计算单元 | 加减法运算 |
| `activationMem_256x16` | 2 | 激活值存储 | 512 字节 |
| `weightMem_256x2` | 2 | 权重存储 | 64 字节 (压缩) |
| `resultMem_256x32` | 2 | 结果存储 | 1 KB |

## 权重编码表

| 编码 | 权重值 | 操作 |
|------|--------|------|
| `2'b00` | 0 | 跳过 (result = accumulator) |
| `2'b01` | +1 | 加法 (result = accumulator + activation) |
| `2'b10` | -1 | 减法 (result = accumulator - activation) |
| `2'b11` | 保留 | 未定义 |

## 寄存器映射

| 地址 | 名称 | 位域 | 说明 |
|------|------|------|------|
| `0x300` | CTRL_REG | [31:0] | 控制寄存器 |
| | | [0] | 矩阵乘法器0启动 (1=启动) |
| | | [1] | 矩阵乘法器1启动 (1=启动) |
| | | [31:2] | 保留 |
| `0x304` | STATUS_REG | [31:0] | 状态寄存器 (只读) |
| | | [0] | 控制寄存器回显 |
| | | [1] | 矩阵乘法器忙碌 (1=忙) |
| | | [2] | 计算完成 (1=完成) |
| | | [3] | 计算单元忙碌 |
| | | [31:4] | 保留 |

## 内存映射

| 地址范围 | 大小 | 内容 | 访问 |
|---------|------|------|------|
| `0x000-0x0FF` | 256×16-bit | 矩阵乘法器0 激活值 | R/W |
| `0x100-0x1FF` | 256×2-bit | 矩阵乘法器0 权重 | R/W |
| `0x200-0x2FF` | 256×32-bit | 矩阵乘法器0 结果 | R/W |
| | | 矩阵乘法器1 激活值 | R/W |
| `0x300` | 32-bit | 控制寄存器 | R/W |
| | | 矩阵乘法器1 权重 | R/W |
| `0x304` | 32-bit | 状态寄存器 | R |

## 典型操作序列

### 1. 单次矩阵乘法

```c
// 1. 写入权重 (2-bit 编码)
for (int i = 0; i < 256; i++) {
    write_axi(0x100 + i, weight_2bit[i]);
}

// 2. 写入激活值 (16-bit)
for (int i = 0; i < 256; i++) {
    write_axi(0x000 + i, activation_16bit[i]);
}

// 3. 启动计算
write_axi(0x300, 0x00000001);  // bit[0] = 1

// 4. 等待完成
while ((read_axi(0x304) & 0x04) == 0) {
    // 轮询 bit[2]
}

// 5. 读取结果 (32-bit)
for (int i = 0; i < 256; i++) {
    result_32bit[i] = read_axi(0x200 + i);
}

// 6. 清除启动标志
write_axi(0x300, 0x00000000);
```

### 2. 双矩阵并行计算

```c
// 矩阵0: 写入数据到 0x000-0x1FF
// 矩阵1: 写入数据到 0x200-0x3FF

// 同时启动两个矩阵乘法器
write_axi(0x300, 0x00000003);  // bit[0]=1, bit[1]=1

// 等待两个都完成
while ((read_axi(0x304) & 0x04) == 0) {
    // 等待完成
}

// 读取两个结果
```

## 性能计算公式

### 延迟
```
单矩阵延迟 = 4096 周期
时间 = 4096 / 频率
例: 100MHz → 40.96 μs
```

### 吞吐量
```
单乘法器吞吐量 = 频率 / 4096
例: 100MHz → 24,414 矩阵/秒

双乘法器吞吐量 = 2 × 单乘法器吞吐量
例: 100MHz → 48,828 矩阵/秒
```

### 峰值性能
```
GOPS = (矩阵大小³ × 2 × 乘法器数) / 延迟
     = (16³ × 2 × 2) / 40.96μs
     = 2 GOPS @ 100MHz
```

## 资源消耗

| 资源类型 | 数量 | 说明 |
|---------|------|------|
| LUTs | ~5,000 | 逻辑单元 |
| FFs | ~350 | 触发器 |
| BRAM | ~3 KB | 块 RAM |
| DSP | 0 | 无乘法器! |

## 功耗估算

| 组件 | 功耗 @ 100MHz |
|------|--------------|
| 计算逻辑 | ~20 mW |
| 存储器 | ~15 mW |
| 接口 | ~10 mW |
| **总计** | **~45 mW** |

## 时序参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| 最大频率 | 100-150 MHz | 取决于工艺 |
| 建立时间 | 2 ns | 输入寄存器 |
| 保持时间 | 0.5 ns | 输入寄存器 |
| 时钟到输出 | 3 ns | 输出寄存器 |

## 调试技巧

### 检查计算是否卡住
```c
uint32_t counter_before = read_perf_counter(3);
delay_ms(1);
uint32_t counter_after = read_perf_counter(3);

if (counter_after == counter_before) {
    printf("计算已停止!\n");
}
```

### 监控性能
```c
uint32_t cycles = read_perf_counter(0);
uint32_t tasks = read_perf_counter(1);
uint32_t active_units = read_perf_counter(2);

printf("总周期: %u\n", cycles);
printf("完成任务: %u\n", tasks);
printf("活跃单元: %u\n", active_units);
```

## 常见问题

### Q: 为什么结果不正确?
A: 检查:
1. 权重编码是否正确 (2-bit)
2. 激活值是否有符号扩展
3. 是否等待计算完成
4. 地址映射是否正确

### Q: 如何提高性能?
A: 
1. 使用双矩阵并行计算
2. 提高时钟频率
3. 使用 DMA 传输数据
4. 启用稀疏性优化

### Q: 功耗如何优化?
A:
1. 降低时钟频率
2. 使用时钟门控
3. 启用稀疏性跳过
4. 优化数据传输

## 版本信息

- **Chisel 版本**: 3.x
- **CIRCT 版本**: 1.62.0
- **设计版本**: 1.0
- **生成日期**: 2024

## 相关链接

- Chisel 源码: `chisel/src/main/scala/BitNetScaleDesign.scala`
- 生成代码: `chisel/generated/BitNetScaleAiChip.sv`
- 详细分析: `chisel/BITNET_CHIP_ANALYSIS.md`
- 模块层次: `chisel/BITNET_MODULE_HIERARCHY.txt`
