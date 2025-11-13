# CompactScaleAiChip 实现状态

## 当前状态

### ✅ 已完成
1. **硬件架构设计**
   - 16个MAC单元
   - 1个8x8矩阵乘法器
   - 512深度存储器
   - AXI-Lite接口
   - 性能计数器
   - 完整的地址映射和控制逻辑

2. **测试框架**
   - 完整的矩阵算法测试 (2x2 到 16x16)
   - 输入/输出矩阵显示
   - 准确度分析
   - 性能统计
   - 时间测量
   - 调试输出

3. **地址映射**
   - 0-255: 矩阵A
   - 256-511: 矩阵B
   - 512-767: 结果矩阵C
   - 0x300: 控制寄存器
   - 0x304: 状态寄存器

### ⚠️ 核心问题：矩阵乘法累加逻辑

**问题描述：**
MatrixMultiplier 的累加逻辑存在根本性的时序问题。当前实现只计算单个乘积，无法正确累加。

**测试结果：**
```
输入矩阵 A: [[5,5], [5,1]]
输入矩阵 B: [[4,7], [6,1]]
期望结果: [[50,40], [26,36]]  (正确的矩阵乘法)
实际输出: [[20,35], [30,5]]   (只有单个乘积，无累加)
```

**根本原因：**
1. 寄存器更新延迟：`accumulators(addr) := value` 在下一个周期才生效
2. 当 k=1 时读取累加器，得到的还是初始值 0，而不是 k=0 的结果
3. 使用 Wire 会导致组合环路错误
4. 使用 RegNext 会导致时序不匹配

**已尝试的方案：**
1. ❌ 直接读写累加器 - 时序延迟问题
2. ❌ 使用 Wire - 组合环路
3. ❌ 使用 RegNext 延迟 - 时序不匹配
4. ❌ 使用 Mem - 同样的时序问题
5. ❌ 使用条件判断 - 逻辑复杂且不work

### 💡 推荐的解决方案

**方案1：使用双缓冲累加器（推荐）**
```scala
// 使用两个累加器数组，交替读写
val accRead = RegInit(VecInit(Seq.fill(N)(0.S)))
val accWrite = RegInit(VecInit(Seq.fill(N)(0.S)))

when(k === 0.U) {
  macUnit.io.c := 0.S
  accWrite(addr) := macUnit.io.result
}.otherwise {
  macUnit.io.c := accRead(addr)
  accWrite(addr) := macUnit.io.result
}

// 在每个新的(i,j)开始时交换缓冲区
when(k === 0.U) {
  accRead := accWrite
}
```

**方案2：重新设计FSM，增加流水线级**
- 将计算分为多个阶段
- 每个阶段有明确的读/写时序
- 增加周期数但保证正确性

**方案3：使用现成的矩阵乘法IP核**
- 考虑使用经过验证的开源IP
- 或参考成熟的实现

### 📁 相关文件
- `chisel/src/main/scala/CompactScaleDesign.scala` - 顶层设计（✅ 正常）
- `chisel/src/main/scala/MatrixMultiplier.scala` - 矩阵乘法器（❌ 需要修复）
- `chisel/src/test/scala/CompactScaleFullMatrixTest.scala` - 完整测试（✅ 正常）

### 🎯 下一步行动
1. **紧急**：修复 MatrixMultiplier 的累加逻辑
2. 验证所有矩阵规模的计算正确性
3. 优化性能
4. 生成最终的 Verilog 文件用于流片

### 📊 当前测试状态
- ✅ 测试框架完整且正常运行
- ✅ 数据写入正确
- ✅ 控制信号正常
- ✅ 计算完成信号正确
- ❌ 计算结果不正确（累加问题）

### 🔧 技术细节
**Chisel 时序规则：**
- `Reg := value` - 下一个周期生效
- `Wire := value` - 当前周期，但可能导致组合环路
- `RegNext(signal)` - 延迟一个周期
- 读写同一个寄存器需要特别小心时序

**矩阵乘法算法：**
```
for i in 0..N:
  for j in 0..N:
    C[i][j] = 0
    for k in 0..N:
      C[i][j] += A[i][k] * B[k][j]
```

关键是在 k 循环中正确累加到 C[i][j]。
