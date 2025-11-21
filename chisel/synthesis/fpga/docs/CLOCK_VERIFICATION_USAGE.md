# 时钟验证测试使用指南

## 概述

本文档说明如何使用时钟验证测试套件来验证 SPI 时钟的正确性。

## 测试内容

### 1. SPI 时钟频率验证
- **目的**: 验证 SPI 时钟频率是否符合设计要求
- **预期**: 8-10 MHz (目标 10 MHz)
- **方法**: 测量 SPI 时钟周期，计算实际频率

### 2. SPI 时钟占空比验证
- **目的**: 验证 SPI 时钟占空比是否接近 50%
- **预期**: 45-55%
- **方法**: 统计高低电平时间比例

### 3. SPI 时钟稳定性验证
- **目的**: 验证 SPI 时钟频率是否稳定
- **预期**: 频率变化 < 5%
- **方法**: 多次测量频率，计算变化范围

### 4. 分频计数器验证
- **目的**: 验证分频逻辑是否正确
- **预期**: 分频比 = 5-6
- **方法**: 观察计数器行为

### 5. 时钟与数据相位关系验证
- **目的**: 验证数据与时钟的建立/保持时间
- **预期**: 无时序违例
- **方法**: 检测时钟边沿时的数据稳定性

### 6. 时钟边沿质量验证
- **目的**: 验证时钟边沿是否干净，无毛刺
- **预期**: 无毛刺，脉宽 ≥ 2 个主时钟周期
- **方法**: 检测短脉冲

## 快速开始

### 方法 1: 使用自动化脚本（推荐）

```bash
cd chisel/synthesis/fpga
./scripts/run_clock_verification.sh
```

### 方法 2: 手动运行

```bash
cd chisel
sbt "testOnly riscv.ai.ClockVerificationTest"
```

### 方法 3: 运行单个测试

```bash
cd chisel

# 只测试频率
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "generate correct SPI clock frequency"'

# 只测试占空比
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "have approximately 50% duty cycle"'

# 只测试稳定性
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "maintain stable frequency"'
```

## 查看测试结果

### 控制台输出

测试会在控制台输出详细信息：

```
==============================================================
测试 1: SPI 时钟频率验证
==============================================================
主时钟频率: 50 MHz
目标 SPI 频率: 10 MHz
预期分频比: 5-6

  SPI 时钟上升沿 #0: 主时钟周期 3
  SPI 时钟上升沿 #1: 主时钟周期 9
  SPI 时钟上升沿 #2: 主时钟周期 15
  ...

测量结果:
  SPI 时钟周期数: 20
  主时钟周期数: 120
  平均 SPI 周期: 6.00 个主时钟周期
  测量频率: 8.333 MHz
  误差: -16.67%

周期统计:
  最小周期: 6 个主时钟周期
  最大周期: 6 个主时钟周期
  平均周期: 6.00 个主时钟周期
  周期抖动: 0 个主时钟周期

✓ 频率测试通过
```

### 波形文件

测试会生成 VCD 波形文件，可以使用 GTKWave 查看：

```bash
# 查找生成的 VCD 文件
find chisel/test_run_dir -name "*.vcd"

# 使用 GTKWave 打开
gtkwave chisel/test_run_dir/ClockVerificationTest/TFTLCD.vcd
```

**关键信号**:
- `clock` - 主时钟 (50 MHz)
- `io_spi_clk` - SPI 时钟输出
- `io_spi_mosi` - SPI 数据输出
- `io_spi_cs` - SPI 片选

### 测试日志

自动化脚本会保存完整日志：

```bash
cat chisel/test_results/clock_verification.log
```

## 解读测试结果

### 成功的测试

```
[info] ClockVerificationTest:
[info] SPI Clock Frequency
[info] - should generate correct SPI clock frequency from 50MHz main clock
[info] SPI Clock Duty Cycle
[info] - should have approximately 50% duty cycle
[info] SPI Clock Stability
[info] - should maintain stable frequency over extended period
[info] ...
[info] Run completed in 5 seconds, 123 milliseconds.
[info] Total number of tests run: 6
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 6, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
```

### 失败的测试

```
[info] - should generate correct SPI clock frequency from 50MHz main clock *** FAILED ***
[info]   SPI 频率 7.500 MHz 超出范围 [8.0, 10.0] MHz (ClockVerificationTest.scala:65)
```

**可能的原因**:
1. 主时钟频率配置错误
2. 分频参数设置错误
3. 分频逻辑实现错误

## 故障排查

### 问题 1: 频率不正确

**症状**: 测量频率超出 8-10 MHz 范围

**检查**:
```scala
// 检查 TFTLCD 实例化参数
class SimpleEdgeAiSoC(clockFreq: Int = 50000000, ...) {
  val lcd = Module(new SimpleLCDWrapper(clockFreq, 10000000))
  //                                     ^^^^^^^^  ^^^^^^^^
  //                                     主时钟    SPI时钟
}
```

**解决**:
1. 确认 `clockFreq` 参数正确 (50000000)
2. 确认 `spiFreq` 参数正确 (10000000)
3. 检查分频逻辑实现

### 问题 2: 占空比不对

**症状**: 占空比不在 45-55% 范围

**检查**:
```scala
// 检查分频逻辑
when(spiCounter >= spiDivider - 1.U) {
  spiCounter := 0.U
  spiClkReg := !spiClkReg  // 应该翻转
}.otherwise {
  spiCounter := spiCounter + 1.U
}
```

**解决**:
1. 确认分频器值计算正确
2. 确认时钟翻转逻辑正确
3. 检查是否有异步复位影响

### 问题 3: 时钟不稳定

**症状**: 频率变化 > 5%

**可能原因**:
1. 分频计数器溢出
2. 时钟域交叉问题
3. 复位逻辑问题

**解决**:
1. 检查计数器位宽是否足够
2. 确认所有逻辑在同一时钟域
3. 检查复位信号

### 问题 4: 检测到毛刺

**症状**: 边沿质量测试失败

**可能原因**:
1. 组合逻辑产生毛刺
2. 时钟门控不当
3. 多驱动冲突

**解决**:
1. 使用寄存器输出，避免组合逻辑
2. 检查是否有多个驱动源
3. 添加同步器

## 高级用法

### 自定义测试参数

修改测试代码中的参数：

```scala
// 测试不同的时钟配置
test(new TFTLCD(clockFreq = 100000000, spiFreq = 20000000)) { dut =>
  // 测试代码
}
```

### 添加自定义测试

在 `ClockVerificationTest.scala` 中添加新的测试：

```scala
it should "test custom behavior" in {
  test(new TFTLCD(clockFreq = 50000000, spiFreq = 10000000)) { dut =>
    // 你的测试代码
  }
}
```

### 生成详细波形

```bash
# 生成所有测试的波形
sbt "testOnly riscv.ai.ClockVerificationTest" \
  -DwriteVcd=1

# 只生成特定测试的波形
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "frequency" -DwriteVcd=1'
```

## 集成到 CI/CD

### GitHub Actions 示例

```yaml
name: Clock Verification

on: [push, pull_request]

jobs:
  verify-clocks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Java
        uses: actions/setup-java@v2
        with:
          java-version: '11'
          
      - name: Setup SBT
        run: |
          echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
          curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
          sudo apt-get update
          sudo apt-get install sbt
          
      - name: Run Clock Verification
        run: |
          cd chisel/synthesis/fpga
          ./scripts/run_clock_verification.sh
          
      - name: Upload Test Results
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: chisel/test_results/
```

## 性能基准

### 预期测试时间

| 测试 | 时间 |
|------|------|
| 频率验证 | ~2 秒 |
| 占空比验证 | ~3 秒 |
| 稳定性验证 | ~5 秒 |
| 分频器验证 | ~1 秒 |
| 相位关系验证 | ~3 秒 |
| 边沿质量验证 | ~2 秒 |
| **总计** | **~16 秒** |

### 优化建议

如果测试时间过长：

1. **减少测试周期数**:
   ```scala
   val maxCycles = 500  // 从 2000 减少到 500
   ```

2. **并行运行测试**:
   ```bash
   sbt "testOnly riscv.ai.ClockVerificationTest" -J-Xmx4G
   ```

3. **跳过波形生成**:
   ```bash
   # 不生成 VCD 文件
   sbt "testOnly riscv.ai.ClockVerificationTest"
   ```

## 参考文档

- [时钟约束规格](CLOCK_CONSTRAINTS_SPEC.md)
- [时钟约束摘要](CLOCK_SPEC_SUMMARY.md)
- [完整验证指南](CLOCK_VERIFICATION_GUIDE.md)
- [Vivado 验证脚本](../scripts/verify_clocks.tcl)

## 常见问题

### Q: 测试失败但波形看起来正常？

A: 可能是测试阈值设置过严。检查测试代码中的断言条件，适当放宽容差。

### Q: 如何测试不同的时钟频率？

A: 修改测试代码中的 `clockFreq` 和 `spiFreq` 参数。

### Q: 波形文件在哪里？

A: 在 `chisel/test_run_dir/` 目录下，按测试名称组织。

### Q: 如何只运行失败的测试？

A: 使用 `-z` 参数指定测试名称：
```bash
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "frequency"'
```

## 联系支持

如有问题，请联系：
- 设计负责人：童老师
- 后端负责人：[您的名字]
