# RISC-V AI 加速器测试文档

## 测试概述

本文档描述了 RISC-V AI 加速器系统的完整测试策略和测试用例。

## 测试架构

```
测试层次结构:
├── 单元测试 (Unit Tests)
│   ├── MacUnit 测试
│   ├── MatrixMultiplier 测试
│   └── CompactScaleAiChip 测试
├── 集成测试 (Integration Tests)
│   ├── RiscvAiSystem 测试
│   └── RiscvAiChip 测试
└── 系统测试 (System Tests)
    ├── 端到端功能测试
    └── 性能基准测试
```

## 测试环境设置

### 前置条件

```bash
# 安装依赖
sbt update

# 验证环境
sbt compile
```

### 运行所有测试

```bash
# 方法 1: 使用 sbt
sbt test

# 方法 2: 使用测试脚本
./run_integration_tests.sh

# 方法 3: 运行特定测试
sbt "testOnly riscv.ai.MacUnitTest"
```

## 单元测试

### 1. MacUnit 测试

**测试文件**: `src/test/scala/RiscvAiIntegrationTest.scala`

**测试用例**:

#### 1.1 基本乘累加操作
```scala
it should "perform multiply-accumulate correctly"
```
- **目的**: 验证 MAC 单元的基本功能
- **输入**: a=3, b=4, c=5
- **预期输出**: result = 3*4+5 = 17
- **验证点**: 
  - 计算结果正确性
  - valid 信号时序

#### 1.2 负数处理
```scala
it should "handle negative numbers"
```
- **目的**: 验证有符号数运算
- **输入**: a=-2, b=3, c=10
- **预期输出**: result = -2*3+10 = 4
- **验证点**: 
  - 负数乘法正确性
  - 符号扩展正确性

**运行测试**:
```bash
sbt "testOnly riscv.ai.MacUnitTest"
```

**预期输出**:
```
[info] MacUnitTest:
[info] MacUnit
[info] - should perform multiply-accumulate correctly
[info] - should handle negative numbers
[info] Run completed in X seconds.
[info] Total number of tests run: 2
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 2, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
```

### 2. MatrixMultiplier 测试

**测试用例**:

#### 2.1 2x2 矩阵乘法
```scala
it should "multiply 2x2 matrices correctly"
```
- **目的**: 验证矩阵乘法器的基本功能
- **输入**: 
  - A = [[1, 2], [3, 4]]
  - B = [[2, 0], [1, 2]]
- **预期输出**: 
  - C = [[4, 4], [10, 8]]
- **验证点**:
  - 矩阵写入正确性
  - 计算结果正确性
  - 完成信号时序
  - 计算周期数

**测试流程**:
1. 写入矩阵 A 的所有元素
2. 写入矩阵 B 的所有元素
3. 启动计算 (start 信号)
4. 等待完成 (done 信号)
5. 读取并验证结果矩阵 C

**运行测试**:
```bash
sbt "testOnly riscv.ai.MatrixMultiplierTest"
```

### 3. CompactScaleAiChip 测试

**测试用例**:

#### 3.1 AXI 事务处理
```scala
it should "instantiate and respond to AXI transactions"
```
- **目的**: 验证 AXI-Lite 接口功能
- **测试内容**:
  - 写控制寄存器
  - 读状态寄存器
  - AXI 握手协议
- **验证点**:
  - awready/wready 信号
  - rvalid/rdata 信号
  - 寄存器读写正确性

#### 3.2 矩阵数据处理
```scala
it should "process matrix data through AXI"
```
- **目的**: 验证通过 AXI 接口访问矩阵数据
- **测试内容**:
  - 写入矩阵 A 数据
  - 读回数据验证
- **验证点**:
  - 数据完整性
  - 地址映射正确性

**运行测试**:
```bash
sbt "testOnly riscv.ai.CompactScaleAiChipTest"
```

## 集成测试

### 4. RiscvAiChip 测试

**测试用例**:

#### 4.1 基本实例化
```scala
it should "instantiate without errors"
```
- **目的**: 验证顶层模块可以正确实例化
- **验证点**:
  - 无编译错误
  - 初始状态正确
  - 基本信号连接

#### 4.2 内存事务处理
```scala
it should "handle memory transactions"
```
- **目的**: 验证 CPU 内存接口
- **测试流程**:
  1. 等待 CPU 发起内存请求
  2. 响应内存请求 (提供 NOP 指令)
  3. 验证握手协议
- **验证点**:
  - mem_valid/mem_ready 握手
  - 地址输出正确性
  - 数据传输正确性

#### 4.3 性能计数器
```scala
it should "report performance counters"
```
- **目的**: 验证性能监控功能
- **验证点**:
  - 计数器递增
  - 计数器值合理性

**运行测试**:
```bash
sbt "testOnly riscv.ai.RiscvAiIntegrationTest"
```

### 5. RiscvAiSystem 测试

**测试用例**:

#### 5.1 CPU 和 AI 加速器集成
```scala
it should "integrate CPU and AI accelerator"
```
- **目的**: 验证 CPU 和 AI 加速器的完整集成
- **测试流程**:
  1. 初始化系统
  2. 模拟 CPU 内存访问
  3. 监控 AI 加速器状态
  4. 验证性能计数器
- **验证点**:
  - CPU 正常运行
  - AI 加速器响应
  - PCPI 接口工作
  - 状态信号正确

**运行测试**:
```bash
sbt "testOnly riscv.ai.RiscvAiSystemTest"
```

## 测试覆盖率

### 功能覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| MacUnit | 100% | TBD |
| MatrixMultiplier | 90% | TBD |
| CompactScaleAiChip | 80% | TBD |
| RiscvAiSystem | 70% | TBD |
| RiscvAiChip | 70% | TBD |

### 代码覆盖率

生成覆盖率报告:
```bash
sbt clean coverage test coverageReport
```

查看报告:
```bash
open target/scala-2.13/scoverage-report/index.html
```

## 性能测试

### 基准测试

创建性能测试文件 `src/test/scala/PerformanceTest.scala`:

```scala
class PerformanceTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Performance"
  
  it should "measure matrix multiplication latency" in {
    test(new MatrixMultiplier(32, 8)) { dut =>
      // 初始化矩阵...
      
      val startCycle = 0
      dut.io.start.poke(true.B)
      dut.clock.step(1)
      dut.io.start.poke(false.B)
      
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 1000) {
        dut.clock.step(1)
        cycles += 1
      }
      
      println(s"8x8 Matrix multiplication: $cycles cycles")
      assert(cycles < 100, "Latency too high")
    }
  }
}
```

### 性能指标

| 操作 | 预期延迟 | 实际延迟 |
|------|---------|---------|
| 8x8 矩阵乘法 | < 100 cycles | TBD |
| MAC 操作 | 2 cycles | 2 cycles |
| AXI 读写 | 2-3 cycles | TBD |

## 调试技巧

### 1. 波形查看

生成 VCD 波形文件:
```scala
test(new MacUnit(32)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // 测试代码...
}
```

查看波形:
```bash
gtkwave test_run_dir/MacUnit_should_perform_multiply_accumulate_correctly/MacUnit.vcd
```

### 2. 打印调试信息

在测试中添加详细输出:
```scala
println(s"Cycle $i: addr=0x${addr.toString(16)}, data=${data}")
```

### 3. 断言检查

添加中间状态断言:
```scala
dut.io.result.expect(expectedValue.S)
assert(cycles < maxCycles, s"Timeout: $cycles > $maxCycles")
```

## 持续集成

### GitHub Actions 配置

创建 `.github/workflows/test.yml`:

```yaml
name: Chisel Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-java@v2
        with:
          java-version: '11'
      - name: Run tests
        run: |
          cd chisel
          sbt test
```

## 故障排查

### 常见问题

#### 1. 编译错误
```
Error: not found: type MacUnit
```
**解决方案**: 确保所有依赖模块已定义

#### 2. 测试超时
```
Timeout waiting for done signal
```
**解决方案**: 
- 增加超时周期数
- 检查状态机逻辑
- 验证启动信号

#### 3. 数据不匹配
```
Expected: 17, Got: 0
```
**解决方案**:
- 检查流水线延迟
- 验证数据路径
- 添加更多调试输出

## 测试最佳实践

1. **隔离测试**: 每个测试应该独立，不依赖其他测试
2. **清晰命名**: 测试名称应该描述测试内容
3. **充分注释**: 解释测试目的和预期行为
4. **边界条件**: 测试边界值和异常情况
5. **性能监控**: 记录关键操作的周期数
6. **回归测试**: 修改代码后重新运行所有测试

## 下一步

- [ ] 添加更多边界条件测试
- [ ] 实现覆盖率收集
- [ ] 添加性能基准测试
- [ ] 集成到 CI/CD 流程
- [ ] 添加形式化验证

## 参考资料

- [ChiselTest 文档](https://github.com/ucb-bar/chiseltest)
- [ScalaTest 文档](https://www.scalatest.org/)
- [Chisel 测试最佳实践](https://www.chisel-lang.org/chiseltest/)
