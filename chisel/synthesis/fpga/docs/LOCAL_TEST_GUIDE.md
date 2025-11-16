# 本地测试指南

本文档说明如何在本地环境（macOS/Linux）进行 FPGA 设计的验证和测试，无需 AWS F1 实例。

## 适用场景

- 日常开发和调试
- RTL 功能验证
- 快速迭代测试
- 学习和实验

## 环境要求

### 必需工具

- **Java 11+**：运行 Chisel/Scala
- **sbt**：Scala 构建工具
- **Git**：版本控制

### 可选工具

- **Verilator**：高性能 Verilog 仿真器
- **GTKWave**：波形查看器
- **Yosys**：开源综合工具（注意：不完全支持 SystemVerilog）

## 安装步骤

### macOS

```bash
# 安装必需工具
brew install openjdk@11 sbt git

# 安装可选工具
brew install verilator gtkwave yosys

# 配置 Java
echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Linux (Ubuntu/Debian)

```bash
# 安装必需工具
sudo apt update
sudo apt install -y openjdk-11-jdk sbt git

# 安装可选工具
sudo apt install -y verilator gtkwave yosys

# 配置环境
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

### 验证安装

```bash
java -version    # 应显示 Java 11+
sbt --version    # 应显示 sbt 1.x
git --version    # 应显示 git 2.x
```

## 本地测试流程

### 步骤 1：生成 Verilog

```bash
cd chisel

# 方法 1：使用自动化脚本
./run.sh generate

# 方法 2：使用 sbt 直接运行
sbt "runMain edgeai.SimpleEdgeAiSoCMain"
```

**预期输出：**
```
[success] Total time: 15 s
Verilog 生成成功！
输出位置: generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
文件大小: 3765 行
```

**生成的文件：**
- `generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv` - 主 SoC 设计
- `generated/simple_edgeaisoc/PicoRV32.v` - RISC-V 处理器核心

### 步骤 2：运行 RTL 仿真

```bash
# 方法 1：运行所有测试
./run.sh test

# 方法 2：运行特定测试
sbt "testOnly edgeai.SimpleEdgeAiSoCTest"

# 方法 3：运行单个加速器测试
sbt "testOnly edgeai.CompactAccelTest"
sbt "testOnly edgeai.BitNetAccelTest"
```

**预期输出：**
```
[info] SimpleEdgeAiSoCTest:
[info] - should boot and run basic instructions
[info] - should perform matrix multiplication
[info] Run completed in 5 seconds.
[info] Total number of tests run: 2
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 2, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
```

### 步骤 3：查看波形文件

```bash
# 查找生成的波形文件
find chisel/test_run_dir -name "*.vcd"

# 使用 GTKWave 查看（如果已安装）
gtkwave chisel/test_run_dir/SimpleEdgeAiSoC*/SimpleEdgeAiSoC.vcd &
```

### 步骤 4：本地综合检查（可选）

```bash
cd synthesis/fpga

# 使用统一脚本
./run_fpga_flow.sh synthesize local

# 或手动运行 Yosys
yosys -p "read_verilog ../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv; synth; stat"
```

**注意：** Yosys 可能不完全支持 SystemVerilog 特性（如 `automatic` 关键字），这是正常的。本地综合主要用于快速检查，真正的 FPGA 综合应在 AWS F1 上使用 Vivado。

## 测试用例说明

### 1. 处理器启动测试

**测试内容：**
- 复位信号正确性
- 指令执行流程
- 寄存器初始化

**测试代码位置：**
`chisel/src/test/scala/edgeai/SimpleEdgeAiSoCTest.scala`

### 2. CompactAccel 测试

**测试内容：**
- 2x2 矩阵乘法
- 4x4 矩阵乘法
- 8x8 矩阵乘法
- 边界条件测试

**测试向量：**
`synthesis/fpga/testbench/test_vectors/matrix_2x2.txt`

### 3. BitNetAccel 测试

**测试内容：**
- 1-bit 权重矩阵乘法
- XNOR-popcount 操作
- 性能验证

### 4. 外设测试

**测试内容：**
- UART 发送/接收
- GPIO 读写
- 中断处理

## 性能分析

### 查看资源使用

```bash
# 统计 Verilog 代码行数
wc -l generated/simple_edgeaisoc/*.sv

# 估算逻辑资源
grep -c "always" generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
grep -c "reg " generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

### 分析关键路径

```bash
# 查看模块层次
grep "module " generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 查看时钟域
grep "posedge clock" generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv | wc -l
```

## 调试技巧

### 1. 使用 ChiselTest 调试

```scala
// 在测试中添加打印
test(new SimpleEdgeAiSoC) { dut =>
  dut.clock.step(10)
  println(s"PC = ${dut.io.debug_pc.peek()}")
  println(s"State = ${dut.io.debug_state.peek()}")
}
```

### 2. 生成详细波形

```bash
# 设置环境变量生成完整波形
export CHISEL_TEST_DUMP_VCD=1
sbt "testOnly edgeai.SimpleEdgeAiSoCTest"
```

### 3. 查看生成的 Verilog

```bash
# 使用 less 查看
less generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 搜索特定模块
grep -A 20 "module CompactAccel" generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

## 常见问题

### Q1: sbt 编译失败

**问题：**
```
[error] (compile) java.lang.OutOfMemoryError: Java heap space
```

**解决方案：**
```bash
# 增加 JVM 内存
export SBT_OPTS="-Xmx4G -Xss2M"
sbt clean compile
```

### Q2: 测试超时

**问题：**
```
[error] Test timed out after 60 seconds
```

**解决方案：**
```scala
// 在测试中增加超时时间
test(new SimpleEdgeAiSoC).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // 测试代码
}
```

### Q3: 找不到生成的 Verilog

**问题：**
```
File not found: generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

**解决方案：**
```bash
# 检查生成目录
ls -la chisel/generated/

# 重新生成
cd chisel
./run.sh clean
./run.sh generate
```

### Q4: Yosys 综合失败

**问题：**
```
ERROR: syntax error, unexpected TOK_AUTOMATIC
```

**解决方案：**

这是因为 Yosys 不完全支持 SystemVerilog。有两个选择：

1. **跳过本地综合**（推荐）：
```bash
# 只做 RTL 验证
./run.sh test
```

2. **使用 Vivado**（如果有 AWS F1 访问权限）：
```bash
# 在 F1 实例上运行
vivado -mode batch -source scripts/build_fpga.tcl
```

## 快速验证检查清单

完成以下检查确保设计正确：

- [ ] Verilog 生成成功（3765 行）
- [ ] 所有 ChiselTest 测试通过
- [ ] 波形文件可以查看
- [ ] 无编译警告或错误
- [ ] 模块层次结构正确
- [ ] 时钟和复位信号正确

## 性能基准

### 本地测试性能

| 操作 | 时间 | 说明 |
|------|------|------|
| Verilog 生成 | 10-20 秒 | 取决于机器性能 |
| 单个测试 | 2-5 秒 | CompactAccel 测试 |
| 完整测试套件 | 30-60 秒 | 所有测试 |
| 波形生成 | +5-10 秒 | 额外开销 |

### 预期测试结果

```bash
$ cd chisel
$ ./run.sh test

[info] CompactAccelTest:
[info] - 2x2 matrix multiplication ✓
[info] - 4x4 matrix multiplication ✓
[info] - 8x8 matrix multiplication ✓
[info] BitNetAccelTest:
[info] - 2x2 BitNet multiplication ✓
[info] - 8x8 BitNet multiplication ✓
[info] SimpleEdgeAiSoCTest:
[info] - Boot and execute instructions ✓
[info] - UART communication ✓
[info] - GPIO operations ✓
[info] 
[info] All tests passed!
[info] Total time: 45 s
```

## 下一步

本地测试通过后：

1. **继续本地开发**：修改设计并重新测试
2. **准备 AWS 部署**：参考 `SETUP_GUIDE.md`
3. **FPGA 综合**：参考 `BUILD_GUIDE.md`
4. **硬件测试**：参考 `TEST_GUIDE.md`

## 开发工作流建议

```
1. 修改 Chisel 代码
   ↓
2. 本地测试 (./run.sh test)
   ↓
3. 检查波形和结果
   ↓
4. 迭代优化
   ↓
5. 积累多个修改后
   ↓
6. AWS F1 完整验证
```

## 参考资料

- **Chisel 文档**：https://www.chisel-lang.org/
- **ChiselTest**：https://github.com/ucb-bar/chiseltest
- **Verilator**：https://www.veripool.org/verilator/
- **GTKWave**：http://gtkwave.sourceforge.net/

---

**版本**：1.0  
**更新时间**：2025年11月16日  
**维护者**：redoop 团队
