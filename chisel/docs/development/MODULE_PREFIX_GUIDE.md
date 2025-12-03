# 模块前缀配置指南

## 概述

为了避免多个项目集成时的模块名冲突，本项目支持为生成的 SystemVerilog 模块自动添加前缀。

## 项目编号与前缀对应

| 项目编号 | 前缀 | 项目名称 |
|---------|------|---------|
| project_1854 | `ip0_` | - |
| project_1984 | `ip1_` | SimpleEdgeAiSoC (当前) |
| project_1839 | `ip2_` | - |
| ysyxSoCASIC | `ip3_` | - |
| project_1988 | `ip4_` | - |
| project_1993 | `ip5_` | - |

## 配置方法

### 方法一：修改配置文件（推荐）

编辑 `src/main/scala/GeneratorConfig.scala`：

```scala
object GeneratorConfig {
  // 修改这里的前缀
  val MODULE_PREFIX = "ip1_"  // 改为你的项目前缀
  
  // 是否启用前缀
  val ENABLE_PREFIX = true
}
```

### 方法二：使用环境变量

```bash
# 设置模块前缀
export MODULE_PREFIX="ip1_"

# 生成 Verilog
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
```

### 方法三：命令行参数

```bash
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain --prefix ip1_"
```

## 生成效果

### 不使用前缀

```systemverilog
module SimpleEdgeAiSoC(
  input clock,
  input reset,
  // ...
);

module SimpleMemAdapter(
  // ...
);
```

### 使用 `ip1_` 前缀

```systemverilog
module ip1_SimpleEdgeAiSoC(
  input clock,
  input reset,
  // ...
);

module ip1_SimpleMemAdapter(
  // ...
);
```

## 注意事项

### 1. PicoRV32 模块

PicoRV32 是第三方 IP 核，**不会**自动添加前缀。如需添加前缀，需要手动修改 `picorv32.v` 文件。

### 2. BlackBox 模块

如果使用了 BlackBox 模块，需要确保：
- BlackBox 的 Verilog 实现文件中的模块名也添加相应前缀
- 或者在 BlackBox 定义中指定正确的模块名

示例：

```scala
class MyBlackBox extends BlackBox {
  override def desiredName = s"${GeneratorConfig.MODULE_PREFIX}MyBlackBox"
}
```

### 3. 测试文件

测试文件中的模块实例化也需要更新：

```scala
// 旧代码
val dut = Module(new SimpleEdgeAiSoC)

// 新代码（如果需要）
// Chisel 会自动处理，无需修改
```

## 验证前缀

生成 Verilog 后，检查文件：

```bash
# 查看所有模块定义
grep "^module" generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 应该看到类似输出：
# module ip1_SimpleEdgeAiSoC(
# module ip1_SimpleMemAdapter(
# module ip1_SimpleAddressDecoder(
# ...
```

## 集成到 ASIC 流程

在 `asic_top.sv` 中实例化时：

```systemverilog
`ifdef ip_1
  ip1_SimpleEdgeAiSoC u_SimpleEdgeAiSoC (
    .clock              (ip1_clk_100m),   
    .reset              (~rst_100m_n),
    // ...
  );
`endif
```

## 故障排除

### 问题：firtool 不支持 --prefix-modules

**解决方案**：使用后处理脚本

在 `GeneratorConfig.scala` 中设置：

```scala
val ENABLE_PREFIX = false  // 禁用 firtool 前缀
```

后处理脚本会自动添加前缀。

### 问题：部分模块没有添加前缀

**原因**：可能是 BlackBox 或外部模块

**解决方案**：
1. 检查是否是 PicoRV32 等第三方模块（这些通常不需要前缀）
2. 手动修改 BlackBox 的 `desiredName`
3. 使用后处理脚本的正则表达式过滤

### 问题：编译错误 - 找不到模块

**原因**：模块名不匹配

**解决方案**：
1. 确保所有相关文件都使用了相同的前缀
2. 检查 BlackBox 实现文件
3. 重新生成所有 Verilog 文件

## 最佳实践

1. **统一配置**：在项目开始时确定前缀，写入配置文件
2. **版本控制**：将 `GeneratorConfig.scala` 纳入版本控制
3. **文档记录**：在 README 中说明使用的前缀
4. **自动化测试**：添加测试验证前缀是否正确应用
5. **CI/CD 集成**：在持续集成中检查模块名规范

## 示例：完整工作流

```bash
# 1. 配置前缀
vim src/main/scala/GeneratorConfig.scala
# 设置 MODULE_PREFIX = "ip1_"

# 2. 清理旧文件
sbt clean

# 3. 生成 Verilog
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 4. 验证前缀
grep "^module" generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 5. 运行测试
sbt test

# 6. 综合验证
cd synthesis
make netlist
```

## 参考资料

- [CIRCT firtool 文档](https://circt.llvm.org/docs/Dialects/HW/)
- [Chisel 命名规范](https://www.chisel-lang.org/)
- 项目文档：`docs/DEV_PLAN_V0.2.md`
