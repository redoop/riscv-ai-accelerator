# 快速开始指南

5 分钟快速上手 RISC-V AI 加速器 FPGA 验证。

## 🎯 目标

- 生成 Verilog 代码
- 运行 RTL 仿真测试
- 验证设计功能

## ⚡ 快速步骤

### 1. 检查环境

```bash
# 确保已安装必需工具
java -version   # 需要 Java 11+
sbt --version   # 需要 sbt 1.x
```

如果缺少工具，参考 [LOCAL_TEST_GUIDE.md](LOCAL_TEST_GUIDE.md) 安装。

### 2. 生成 Verilog

```bash
cd chisel
./run.sh generate
```

**预期输出：**
```
✓ Verilog 生成成功 (3765 行)
```

### 3. 运行测试

```bash
./run.sh test
```

**预期输出：**
```
[info] All tests passed!
```

### 4. 查看结果

```bash
# 查看生成的 Verilog
ls -lh generated/simple_edgeaisoc/

# 查看测试报告
ls -lh test_run_dir/
```

## ✅ 成功标志

如果看到以下输出，说明一切正常：

```
✓ Verilog 已生成 (3765 行)
✓ 所有测试通过
✓ 无编译错误
```

## 🚀 下一步

### 本地开发

继续在本地开发和测试：
- 修改 `chisel/src/main/scala/edgeai/` 中的代码
- 运行 `./run.sh test` 验证
- 查看 [LOCAL_TEST_GUIDE.md](LOCAL_TEST_GUIDE.md) 了解更多

### AWS F1 部署

准备在真实 FPGA 上验证：
1. [SETUP_GUIDE.md](SETUP_GUIDE.md) - 配置 AWS 环境
2. [BUILD_GUIDE.md](BUILD_GUIDE.md) - 构建 FPGA 镜像
3. [TEST_GUIDE.md](TEST_GUIDE.md) - 硬件测试

## 📚 完整文档

| 文档 | 用途 | 时间 |
|------|------|------|
| [QUICK_START.md](QUICK_START.md) | 快速开始 | 5 分钟 |
| [LOCAL_TEST_GUIDE.md](LOCAL_TEST_GUIDE.md) | 本地测试详解 | 30 分钟 |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | AWS 环境搭建 | 1 小时 |
| [BUILD_GUIDE.md](BUILD_GUIDE.md) | FPGA 构建 | 3-5 小时 |
| [TEST_GUIDE.md](TEST_GUIDE.md) | 硬件测试 | 30 分钟 |

## ❓ 遇到问题？

### 常见问题

**Q: sbt 编译很慢**
```bash
# 增加内存
export SBT_OPTS="-Xmx4G"
```

**Q: 测试失败**
```bash
# 清理并重试
./run.sh clean
./run.sh test
```

**Q: 找不到生成的文件**
```bash
# 检查目录
ls -la generated/
```

更多问题参考 [LOCAL_TEST_GUIDE.md](LOCAL_TEST_GUIDE.md#常见问题)。

---

**提示**：本地测试完全免费，建议先在本地充分验证后再使用 AWS F1。
