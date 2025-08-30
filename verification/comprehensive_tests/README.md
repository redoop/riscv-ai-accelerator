# RISC-V AI Accelerator Comprehensive Test Suite

这是一个完整的测试套件，用于验证RISC-V AI加速器项目的所有主要功能。测试覆盖了从基础RISC-V指令到AI加速器、性能基准测试和系统集成的各个方面。

## 测试套件概述

### 1. 综合功能测试 (`test_riscv_ai_comprehensive.sv`)
- **目标**: 验证核心功能和基本集成
- **覆盖范围**:
  - RISC-V基础指令执行
  - TPU矩阵运算
  - VPU向量操作
  - 内存子系统
  - AI指令集成
  - 性能监控

### 2. AI指令详细测试 (`test_ai_instructions_detailed.sv`)
- **目标**: 验证自定义AI指令扩展 (任务2.3)
- **覆盖范围**:
  - 矩阵乘法指令 (INT8/FP16/FP32)
  - 2D卷积指令 (3x3/5x5核)
  - 激活函数 (ReLU/Sigmoid/Tanh)
  - 池化操作 (Max/Average)
  - 批量归一化
  - 错误处理机制

### 3. 性能基准测试 (`test_performance_benchmarks.sv`)
- **目标**: 验证性能要求 (任务11.2, 12.1)
- **覆盖范围**:
  - 矩阵乘法性能 (不同尺寸和数据类型)
  - 向量操作吞吐量
  - 内存带宽测试
  - AI工作负载模拟 (图像分类)
  - 功耗效率估算

### 4. 系统集成测试 (`test_system_integration.sv`)
- **目标**: 验证完整系统功能 (任务11.3)
- **覆盖范围**:
  - 多核协调
  - AI加速器共享
  - 缓存一致性
  - 热管理
  - 压力测试
  - 硬件软件协同验证

## 快速开始

### 前置要求
- Verilator (推荐版本 4.0+)
- Make
- GCC/Clang (用于C++测试台)
- 至少4GB可用内存

### 安装依赖 (macOS)
```bash
# 使用Homebrew安装Verilator
brew install verilator

# 或者使用MacPorts
sudo port install verilator
```

### 运行所有测试
```bash
cd verification/comprehensive_tests
./run_tests.sh
```

### 运行特定测试套件
```bash
# 运行综合功能测试
./run_tests.sh comprehensive

# 运行AI指令测试
./run_tests.sh ai_instructions

# 运行性能基准测试
./run_tests.sh performance

# 运行系统集成测试
./run_tests.sh integration
```

### 其他选项
```bash
# 仅进行语法检查
./run_tests.sh --syntax-only

# 快速测试模式 (减少仿真时间)
./run_tests.sh --quick

# 清理后运行
./run_tests.sh --clean

# 显示帮助
./run_tests.sh --help
```

## 使用Makefile

### 基本命令
```bash
# 构建并运行所有测试
make test_all

# 运行单个测试套件
make comprehensive
make ai_instructions
make performance
make integration

# 语法检查
make syntax_check

# 覆盖率分析
make coverage

# 调试模式
make debug

# 优化构建
make optimized

# 清理
make clean
```

### 高级选项
```bash
# 创建缺失的RTL存根文件
make create_stubs

# 快速测试
make quick_test

# 显示帮助
make help
```

## 测试结果

### 输出文件
- `logs/`: 包含所有测试日志
- `results/`: 包含测试结果和HTML报告
- `obj_dir/`: Verilator构建输出
- `*.vcd`: 波形文件 (如果启用跟踪)

### 结果解读
测试成功标准：
- ✓ 表示测试通过
- ✗ 表示测试失败
- 最终显示通过/失败统计

性能目标：
- INT8: 100 TOPS
- FP16: 50 TFLOPS  
- FP32: 25 TFLOPS
- 内存带宽: 10 GB/s
- 图像分类: 1000 images/sec

## 测试架构

### 文件结构
```
verification/comprehensive_tests/
├── test_riscv_ai_comprehensive.sv    # 综合功能测试
├── test_ai_instructions_detailed.sv  # AI指令测试
├── test_performance_benchmarks.sv    # 性能基准测试
├── test_system_integration.sv        # 系统集成测试
├── sim_main.cpp                      # C++测试台
├── Makefile                          # 构建脚本
├── run_tests.sh                      # 测试运行脚本
├── README.md                         # 本文档
├── logs/                             # 日志目录
└── results/                          # 结果目录
```

### 测试方法论
1. **单元测试**: 验证单个模块功能
2. **集成测试**: 验证模块间交互
3. **系统测试**: 验证完整系统行为
4. **性能测试**: 验证性能指标
5. **压力测试**: 验证系统稳定性

## 自定义测试

### 添加新测试用例
1. 在相应的测试文件中添加新的`task`
2. 在主测试序列中调用新任务
3. 更新期望结果和通过标准

### 修改测试参数
编辑测试文件顶部的参数定义：
```systemverilog
parameter CLK_PERIOD = 10;        // 时钟周期
parameter TIMEOUT_CYCLES = 10000; // 超时周期
parameter MATRIX_SIZE = 8;        // 矩阵大小
```

### 添加新的测试套件
1. 创建新的`.sv`测试文件
2. 在`Makefile`中添加构建规则
3. 在`run_tests.sh`中添加到`TEST_SUITES`数组

## 调试指南

### 常见问题
1. **编译错误**: 检查RTL文件路径和依赖关系
2. **仿真超时**: 增加`TIMEOUT_CYCLES`或使用`--quick`模式
3. **性能不达标**: 检查时钟频率和算法实现
4. **内存错误**: 检查地址范围和数据对齐

### 调试技巧
```bash
# 启用详细输出
make debug

# 生成波形文件
make WAVES=--trace

# 检查特定模块
verilator --lint-only rtl/core/riscv_core.sv
```

### 波形分析
使用GTKWave查看生成的VCD文件：
```bash
gtkwave comprehensive_trace.vcd
```

## 持续集成

### GitHub Actions示例
```yaml
name: RISC-V AI Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install Verilator
      run: sudo apt-get install verilator
    - name: Run Tests
      run: |
        cd verification/comprehensive_tests
        ./run_tests.sh --quick
```

## 性能优化建议

### 仿真加速
1. 使用`--quick`模式进行快速验证
2. 减少不必要的波形跟踪
3. 使用优化构建 (`make optimized`)
4. 并行运行独立测试

### 内存使用
1. 限制内存模型大小
2. 使用稀疏数据结构
3. 定期清理临时文件

## 贡献指南

### 添加测试
1. 遵循现有的命名约定
2. 添加适当的注释和文档
3. 确保测试具有确定性结果
4. 包含错误情况测试

### 代码风格
- 使用4空格缩进
- 添加有意义的信号名称
- 包含测试描述和目标
- 遵循SystemVerilog最佳实践

## 许可证

本测试套件遵循与主项目相同的许可证。

## 联系信息

如有问题或建议，请通过以下方式联系：
- 项目Issues页面
- 邮件: [项目维护者邮箱]

---

**注意**: 这些测试基于项目规格和实施计划创建。实际的RTL实现可能需要调整测试用例以匹配具体的接口和行为。