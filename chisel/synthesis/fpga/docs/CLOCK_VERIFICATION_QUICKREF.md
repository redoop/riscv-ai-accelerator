# 时钟验证快速参考

## 一键运行

```bash
cd chisel/synthesis/fpga
./scripts/run_clock_verification.sh
```

## 测试项目

| # | 测试项 | 预期结果 | 时间 |
|---|--------|----------|------|
| 1 | 频率 | 8-10 MHz | 2s |
| 2 | 占空比 | 45-55% | 3s |
| 3 | 稳定性 | 变化 < 5% | 5s |
| 4 | 分频器 | 分频比 5-6 | 1s |
| 5 | 相位关系 | 无违例 | 3s |
| 6 | 边沿质量 | 无毛刺 | 2s |

## 常用命令

```bash
# 运行所有测试
sbt "testOnly riscv.ai.ClockVerificationTest"

# 运行单个测试
sbt 'testOnly riscv.ai.ClockVerificationTest -- -z "frequency"'

# 生成波形
sbt "testOnly riscv.ai.ClockVerificationTest" -DwriteVcd=1

# 查看波形
gtkwave chisel/test_run_dir/*/TFTLCD.vcd
```

## 关键参数

| 参数 | 值 |
|------|-----|
| 主时钟 | 100 MHz |
| SPI 目标频率 | 10 MHz |
| SPI 实际频率 | 10.0 MHz |
| 分频比 | 10 |
| 占空比 | ~50% |

## 故障排查

| 问题 | 检查 | 解决 |
|------|------|------|
| 频率错误 | clockFreq 参数 | 确认为 50000000 |
| 占空比错误 | 分频逻辑 | 检查翻转逻辑 |
| 不稳定 | 计数器位宽 | 增加位宽 |
| 毛刺 | 组合逻辑 | 使用寄存器输出 |

## 文件位置

```
chisel/
├── src/test/scala/
│   └── ClockVerificationTest.scala    # 测试代码
├── test_results/
│   └── clock_verification.log         # 测试日志
└── test_run_dir/
    └── */TFTLCD.vcd                   # 波形文件

chisel/synthesis/fpga/
├── docs/
│   ├── CLOCK_VERIFICATION_GUIDE.md    # 完整指南
│   └── CLOCK_VERIFICATION_USAGE.md    # 使用说明
└── scripts/
    ├── run_clock_verification.sh      # 自动化脚本
    └── verify_clocks.tcl              # Vivado 验证
```

## 成功标志

```
✓ 频率测试通过
✓ 占空比测试通过
✓ 稳定性测试通过
✓ 分频器测试通过
✓ 相位关系测试通过
✓ 边沿质量测试通过

总计: 6/6 测试通过
```

## 下一步

1. ✅ 仿真验证通过
2. → Vivado 综合验证
3. → Vivado 实现验证
4. → FPGA 硬件测试

## 联系方式

- 设计: 童老师
- 后端: [您的名字]
