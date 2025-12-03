# OpenSTA 快速使用指南

## 快速运行 STA

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
sta -exit run_sta_with_io.tcl
```

## 查看结果

```bash
# 查看完整结果
cat sta_with_io_result.log

# 查看关键指标
grep "worst slack" sta_with_io_result.log
grep "tns" sta_with_io_result.log
```

## 当前时序状态

- **Setup WNS**: -28.44 ns ❌ 违反
- **Hold WNS**: -0.43 ns ❌ 违反
- **TNS**: -322,249.62 ns
- **目标频率**: 100 MHz (10 ns)
- **实际可达**: ~26 MHz (37.9 ns)

## 关键文件

| 文件 | 说明 |
|------|------|
| `run_sta_with_io.tcl` | STA 运行脚本 |
| `sta_with_io_result.log` | STA 结果 |
| `STA_SUCCESS_REPORT.md` | 详细分析报告 |

## 下一步

1. 查看详细报告：`STA_SUCCESS_REPORT.md`
2. 分析关键路径
3. 优化设计或降低频率

---

**快速命令**：
```bash
# 运行 STA
sta -exit run_sta_with_io.tcl

# 查看 WNS
grep "worst slack" sta_with_io_result.log
```
