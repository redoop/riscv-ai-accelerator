# OpenROAD P&R 流程

## 目录结构

```
openroad/
├── config.tcl              # 主配置文件
├── run_all.tcl             # 完整流程脚本
├── run.sh                  # 运行脚本
├── scripts/                # 各步骤脚本
│   ├── 1_floorplan.tcl     # 布图规划
│   ├── 2_placement.tcl     # 布局
│   ├── 3_cts.tcl           # 时钟树综合
│   └── 4_routing.tcl       # 布线
├── results/                # 输出结果
└── logs/                   # 运行日志
```

## 快速开始

### 运行完整流程

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
./run.sh all
```

**预计时间**: 2-4 小时（取决于机器性能）

### 分步运行

```bash
# 1. Floorplan
./run.sh floorplan

# 2. Placement
./run.sh placement

# 3. CTS（关键步骤）
./run.sh cts

# 4. Routing
./run.sh routing
```

## 配置说明

### config.tcl

主要配置参数：

```tcl
# 时钟
set CLOCK_PERIOD 10.0        # 100 MHz

# Floorplan
set DIE_AREA "0 0 600 600"   # 芯片尺寸 600x600 um
set CORE_AREA "50 50 550 550" # 核心区域
set CORE_UTILIZATION 0.7     # 利用率 70%

# CTS
set CTS_ROOT_BUF "BUFX8H7L"  # 根缓冲器
set CTS_BUF_LIST "BUFX2H7L BUFX4H7L BUFX8H7L"
```

### 修改配置

编辑 `config.tcl` 文件，然后重新运行。

## 输出文件

### DEF 文件
- `results/1_floorplan.def` - 布图规划结果
- `results/2_placement.def` - 布局结果
- `results/3_cts.def` - CTS 结果
- `results/4_routing.def` - 布线结果

### Verilog 网表
- `results/3_cts.v` - CTS 后的网表（包含时钟缓冲器）
- `results/4_routing.v` - 最终网表

### 数据库
- `results/*.odb` - OpenROAD 数据库文件

## 验证结果

### 查看时序报告

```bash
# 查看 CTS 日志
cat logs/3_cts.log | grep -A 10 "时序报告"

# 查看最终时序
cat logs/4_routing.log | grep -A 10 "最终时序报告"
```

### 关键指标

- **Clock Skew**: < 0.1 ns（目标）
- **Setup Slack**: > 0 ns（满足 100MHz）
- **Hold Slack**: > 0 ns（无违反）
- **TNS**: 0 ns（无总负时序）

## 常见问题

### Q: 运行时间太长？
**A**: 正常。96,087 个单元需要数小时处理。可以：
- 减小设计规模
- 使用更快的机器
- 调整 PLACE_DENSITY

### Q: 内存不足？
**A**: 需要至少 8GB RAM。可以：
- 关闭其他程序
- 增加 swap
- 使用云服务器

### Q: 出现错误？
**A**: 查看日志文件：
```bash
tail -100 logs/run_all.log
```

## 预期结果

### CTS 后的改进

```
改进前（无 CTS）:
  时钟延迟: 27.593 ns
  总延迟: 37.889 ns
  最大频率: 26 MHz

改进后（CTS）:
  时钟延迟: < 1 ns
  总延迟: ~11 ns
  最大频率: 90-100 MHz ✅
```

### 时钟树结构

```
sys_clk_i_pad
    ↓
[Root Buffer BUFX8H7L]
    ↓
├─ [Level 1: ~16 BUFX4H7L]
│   └─ [Level 2: ~256 BUFX2H7L]
│       └─ 触发器 (每个 ~100)
```

## 下一步

1. **验证时序**: 确认达到 100MHz
2. **物理验证**: DRC/LVS 检查
3. **生成 GDSII**: 准备流片

## 参考

- OpenROAD: https://openroad.readthedocs.io/
- ICS55 PDK: 55nm 工艺

---

**创建时间**: 2025-12-03 10:57
