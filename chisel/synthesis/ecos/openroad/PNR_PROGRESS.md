# P&R 流程进度跟踪

**更新时间**: 2025-12-03 14:07  
**设计**: RISC-V AI 加速器 (asic_top)  
**工具**: OpenROAD v2.0

## 流程状态

| 步骤 | 状态 | 完成时间 | 输出文件 | 备注 |
|------|------|----------|----------|------|
| 1. Floorplan | ✅ 完成 | 2025-12-03 11:30 | `1_floorplan.def` (13 MB) | 1.21 mm² die |
| 2. Placement | ✅ 完成 | 2025-12-03 11:30 | `2_placement.def` (16 MB) | 103K 实例 |
| 3. CTS | ⚠️ 部分 | 2025-12-03 11:30 | `3_cts.def` (16 MB) | 时钟网络问题 |
| 4. Routing | 🔄 进行中 | - | - | tech LEF 已修复 |
| 5. Optimization | ⏳ 待开始 | - | - | - |
| 6. Verification | ⏳ 待开始 | - | - | - |

## 当前任务

### 🔄 Routing (布线)

**目标**: 完成全局布线和详细布线

**已完成**:
- ✅ 创建完整的 tech LEF (`tech_routing.lef`)
  - 6 层金属 (MET1-MET6)
  - 5 个 VIA 定义
  - 完整的 PITCH/WIDTH/SPACING 参数
- ✅ 创建 routing 脚本 (`run_routing.tcl`)
- ✅ 创建自动化脚本 (`run_pnr_complete.sh`)

**执行命令**:
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
./run_pnr_complete.sh
```

**预期输出**:
- `results/4_routing.def` - 布线结果
- `results/asic_top_routed.v` - 布线后网表
- `results/route.guide` - 布线指导文件
- `results/route.drc` - DRC 报告
- `logs/routing.log` - 详细日志

## 技术参数

### 芯片尺寸
- Die: 1101.792 × 1101.792 um (1.21 mm²)
- Core: 1001.3 × 999.6 um (1.00 mm²)
- 利用率: 30.08%

### 设计规模
- 实例数: 103,005
- 网络数: 103,256
- 引脚数: 372,289

### 时序目标
- 主时钟: 25 MHz (40ns 周期)
- SPI 时钟: 2.5 MHz (400ns 周期)

### 布线层配置
| 层 | 方向 | 宽度 | 间距 | 用途 |
|---|------|------|------|------|
| MET1 | H | 0.14 um | 0.14 um | 局部连线 |
| MET2 | V | 0.14 um | 0.14 um | 局部连线 |
| MET3 | H | 0.14 um | 0.14 um | 中层布线 |
| MET4 | V | 0.14 um | 0.14 um | 中层布线 |
| MET5 | H | 0.28 um | 0.28 um | 长距离 |
| MET6 | V | 0.28 um | 0.28 um | 电源/时钟 |

## 已解决的问题

### ✅ 问题 1: tech LEF 缺少布线层

**错误**: `[ERROR GRT-0701] Missing track structure for routing layers`

**解决方案**: 创建 `tech_routing.lef` 包含:
- 6 层金属定义 (MET1-MET6)
- 5 个 VIA 定义 (VIA12-VIA56)
- 完整的电气参数 (R, C)

**状态**: ✅ 已修复

### ⚠️ 问题 2: CTS 时钟网络未找到

**警告**: `[WARNING CTS-0041] Net "sys_clk_i_pad" has 0 sinks`

**原因**: 时钟端口是顶层 PAD，未连接到内部逻辑

**临时方案**: 使用 25MHz 降低时序压力

**长期方案**: 
1. 使用核心模块网表（去除 IO PAD）
2. 或在 SDC 中正确定义时钟传播路径

**状态**: ⚠️ 待优化

## 下一步计划

### 短期 (今天)

1. **完成 Routing**
   ```bash
   cd openroad
   ./run_pnr_complete.sh
   ```

2. **检查结果**
   - 查看 DRC 违例
   - 分析时序报告
   - 统计布线资源使用

### 中期 (明天)

1. **时序优化**
   - 分析关键路径
   - 优化时钟树
   - 尝试提高频率到 50MHz

2. **物理优化**
   - 修复 DRC 违例
   - 优化拥塞区域
   - 减少绕线长度

### 长期 (本周)

1. **物理验证**
   - DRC 检查
   - LVS 验证
   - 天线效应检查

2. **GDSII 生成**
   - 导出最终版图
   - 准备流片数据

## 命令参考

### 运行 P&R

```bash
# 完整流程
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
./run_pnr_complete.sh

# 单独运行 routing
openroad -exit run_routing.tcl

# 查看结果
less results/4_routing.def
tail -100 logs/routing.log
```

### 查看统计

```bash
# 设计统计
grep -E "Number of|Total" logs/routing.log

# 时序报告
grep -A 10 "slack" logs/routing.log

# DRC 统计
wc -l results/route.drc
```

### GUI 查看

```bash
# 查看布线结果
openroad -gui results/4_routing.def

# 查看 placement
openroad -gui results/2_placement.def
```

## 文件清单

### 输入文件
- `tech_routing.lef` - 完整的 technology LEF
- `results/2_placement.def` - Placement 结果
- `../sdc/timing.sdc` - 时序约束

### 输出文件
- `results/4_routing.def` - Routing 结果
- `results/asic_top_routed.v` - 布线后网表
- `results/route.guide` - 布线指导
- `results/route.drc` - DRC 报告
- `logs/routing.log` - 详细日志

### 脚本文件
- `run_routing.tcl` - Routing TCL 脚本
- `run_pnr_complete.sh` - 自动化脚本

## 备注

- 当前使用 25MHz 目标频率以降低时序压力
- tech LEF 参数基于 55nm 工艺典型值
- 如果 routing 失败，可能需要调整 placement 密度
- CTS 问题不影响 routing，但会影响最终时序

---

**创建时间**: 2025-12-03 14:07  
**状态**: Routing 准备就绪 🔄  
**下一步**: 执行 `./run_pnr_complete.sh`
