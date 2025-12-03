# OpenLane ASIC 流程

**日期**: 2025-12-03  
**设计**: RISC-V AI 加速器  
**工具**: OpenLane (开源 ASIC 工具链)

## 快速开始

### 1. 安装 OpenLane

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./run_openlane.sh
```

脚本会自动:
- 检查并安装 OpenLane
- 配置设计文件
- 运行完整 ASIC 流程

### 2. 手动运行

```bash
# 安装 OpenLane
cd ~
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenLane.git
cd OpenLane
make

# 创建设计
mkdir -p designs/asic_top/src
cp /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane/config.json designs/asic_top/
cp /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/project/netlist/asic_top_ics55.v designs/asic_top/src/

# 运行流程
make mount
# 在 Docker 容器中:
./flow.tcl -design asic_top
```

## 配置说明

### config.json

```json
{
  "DESIGN_NAME": "asic_top",
  "CLOCK_PORT": "sys_clk_i_pad",
  "CLOCK_PERIOD": 40.0,        // 25MHz
  "FP_CORE_UTIL": 30,          // 30% 利用率
  "PL_TARGET_DENSITY": 0.35    // Placement 密度
}
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CLOCK_PERIOD | 40.0 | 时钟周期 (ns) |
| FP_CORE_UTIL | 30 | 核心利用率 (%) |
| PL_TARGET_DENSITY | 0.35 | Placement 密度 |
| ROUTING_CORES | 4 | 布线并行核心数 |

## 流程步骤

OpenLane 自动执行:

1. **Synthesis** - 逻辑综合
2. **Floorplan** - 布图规划
3. **Placement** - 单元布局
4. **CTS** - 时钟树综合
5. **Routing** - 全局和详细布线
6. **GDSII** - 版图生成
7. **LVS/DRC** - 物理验证

## 输出文件

```
~/OpenLane/designs/asic_top/runs/run_1/
├── results/
│   ├── synthesis/
│   ├── floorplan/
│   ├── placement/
│   ├── cts/
│   ├── routing/
│   └── final/
│       ├── gds/          # GDSII 文件
│       ├── def/          # DEF 文件
│       ├── lef/          # LEF 文件
│       └── verilog/      # 网表
├── reports/              # 报告
└── logs/                 # 日志
```

## 预期时间

| 步骤 | 时间 |
|------|------|
| 安装 OpenLane | 10-30 分钟 |
| Synthesis | 5-10 分钟 |
| Floorplan | 1-2 分钟 |
| Placement | 10-20 分钟 |
| CTS | 5-10 分钟 |
| Routing | 30-60 分钟 |
| **总计** | **1-2 小时** |

## 常见问题

### Q: Docker 内存不足?
```bash
# 增加 Docker 内存限制
docker update --memory 8g --memory-swap 16g openlane
```

### Q: 如何查看进度?
```bash
# 查看日志
tail -f ~/OpenLane/designs/asic_top/runs/run_1/logs/synthesis/1-synthesis.log
```

### Q: 如何重新运行?
```bash
cd ~/OpenLane
make mount
./flow.tcl -design asic_top -tag run_2
```

## 优势

✅ **完整流程**: 从 RTL 到 GDSII  
✅ **IO PAD 支持**: 原生支持 IO PAD  
✅ **自动化**: 一键运行  
✅ **开源**: 完全免费  
✅ **社区支持**: 活跃的社区

## 与 OpenROAD 对比

| 特性 | OpenROAD | OpenLane |
|------|----------|----------|
| Floorplan | ✅ | ✅ |
| Placement | ✅ | ✅ |
| CTS | ⚠️ 需配置 | ✅ 自动 |
| Routing | ⚠️ IO PAD 问题 | ✅ 完整支持 |
| GDSII | ❌ | ✅ |
| 自动化 | 手动 | 全自动 |

## 下一步

1. **运行 OpenLane**
   ```bash
   ./run_openlane.sh
   ```

2. **查看结果**
   ```bash
   cd ~/OpenLane/designs/asic_top/runs/run_1/results/final
   ls -lh gds/
   ```

3. **验证**
   - 检查 DRC 报告
   - 检查 LVS 报告
   - 查看时序报告

## 参考

- OpenLane: https://github.com/The-OpenROAD-Project/OpenLane
- 文档: https://openlane.readthedocs.io/
- 教程: https://github.com/efabless/openlane-workshop

---

**创建时间**: 2025-12-03 11:56  
**状态**: 准备就绪，可以运行
