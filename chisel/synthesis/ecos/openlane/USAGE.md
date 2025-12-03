# OpenLane 使用指南

## 快速开始

### 方法 1: 一键运行 (推荐)

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
sudo ./run_openlane_full.sh
```

这个脚本会自动:
1. 准备设计文件
2. 下载 SkyWater 130nm PDK (首次运行)
3. 运行完整 ASIC 流程
4. 生成 GDSII 版图

**预计时间**:
- 首次运行: 1.5-3 小时 (包含 PDK 下载)
- 后续运行: 1-2 小时

### 方法 2: 使用已安装的 OpenLane

如果你已经安装了 OpenLane:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./quick_start.sh
```

## 运行状态

运行过程中会显示:
- ✅ 成功步骤
- ⚠️ 警告信息
- ❌ 错误信息

## 查看结果

### 1. 查看运行目录

```bash
cd work/asic_top/runs/
ls -lh
```

### 2. 查看 GDSII 文件

```bash
cd work/asic_top/runs/run_*/results/final/gds/
ls -lh *.gds
```

### 3. 查看报告

```bash
cd work/asic_top/runs/run_*/reports/
ls -lh
```

关键报告:
- `synthesis/` - 综合报告
- `placement/` - 布局报告
- `routing/` - 布线报告
- `final/` - 最终报告 (时序、面积、功耗)

### 4. 查看日志

```bash
cd work/asic_top/runs/run_*/logs/
tail -f synthesis/1-synthesis.log
```

## 流程步骤

OpenLane 自动执行以下步骤:

1. **Synthesis** (5-10 分钟)
   - 逻辑综合
   - 技术映射

2. **Floorplan** (1-2 分钟)
   - 芯片布图规划
   - IO 放置

3. **Placement** (10-20 分钟)
   - 全局布局
   - 详细布局

4. **CTS** (5-10 分钟)
   - 时钟树综合
   - 时钟缓冲器插入

5. **Routing** (30-60 分钟)
   - 全局布线
   - 详细布线

6. **Finishing** (5-10 分钟)
   - 填充单元
   - 天线修复

7. **Verification** (5-10 分钟)
   - DRC 检查
   - LVS 检查

## 配置参数

编辑 `config.json` 来调整参数:

```json
{
  "DESIGN_NAME": "asic_top",
  "CLOCK_PORT": "sys_clk_i_pad",
  "CLOCK_PERIOD": 40.0,        // 时钟周期 (ns)
  "FP_CORE_UTIL": 30,          // 核心利用率 (%)
  "PL_TARGET_DENSITY": 0.35,   // 布局密度
  "ROUTING_CORES": 4           // 布线并行核心数
}
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CLOCK_PERIOD | 40.0 | 时钟周期 (ns)，40ns = 25MHz |
| FP_CORE_UTIL | 30 | 核心利用率，30% 较宽松 |
| PL_TARGET_DENSITY | 0.35 | 布局密度，0.35 适中 |
| ROUTING_CORES | 4 | 布线并行核心数 |

## 常见问题

### Q1: Docker 权限错误

```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER
# 重新登录生效
```

或者使用 sudo:
```bash
sudo ./run_openlane_full.sh
```

### Q2: 内存不足

```bash
# 增加 Docker 内存限制
sudo docker update --memory 8g --memory-swap 16g <container_id>
```

### Q3: PDK 下载失败

手动下载:
```bash
cd work/pdks
sudo docker run --rm -v $(pwd):/pdk \
  ghcr.io/the-openroad-project/openlane:latest \
  bash -c "cd /pdk && volare enable --pdk sky130 --pdk-root /pdk"
```

### Q4: 如何重新运行

```bash
# 删除旧的运行结果
rm -rf work/asic_top/runs/run_*

# 重新运行
sudo ./run_openlane_full.sh
```

### Q5: 如何查看进度

在另一个终端:
```bash
# 查看当前步骤
tail -f work/asic_top/runs/run_*/logs/synthesis/1-synthesis.log

# 查看所有日志
watch -n 1 'ls -lht work/asic_top/runs/run_*/logs/*/*.log | head -5'
```

## 输出文件说明

```
work/asic_top/runs/run_YYYYMMDD_HHMMSS/
├── results/
│   ├── synthesis/
│   │   └── asic_top.v          # 综合后网表
│   ├── floorplan/
│   │   └── asic_top.def        # 布图规划
│   ├── placement/
│   │   └── asic_top.def        # 布局结果
│   ├── cts/
│   │   └── asic_top.def        # 时钟树综合
│   ├── routing/
│   │   └── asic_top.def        # 布线结果
│   └── final/
│       ├── gds/
│       │   └── asic_top.gds    # GDSII 版图 ⭐
│       ├── def/
│       │   └── asic_top.def    # DEF 文件
│       ├── lef/
│       │   └── asic_top.lef    # LEF 文件
│       └── verilog/
│           └── asic_top.v      # 最终网表
├── reports/
│   ├── synthesis/
│   │   ├── 1-synthesis.rpt     # 综合报告
│   │   └── stat.rpt            # 统计报告
│   ├── placement/
│   │   └── placement.rpt       # 布局报告
│   ├── routing/
│   │   └── routing.rpt         # 布线报告
│   └── final/
│       ├── timing.rpt          # 时序报告 ⭐
│       ├── area.rpt            # 面积报告 ⭐
│       └── power.rpt           # 功耗报告 ⭐
└── logs/
    ├── synthesis/
    ├── floorplan/
    ├── placement/
    ├── cts/
    ├── routing/
    └── finishing/
```

## 下一步

1. **运行 OpenLane**
   ```bash
   sudo ./run_openlane_full.sh
   ```

2. **查看 GDSII**
   ```bash
   # 使用 KLayout 查看 (如果已安装)
   klayout work/asic_top/runs/run_*/results/final/gds/asic_top.gds
   ```

3. **检查报告**
   ```bash
   # 时序报告
   cat work/asic_top/runs/run_*/reports/final/timing.rpt
   
   # 面积报告
   cat work/asic_top/runs/run_*/reports/final/area.rpt
   ```

## 参考资料

- OpenLane 官方文档: https://openlane.readthedocs.io/
- OpenLane GitHub: https://github.com/The-OpenROAD-Project/OpenLane
- SkyWater PDK: https://github.com/google/skywater-pdk

---

**创建时间**: 2025-12-03  
**状态**: 准备就绪
