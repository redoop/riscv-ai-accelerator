# OpenLane 运行状态

**日期**: 2025-12-03 12:47  
**状态**: 配置中 - PDK 问题

## 当前情况

### ✅ 已完成

1. **Docker 已安装**: Docker version 28.5.1
2. **OpenLane 已安装**: ~/OpenLane (v1.0.2)
3. **设计文件已准备**: asic_top_ics55.v (16MB 网表)
4. **配置文件已创建**: config.json

### ❌ 遇到的问题

**PDK (Process Design Kit) 缺失**

OpenLane 需要 SkyWater 130nm PDK，但容器中没有预装。

错误信息:
```
[ERROR]: Failed to compare PDKs.
/build/pdk/sky130A not found.
```

## 解决方案

### 方案 1: 下载 PDK (推荐)

```bash
# 1. 进入 OpenLane 目录
cd ~/OpenLane

# 2. 下载 PDK (需要 10-30 分钟)
sudo docker run --rm \
    -v $(pwd)/pdks:/pdk \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "cd /pdk && volare enable --pdk sky130 --pdk-root /pdk"

# 3. 运行 OpenLane
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
sudo docker run --rm \
    -v ~/OpenLane:/openlane \
    -v ~/OpenLane/designs:/openlane/install \
    -v ~/OpenLane/pdks:/build/pdk \
    -e PDK_ROOT=/build/pdk \
    -e PDK=sky130A \
    ghcr.io/the-openroad-project/openlane:latest \
    bash -c "./flow.tcl -design asic_top"
```

### 方案 2: 使用 OpenROAD (已验证)

OpenROAD 已经在项目中成功运行过，可以完成部分 ASIC 流程:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
./solution_openroad.sh
```

**OpenROAD 可以完成**:
- ✅ 逻辑综合 (Yosys)
- ✅ 布图规划 (Floorplan)
- ✅ 单元布局 (Placement)
- ✅ 时钟树综合 (CTS)
- ⚠️ 布线 (Routing) - 部分支持

**OpenLane 额外提供**:
- ✅ 完整的布线支持
- ✅ GDSII 生成
- ✅ DRC/LVS 验证
- ✅ 更好的 IO PAD 支持

### 方案 3: 使用 ECOS (已完成)

ECOS 综合已经成功完成:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
./run_synthesis.sh
```

**已完成**:
- ✅ 逻辑综合 (623,516 行网表)
- ✅ 芯片面积: 292,992 µm²
- ✅ 标准单元: 96,087 个
- ✅ 网表仿真环境

## 推荐流程

### 短期 (今天)

使用 OpenROAD 完成基本流程:

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
./solution_openroad.sh
```

### 中期 (本周)

下载 PDK 并运行完整 OpenLane 流程:

```bash
# 1. 下载 PDK (一次性，10-30 分钟)
cd ~/OpenLane
make pdk

# 2. 运行 OpenLane (1-2 小时)
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./run_final.sh
```

### 长期 (下周)

完成完整的 ASIC 流程:
1. OpenLane 生成 GDSII
2. DRC/LVS 验证
3. 准备流片文件

## 文件说明

| 文件 | 说明 | 状态 |
|------|------|------|
| `config.json` | OpenLane 配置 | ✅ 已创建 |
| `run_final.sh` | 最终运行脚本 | ✅ 已创建 |
| `run_batch.sh` | 批处理脚本 | ✅ 已创建 |
| `run_local.sh` | 本地运行脚本 | ✅ 已创建 |
| `README.md` | 详细文档 | ✅ 已创建 |
| `USAGE.md` | 使用指南 | ✅ 已创建 |

## 下一步

### 立即可做

1. **使用 OpenROAD**
   ```bash
   cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
   ./solution_openroad.sh
   ```

2. **查看 ECOS 综合结果**
   ```bash
   cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos
   cat SYNTHESIS_REPORT.md
   ```

### 需要时间

1. **下载 PDK** (10-30 分钟)
   ```bash
   cd ~/OpenLane
   make pdk
   ```

2. **运行 OpenLane** (1-2 小时)
   ```bash
   cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
   ./run_final.sh
   ```

## 参考

- OpenLane: https://github.com/The-OpenROAD-Project/OpenLane
- OpenROAD: https://github.com/The-OpenROAD-Project/OpenROAD
- SkyWater PDK: https://github.com/google/skywater-pdk
- ECOS 综合报告: ../SYNTHESIS_REPORT.md

---

**更新时间**: 2025-12-03 12:47  
**下次更新**: PDK 下载完成后
