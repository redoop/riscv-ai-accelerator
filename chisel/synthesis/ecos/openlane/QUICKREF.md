# OpenLane 快速参考

## 🚀 快速开始

### 1. 下载 PDK (首次运行，10-30 分钟)

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openlane
./download_pdk.sh
```

### 2. 运行 OpenLane (1-2 小时)

```bash
./run_final.sh
```

## 📋 常用命令

### 查看状态

```bash
# 查看运行目录
ls -lh ~/OpenLane/designs/asic_top/runs/

# 查看最新运行
ls -lht ~/OpenLane/designs/asic_top/runs/ | head -5
```

### 查看结果

```bash
# 进入最新运行目录
cd ~/OpenLane/designs/asic_top/runs/run_*/

# 查看 GDSII
ls -lh results/final/gds/

# 查看报告
cat reports/final/summary.rpt
```

### 查看日志

```bash
# 综合日志
tail -f ~/OpenLane/designs/asic_top/runs/run_*/logs/synthesis/1-synthesis.log

# 布局日志
tail -f ~/OpenLane/designs/asic_top/runs/run_*/logs/placement/6-placement.log

# 布线日志
tail -f ~/OpenLane/designs/asic_top/runs/run_*/logs/routing/13-routing.log
```

## 🔧 脚本说明

| 脚本 | 用途 | 时间 |
|------|------|------|
| `download_pdk.sh` | 下载 PDK | 10-30 分钟 |
| `run_final.sh` | 运行 OpenLane | 1-2 小时 |
| `run_batch.sh` | 批处理模式 | 1-2 小时 |
| `run_local.sh` | 使用本地安装 | 1-2 小时 |

## 📊 流程步骤

| 步骤 | 时间 | 日志位置 |
|------|------|----------|
| Synthesis | 5-10 分钟 | logs/synthesis/ |
| Floorplan | 1-2 分钟 | logs/floorplan/ |
| Placement | 10-20 分钟 | logs/placement/ |
| CTS | 5-10 分钟 | logs/cts/ |
| Routing | 30-60 分钟 | logs/routing/ |
| Finishing | 5-10 分钟 | logs/finishing/ |

## 🎯 关键文件

### 输入文件

```
config.json                           # OpenLane 配置
../project/netlist/asic_top_ics55.v  # 设计网表
```

### 输出文件

```
results/final/gds/asic_top.gds       # GDSII 版图 ⭐
results/final/def/asic_top.def       # DEF 文件
results/final/verilog/asic_top.v     # 最终网表
reports/final/summary.rpt            # 总结报告 ⭐
reports/final/timing.rpt             # 时序报告
reports/final/area.rpt               # 面积报告
```

## ⚠️ 常见问题

### PDK 未找到

```bash
# 下载 PDK
./download_pdk.sh
```

### Docker 权限错误

```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER
# 重新登录

# 或使用 sudo
sudo ./run_final.sh
```

### 内存不足

```bash
# 增加 Docker 内存
sudo docker update --memory 8g --memory-swap 16g <container_id>
```

### 重新运行

```bash
# 删除旧结果
rm -rf ~/OpenLane/designs/asic_top/runs/run_*

# 重新运行
./run_final.sh
```

## 📚 文档

- [README.md](README.md) - 详细说明
- [USAGE.md](USAGE.md) - 使用指南
- [STATUS.md](STATUS.md) - 当前状态
- [OPENLANE_SUMMARY.md](OPENLANE_SUMMARY.md) - OpenLane 总结

## 🔗 参考

- OpenLane: https://github.com/The-OpenROAD-Project/OpenLane
- 文档: https://openlane.readthedocs.io/
- SkyWater PDK: https://github.com/google/skywater-pdk

---

**更新**: 2025-12-03 12:47
