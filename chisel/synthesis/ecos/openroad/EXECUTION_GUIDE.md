# OpenROAD 执行指南

## ✅ 配置已完成

所有配置文件已生成并验证通过！

## 📁 目录结构

```
openroad/
├── config.tcl              ✅ 主配置
├── run_all.tcl             ✅ 完整流程
├── run.sh                  ✅ 运行脚本
├── test_config.tcl         ✅ 配置验证
├── scripts/
│   ├── 1_floorplan.tcl     ✅ 布图规划
│   ├── 2_placement.tcl     ✅ 布局
│   ├── 3_cts.tcl           ✅ 时钟树综合
│   └── 4_routing.tcl       ✅ 布线
├── results/                📁 输出目录
└── logs/                   📁 日志目录
```

## 🚀 立即执行

### 方式 1：完整流程（推荐）

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
./run.sh all
```

**预计时间**: 2-4 小时  
**输出**: results/4_routing.v（包含时钟树的最终网表）

### 方式 2：分步执行

```bash
# Step 1: Floorplan (5-10 分钟)
./run.sh floorplan

# Step 2: Placement (30-60 分钟)
./run.sh placement

# Step 3: CTS (10-20 分钟) ⭐ 关键步骤
./run.sh cts

# Step 4: Routing (1-2 小时)
./run.sh routing
```

## 📊 预期结果

### 时序改进

| 指标 | 改进前 | 改进后 | 改进 |
|------|--------|--------|------|
| 时钟延迟 | 27.593 ns | < 1 ns | 27x ⚡ |
| 总延迟 | 37.889 ns | ~11 ns | 3.4x |
| 最大频率 | 26 MHz | 90-100 MHz | 3.8x |

### 时钟树统计

- **插入缓冲器**: ~256 个
- **时钟 Skew**: < 0.1 ns
- **扇出**: 每个缓冲器 ~100 个触发器

## 📝 监控进度

### 实时查看日志

```bash
# 另开一个终端
tail -f logs/run_all.log
```

### 检查进度

```bash
# 查看已完成的步骤
ls -lh results/
```

## ⚠️ 注意事项

### 系统要求

- **内存**: 至少 8GB RAM
- **磁盘**: 至少 5GB 可用空间
- **时间**: 2-4 小时不间断运行

### 如果中断

```bash
# 从上次完成的步骤继续
./run.sh placement  # 如果 floorplan 已完成
./run.sh cts        # 如果 placement 已完成
./run.sh routing    # 如果 cts 已完成
```

## 🔍 验证结果

### 查看时序报告

```bash
# CTS 后的时序
grep -A 20 "时序报告" logs/3_cts.log

# 最终时序
grep -A 20 "最终时序报告" logs/4_routing.log
```

### 关键指标

```bash
# Setup Slack (应该 > 0)
grep "worst slack" logs/4_routing.log

# TNS (应该 = 0)
grep "tns" logs/4_routing.log
```

## 📈 成功标准

### ✅ CTS 成功

- Clock Skew < 0.1 ns
- 插入了时钟缓冲器
- 生成了 results/3_cts.v

### ✅ 达到 100MHz

- Setup Slack > 0 ns
- Hold Slack > 0 ns
- TNS = 0 ns

## 🐛 故障排除

### 错误: 内存不足

```bash
# 增加 swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 错误: site core7 not found

已在配置中处理，如果仍有问题：
```bash
# 编辑 scripts/1_floorplan.tcl
# 注释掉 -site core7 行
```

### 进程卡住

```bash
# 检查内存使用
free -h

# 检查进程
ps aux | grep openroad
```

## 📚 输出文件说明

### DEF 文件
- 包含单元位置和连接信息
- 可用于物理验证

### Verilog 网表
- `3_cts.v`: 包含时钟缓冲器
- `4_routing.v`: 最终网表

### ODB 数据库
- OpenROAD 内部格式
- 用于步骤间传递数据

## 🎯 下一步

### 完成 P&R 后

1. **验证时序**: 确认达到 100MHz
2. **提取网表**: 使用 results/4_routing.v
3. **运行 STA**: 验证最终时序
4. **物理验证**: DRC/LVS 检查

### 运行最终 STA

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos

# 创建 STA 脚本使用 CTS 后的网表
cat > run_sta_final.tcl << 'EOF'
read_liberty ...
read_verilog openroad/results/3_cts.v
link_design asic_top
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]
report_checks -path_delay max
report_worst_slack
EOF

sta -exit run_sta_final.tcl
```

## 💡 提示

- 第一次运行建议使用 `./run.sh all`
- 可以在后台运行：`nohup ./run.sh all &`
- 定期检查日志文件
- 保存 results/ 目录

---

**准备就绪！执行命令**：
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
./run.sh all
```

**创建时间**: 2025-12-03 10:57
