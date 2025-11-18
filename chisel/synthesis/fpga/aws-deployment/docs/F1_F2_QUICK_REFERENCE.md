# F1 vs F2 快速参考

## 🚀 快速命令

### 使用 F1（推荐）
```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga

# 方式 1：直接指定 F1
./run_fpga_flow.sh aws-launch f1

# 方式 2：使用 F1 目录
cd aws-deployment/f1
bash launch.sh
```

### 使用 F2（不推荐）
```bash
# 方式 1：直接指定 F2
./run_fpga_flow.sh aws-launch f2

# 方式 2：使用 F2 目录
cd aws-deployment/f2
bash launch.sh
```

### 交互式选择
```bash
./run_fpga_flow.sh aws-launch
# 会提示选择 F1 或 F2
```

## 📊 对比表

| 特性 | F1 | F2 |
|------|----|----|
| **设备** | xcvu9p | xcvu47p |
| **LUTs** | 2.5M | 9M |
| **AFI 支持** | ✅ 是 | ❌ 否 |
| **Spot 价格** | ~$0.50/hr | ~$2.30/hr |
| **推荐用途** | AFI 创建 | 仅开发 |

## 🎯 决策树

```
需要创建 AFI？
├─ 是 → 使用 F1 ✅
└─ 否 → 
    ├─ 需要 9M LUTs？
    │   ├─ 是 → 使用 F2（但无法部署）
    │   └─ 否 → 使用 F1（更便宜）
    └─ 仅本地开发？→ 使用 F1（更便宜）
```

## 📁 目录结构

```
aws-deployment/
├── f1/                    # F1 实例流程
│   ├── README.md
│   ├── launch.sh         → ../launch_f1_vivado.sh
│   ├── upload.sh         → ../upload_project.sh
│   ├── build.sh          → ../start_build.sh
│   ├── monitor.sh        → ../continuous_monitor.sh
│   ├── download_dcp.sh   # F1 专用下载
│   ├── create_afi.sh     → ../create_afi.sh
│   └── cleanup.sh        → ../cleanup_fpga_instances.sh
│
├── f2/                    # F2 实例流程（不推荐）
│   ├── README.md
│   └── ... (类似 F1)
│
├── launch_f1_vivado.sh    # F1 启动脚本
├── launch_f2_vivado.sh    # F2 启动脚本
└── launch_fpga_instance.sh # 交互式选择
```

## 💡 使用建议

### 场景 1：创建 AFI 并部署
```bash
# 必须使用 F1
./run_fpga_flow.sh aws-launch f1
./run_fpga_flow.sh prepare
./run_fpga_flow.sh aws-upload
./run_fpga_flow.sh aws-build
./run_fpga_flow.sh aws-download-dcp
./run_fpga_flow.sh aws-create-afi
./run_fpga_flow.sh aws-cleanup
```

### 场景 2：本地开发（不需要 AFI）
```bash
# 推荐使用 F1（更便宜）
./run_fpga_flow.sh aws-launch f1
# ... 开发和测试
./run_fpga_flow.sh aws-cleanup
```

### 场景 3：大型设计（需要 9M LUTs）
```bash
# 使用 F2，但无法创建 AFI
./run_fpga_flow.sh aws-launch f2
# ... 开发和测试
# ❌ 无法执行 aws-create-afi
./run_fpga_flow.sh aws-cleanup
```

## ⚠️ 常见错误

### 错误 1：使用 F2 DCP 创建 AFI
```
ERROR: device xcvu47p vs xcvu9p mismatch
```
**解决**: 必须使用 F1 重新构建

### 错误 2：忘记清理实例
```
成本: $2.30/hr × 24hr = $55.20
```
**解决**: 构建完成后立即运行 `aws-cleanup`

## 🔧 故障排除

### 检查当前使用的实例类型
```bash
# 查看实例信息文件
cat aws-deployment/.f1_instance_info
# 或
cat aws-deployment/.f2_instance_info
```

### 验证 DCP 设备
```bash
cd build/checkpoints/to_aws
unzip -p SH_CL_routed.dcp dcp.xml | grep -o 'xcvu[0-9]*p'
# 应该显示: xcvu9p (F1) 或 xcvu47p (F2)
```

### 切换实例类型
```bash
# 清理当前实例
./run_fpga_flow.sh aws-cleanup

# 启动新实例
./run_fpga_flow.sh aws-launch f1  # 或 f2
```

## 📞 获取帮助

```bash
# 查看完整帮助
./run_fpga_flow.sh help

# 查看 F1 文档
cat aws-deployment/f1/README.md

# 查看 F2 文档
cat aws-deployment/f2/README.md

# 查看详细对比
cat aws-deployment/F1_VS_F2_GUIDE.md
```

## 💰 成本计算器

### F1 流程（4小时构建 + 30分钟测试）
```
构建: 4hr × $0.50 = $2.00
测试: 0.5hr × $0.50 = $0.25
总计: $2.25
```

### F2 流程（4小时构建，无法测试）
```
构建: 4hr × $2.30 = $9.20
AFI: 不支持
总计: $9.20（且无法部署）
```

**结论**: F1 更便宜且功能完整！
