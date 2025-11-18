# AWS FPGA 快速开始指南

## 🚀 一键启动（推荐）

```bash
cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
./run_fpga_flow.sh aws-launch
```

会自动提示选择 F1 或 F2 实例。

## ✅ 推荐流程（使用 F1）

### 1. 启动 F1 实例
```bash
./run_fpga_flow.sh aws-launch
# 选择选项 1 (F1 实例)
```

### 2. 生成 Verilog
```bash
./run_fpga_flow.sh prepare
```

### 3. 上传项目
```bash
./run_fpga_flow.sh aws-upload
```

### 4. 启动构建（2-4小时）
```bash
./run_fpga_flow.sh aws-build
```

### 5. 监控进度
```bash
./run_fpga_flow.sh aws-monitor
```

### 6. 下载 DCP
```bash
./run_fpga_flow.sh aws-download-dcp
```

### 7. 创建 AFI（30-60分钟）
```bash
./run_fpga_flow.sh aws-create-afi
```

### 8. 检查状态
```bash
./run_fpga_flow.sh status
```

### 9. 清理实例（重要！）
```bash
./run_fpga_flow.sh aws-cleanup
```

## 💰 成本估算

| 项目 | 时间 | 成本 (Spot) |
|------|------|-------------|
| F1 构建 | 2-4小时 | $1-2 |
| AFI 创建 | 30-60分钟 | 免费 |
| F1 测试 | 30分钟 | $0.25 |
| **总计** | | **$1.25-2.25** |

## ⚠️ 重要提醒

### ✅ 使用 F1
- 设备: xcvu9p
- 支持 AFI 创建
- 成本: ~$0.50/小时 (Spot)

### ❌ 避免 F2
- 设备: xcvu47p
- **不支持 AFI 创建**
- 成本: ~$2.30/小时 (Spot)
- 仅用于本地开发

## 🔍 检查当前状态

```bash
# 查看完整状态
./run_fpga_flow.sh status

# 仅查看 AFI 状态
cd aws-deployment
bash check_afi_status.sh
```

## 🐛 故障排除

### AFI 创建失败

```bash
# 1. 检查错误
./run_fpga_flow.sh status

# 2. 下载日志
cd aws-deployment
aws s3 ls s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/ --recursive

# 3. 查看 Vivado 日志
aws s3 cp s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/afi-*/\*_vivado.log vivado.log
grep -i error vivado.log
```

### 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| MANIFEST_NOT_FOUND | Manifest 格式错误 | 使用最新的 create_afi.sh |
| TOOL_VERSION_INVALID | Vivado 版本错误 | 使用 Vivado 2024.1 |
| DEVICE_MISMATCH | 使用了 F2 (xcvu47p) | 必须使用 F1 (xcvu9p) |
| TIMING_VIOLATION | 时序不收敛 | 优化设计或降低频率 |

## 📚 文档

- [完整工作流程](./COMPLETE_WORKFLOW.md)
- [F1 vs F2 对比](./F1_VS_F2_GUIDE.md)
- [设备不匹配问题](./DEVICE_MISMATCH_ISSUE.md)
- [AFI 创建成功](./AFI_CREATION_SUCCESS.md)

## 🆘 获取帮助

```bash
# 查看所有命令
./run_fpga_flow.sh help

# 查看 AFI 状态
./run_fpga_flow.sh status

# 查看实例信息
cat aws-deployment/.f1_instance_info
```

## 📞 支持

- GitHub Issues: https://github.com/aws/aws-fpga/issues
- AWS re:Post: https://repost.aws/tags/TAc7ofO5tbQRO57aX1lBYbjA/fpga-development
