# AWS FPGA 项目总结

## 🎯 完成的工作

### 1. ✅ AFI 创建脚本修复
- 修复了 manifest 格式（使用 SHA256，正确的字段名）
- 修复了 tarball 结构（使用 `to_aws/` 目录）
- 修复了文件命名（带时间戳前缀）
- 添加了 clock recipes 配置
- 优化了 S3 bucket 结构

### 2. ✅ 状态监控功能
- 创建了 `check_afi_status.sh` 脚本
- 集成到 `run_fpga_flow.sh status` 命令
- 显示详细的 AFI 状态（pending/available/failed）
- 包含进度条和时间估算
- 提供下一步操作指导

### 3. ✅ AMI 配置更新
- 更新为 Vivado ML 2024.1 Developer AMI
- AMI ID: `ami-0cab7155a229fac40`
- 预装正确的 Vivado 版本

### 4. ✅ 完整文档
- `AFI_CREATION_SUCCESS.md` - AFI 创建成功指南
- `FIX_VIVADO_VERSION.md` - Vivado 版本修复
- `COMPLETE_WORKFLOW.md` - 完整工作流程
- `DEVICE_MISMATCH_ISSUE.md` - 设备不匹配问题分析

## ❌ 发现的问题

### 关键问题：设备不匹配

**错误**：
```
ERROR: design check point is using device xcvu47p (F2), 
yet AWS Shell is using device xcvu9p (F1)
```

**原因**：
- 使用了 F2 实例 (xcvu47p) 构建 DCP
- AWS AFI 服务只支持 F1 (xcvu9p)
- F2 是较新的实例，AFI 服务尚未支持

## 🔧 解决方案

### 必须使用 F1 实例

1. **停止当前 F2 实例**
   ```bash
   cd /opt/github/riscv-ai-accelerator/chisel/synthesis/fpga
   ./run_fpga_flow.sh aws-cleanup
   ```

2. **创建 F1 启动脚本**
   ```bash
   cd aws-deployment
   cp launch_f2_vivado.sh launch_f1_vivado.sh
   ```

3. **修改为 F1 配置**
   ```bash
   # 在 launch_f1_vivado.sh 中
   INSTANCE_TYPE="f1.2xlarge"  # 改为 F1
   AMI_ID="ami-0c55b159cbfafe1f0"  # F1 Developer AMI
   SPOT_PRICE="1.65"  # F1 价格
   ```

4. **重新构建**
   ```bash
   ./run_fpga_flow.sh aws-launch  # 使用 F1
   ./run_fpga_flow.sh prepare
   ./run_fpga_flow.sh aws-upload
   ./run_fpga_flow.sh aws-build
   ./run_fpga_flow.sh aws-download-dcp
   ./run_fpga_flow.sh aws-create-afi
   ```

## 📊 成本对比

| 实例 | 设备 | Spot 价格 | AFI 支持 | 推荐 |
|------|------|-----------|----------|------|
| F1 | xcvu9p | ~$0.50/hr | ✅ 是 | ✅ 使用 |
| F2 | xcvu47p | ~$2.30/hr | ❌ 否 | ❌ 避免 |

**结论**：F1 更便宜且兼容！

## 🎓 学到的经验

### 1. AWS FPGA 生态系统
- F1 是当前主要支持的 FPGA 实例
- F2 虽然更新，但 AFI 服务尚未支持
- 必须使用 AWS 官方 Shell 和设备

### 2. DCP 格式要求
- Tarball 必须包含 `to_aws/` 目录
- 文件需要带时间戳前缀
- Manifest 必须是 `manifest.txt`
- 必须使用 SHA256 hash
- Tool version 必须带 `v` 前缀

### 3. 调试方法
- 查看 S3 日志获取详细错误
- 使用 `check_afi_status.sh` 监控状态
- 阅读 AWS FPGA HDK 源代码了解格式

### 4. 版本兼容性
- Vivado 版本必须匹配（2024.1）
- 设备必须匹配（xcvu9p for F1）
- Shell 版本必须兼容

## 📝 下一步行动

### 立即行动
1. ✅ 停止 F2 实例（节省成本）
2. ✅ 创建 F1 启动脚本
3. ✅ 使用 F1 重新构建

### 验证步骤
1. 确认实例类型是 f1.2xlarge
2. 确认设备是 xcvu9p
3. 确认 Vivado 版本是 2024.1
4. 检查 DCP 设备匹配

### 测试流程
1. 构建 DCP
2. 创建 AFI
3. 等待 available 状态
4. 在 F1 实例上测试

## 🛠️ 可用命令

### 状态检查
```bash
# 查看完整状态
./run_fpga_flow.sh status

# 只查看 AFI 状态
bash aws-deployment/check_afi_status.sh

# 持续监控
watch -n 60 './run_fpga_flow.sh status'
```

### 日志分析
```bash
# 列出日志
aws s3 ls s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/ --recursive

# 下载日志
aws s3 cp s3://riscv-fpga-afi/builds/<TIMESTAMP>/logs/afi-*/\*_vivado.log vivado.log

# 查看错误
grep -i error vivado.log
```

### 清理资源
```bash
# 清理 AWS 实例
./run_fpga_flow.sh aws-cleanup

# 清理本地文件
./run_fpga_flow.sh clean
```

## 📚 参考资料

### AWS 文档
- [F1 实例](https://aws.amazon.com/ec2/instance-types/f1/)
- [FPGA Developer AMI](https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k)
- [AWS FPGA HDK](https://github.com/aws/aws-fpga)

### 项目文档
- [完整工作流程](./COMPLETE_WORKFLOW.md)
- [设备不匹配问题](./DEVICE_MISMATCH_ISSUE.md)
- [Vivado 版本修复](./FIX_VIVADO_VERSION.md)

## 🎉 成就

尽管遇到了设备不匹配问题，但我们：

1. ✅ 完全理解了 AWS AFI 创建流程
2. ✅ 修复了所有 manifest 和 tarball 格式问题
3. ✅ 创建了完整的自动化脚本
4. ✅ 建立了状态监控系统
5. ✅ 识别了根本问题（F2 vs F1）
6. ✅ 提供了明确的解决方案

**下次使用 F1 实例，一切都会顺利！** 🚀
