# F1 清理总结

## 背景

AWS F1 实例已于 2024 年退役，本项目已全面迁移到 F2 实例。

## 已删除的 F1 专用文件

### 脚本文件
- `check_f1_availability.sh` - F1 可用性检查脚本
- `diagnose_f1_launch.sh` - F1 启动诊断脚本
- `launch_f1_ondemand.sh` - F1 按需实例启动脚本
- `launch_f1_vivado.sh` - F1 Vivado 实例启动脚本

### 文档文件
- `docs/F1_CAPACITY_ISSUE.md` - F1 容量问题文档
- `docs/F1_UNAVAILABLE_ISSUE.md` - F1 不可用问题文档
- `docs/F1_LAUNCH_OPTIONS.md` - F1 启动选项文档
- `docs/F1_VS_F2_COMPARISON.md` - F1 vs F2 对比文档
- `docs/F1_VS_F2_GUIDE.md` - F1 vs F2 指南
- `docs/F1_F2_QUICK_REFERENCE.md` - F1/F2 快速参考

## 已更新的文件

### 主要脚本
1. **run_fpga_flow.sh**
   - 移除所有 F1 相关选项
   - 简化为仅支持 F2
   - 更新帮助文档

2. **launch_fpga_instance.sh**
   - 完全重写，仅支持 F2
   - 添加 F1 退役说明
   - 简化用户交互

3. **cleanup_fpga_instances.sh**
   - 更新为仅清理 F2 实例
   - 移除 F1 实例过滤器

4. **test_aws_cli.sh**
   - 更新配额检查为 F2
   - 更新实例类型检查为 F2
   - 添加 F1 退役说明

5. **check_ami_subscription.sh**
   - 更新测试实例类型为 f2.6xlarge
   - 更新启动命令建议

6. **check_afi_status.sh**
   - 更新加载说明为 F2 实例

7. **setup_aws.sh**
   - 更新实例类型检查为 F2

8. **verify_instance_info.sh**
   - 完全重写为 F2 版本
   - 更新文件路径为 `.f2_instance_info`

### 文档
1. **README.md**
   - 移除所有 F1 引用
   - 更新为 F2 专用文档
   - 添加 F1 退役说明
   - 更新成本估算
   - 更新配额检查说明

## 当前架构

### 实例类型
- **主要**: F2 (f2.6xlarge, f2.16xlarge)
- **设备**: xcvu47p (Virtex UltraScale+ VU47P)
- **用途**: FPGA 开发和测试

### 文件命名
- 实例信息文件: `.f2_instance_info`
- 启动脚本: `launch_f2_vivado.sh`
- 环境设置: `setup_f2_environment.sh`

### 成本优化
- 使用 Spot 实例（节省 ~70%）
- F2.6xlarge Spot: ~$1.00/小时
- 典型构建: 2-4 小时 = $2-4

## 迁移指南

### 对于现有用户

如果你之前使用 F1 实例：

1. **删除旧的实例信息文件**
   ```bash
   rm .f1_instance_info
   ```

2. **使用新的 F2 启动脚本**
   ```bash
   ./launch_f2_vivado.sh
   ```

3. **更新自动化脚本**
   - 将 `f1.2xlarge` 替换为 `f2.6xlarge`
   - 将 `.f1_instance_info` 替换为 `.f2_instance_info`

### 命令对照表

| 旧命令 (F1) | 新命令 (F2) |
|------------|------------|
| `./launch_f1_vivado.sh` | `./launch_f2_vivado.sh` |
| `./launch_f1_ondemand.sh` | `./launch_f2_vivado.sh` |
| `./check_f1_availability.sh` | (已删除) |
| `./diagnose_f1_launch.sh` | (已删除) |

## 注意事项

1. **AFI 创建**: F2 生成的 DCP 文件可能与 AWS AFI 服务不兼容，因为设备类型不同（xcvu47p vs xcvu9p）

2. **成本**: F2 实例比 F1 更贵，但提供更多资源（9M LUTs vs 2.5M LUTs）

3. **可用性**: F2 Spot 实例可能因容量不足而失败，建议在多个可用区尝试

## 验证清单

- [x] 删除所有 F1 专用脚本
- [x] 删除所有 F1 专用文档
- [x] 更新所有脚本中的 F1 引用
- [x] 更新所有文档中的 F1 引用
- [x] 更新实例类型为 F2
- [x] 更新文件路径为 F2
- [x] 添加 F1 退役说明
- [x] 更新成本估算
- [x] 简化用户界面

## 相关资源

- [AWS F2 实例文档](https://aws.amazon.com/ec2/instance-types/f2/)
- [F2 实例指南](docs/F2_VIVADO_GUIDE.md)
- [F2 实例信息](docs/F2_INSTANCE_INFO.md)

---

**更新日期**: 2025-11-19  
**版本**: 2.0  
**状态**: F1 清理完成，全面迁移到 F2
