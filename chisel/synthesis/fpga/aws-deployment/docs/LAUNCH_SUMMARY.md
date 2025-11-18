# F1 实例启动问题总结

## 问题

执行 `./launch_f1_ondemand.sh` 时，脚本显示"尝试可用区: us-east-1a"后停止，没有创建任何实例。

## 诊断过程

### 1. 检查配置
运行 `diagnose_f1_launch.sh` 发现所有配置都正确：
- ✓ AWS CLI 已安装
- ✓ 凭证有效
- ✓ F1 配额充足 (96.0)
- ✓ 密钥对存在
- ✓ 安全组存在
- ✓ AMI 可用

### 2. 测试实际启动
手动测试发现所有可用区都返回 "Unsupported" 错误：

```
An error occurred (Unsupported) when calling the RunInstances operation: 
Your requested instance type (f1.2xlarge) is not supported in your requested 
Availability Zone
```

### 3. 根本原因
**F1 实例当前在 us-east-1 区域完全不可用**

## 解决方案

### 立即可用的方案

#### 方案 A: 切换到 us-west-2 区域（推荐）

1. 查找 us-west-2 的 FPGA Developer AMI:
```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=FPGA Developer AMI*" \
  --region us-west-2 \
  --query 'Images[*].[ImageId,Name]' \
  --output table | head -10
```

2. 在 us-west-2 创建密钥对和安全组（如果还没有）

3. 修改脚本使用 us-west-2

#### 方案 B: 等待并重试

F1 容量可能在几小时或几天后恢复。

#### 方案 C: 联系 AWS 支持

请求帮助获取 F1 实例访问权限。

### 长期方案

创建多区域支持的启动脚本，自动尝试多个区域。

## 脚本改进

已对脚本进行以下改进：

### 1. 添加详细调试信息
```bash
echo "  调试: EXIT_CODE=$EXIT_CODE"
echo "  调试: OUTPUT=${INSTANCE_OUTPUT:0:100}..."
```

### 2. 添加超时机制
```bash
INSTANCE_OUTPUT=$(timeout 60 aws ec2 run-instances ...)
```

### 3. 改进错误提取
多种方式提取错误信息，确保用户能看到具体原因。

### 4. 验证实例信息文件
创建后验证文件是否成功写入。

### 5. 更新可用区列表
移除不支持 F1 的可用区（虽然当前所有区都不可用）。

## 文件清单

创建的诊断和文档文件：

1. `diagnose_f1_launch.sh` - 完整的诊断工具
2. `verify_instance_info.sh` - 验证实例信息文件
3. `F1_UNAVAILABLE_ISSUE.md` - 详细的问题说明和解决方案
4. `LAUNCH_SUMMARY.md` - 本文件，问题总结

## 下一步行动

### 选项 1: 切换区域（推荐）

```bash
# 1. 检查 us-west-2 的 F1 可用性
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=f1.2xlarge \
  --region us-west-2

# 2. 如果可用，修改脚本使用 us-west-2
# 3. 重新运行启动脚本
```

### 选项 2: 等待重试

```bash
# 设置定时任务，每小时检查一次
crontab -e
# 添加: 0 * * * * /path/to/diagnose_f1_launch.sh >> /tmp/f1_check.log 2>&1
```

### 选项 3: 使用其他实例类型

如果只是测试流程，可以暂时使用其他实例类型（如 c5.2xlarge）进行开发，等 F1 可用后再切换。

## 成本影响

由于无法启动实例，目前没有产生任何费用。

## 技术细节

### 为什么所有可用区都失败？

AWS 的 "Unsupported" 错误通常意味着：
1. 该实例类型在该可用区从未支持过
2. 或者临时容量不足

在我们的案例中，由于每个可用区的错误信息都建议其他可用区，这表明是**临时容量不足**而非永久不支持。

### 为什么 dry-run 成功但实际启动失败？

Dry-run 只检查权限和配置，不检查实际容量。这就是为什么 dry-run 通过但实际启动失败。

## 联系信息

如需帮助：
1. 查看 AWS 服务健康仪表板
2. 联系 AWS 支持
3. 在项目 issue 中报告问题

## 参考文档

- [F1_UNAVAILABLE_ISSUE.md](./F1_UNAVAILABLE_ISSUE.md) - 详细解决方案
- [F1_LAUNCH_OPTIONS.md](./F1_LAUNCH_OPTIONS.md) - 启动选项说明
- [AVAILABILITY_ZONE_SELECTION.md](./AVAILABILITY_ZONE_SELECTION.md) - 可用区选择机制
