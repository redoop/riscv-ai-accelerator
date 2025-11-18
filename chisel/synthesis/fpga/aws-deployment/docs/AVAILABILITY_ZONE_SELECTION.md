# 可用区自动选择机制

## 概述

所有 F1 启动脚本都实现了智能可用区自动选择机制，当某个可用区不可用时会自动尝试下一个。

## 工作原理

### 可用区列表

脚本按优先级顺序尝试以下可用区：

```bash
AVAILABILITY_ZONES=("us-east-1a" "us-east-1b" "us-east-1d" "us-east-1e" "us-east-1c")
```

### 选择流程

```
开始
  ↓
尝试 us-east-1a
  ↓
成功？
├─ 是 → 使用该可用区，结束
└─ 否 → 继续
  ↓
尝试 us-east-1b
  ↓
成功？
├─ 是 → 使用该可用区，结束
└─ 否 → 继续
  ↓
尝试 us-east-1d
  ↓
... (依此类推)
  ↓
所有可用区都失败？
└─ 是 → 显示错误信息，退出
```

## 改进功能

### 1. 详细错误信息

脚本现在会显示每个可用区失败的具体原因：

```bash
尝试可用区: us-east-1a
  ✗ us-east-1a 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1b
  ✓ 成功在 us-east-1b 启动实例: i-0123456789abcdef0
```

### 2. 智能错误检测

**按需实例 (launch_f1_ondemand.sh):**
- 检查 AWS CLI 退出码
- 验证实例 ID 格式 (i-xxxxxxxxx)
- 检测错误关键字
- 提取并显示错误消息

**Spot 实例 (launch_f1_vivado.sh):**
- 检查 Spot 请求 ID 格式 (sir-xxxxxxxx)
- 验证请求状态
- 获取失败原因 (Fault.Message)
- 自动取消失败的请求

### 3. API 限流保护

在尝试之间添加延迟，避免触发 AWS API 限流：

```bash
sleep 2  # 避免 API 限流
```

### 4. 自动清理

失败的 Spot 请求会自动取消，避免产生不必要的费用。

## 常见失败原因

### InsufficientInstanceCapacity

**含义**: 该可用区没有足够的 F1 实例容量

**解决方案**: 
- 脚本会自动尝试下一个可用区
- 如果所有可用区都失败，考虑：
  - 提高 Spot 出价（仅 Spot 实例）
  - 使用按需实例
  - 稍后重试

### Unsupported

**含义**: 该可用区不支持 F1 实例类型

**解决方案**: 
- 脚本会自动跳过该可用区
- 尝试其他可用区

### InvalidParameterValue

**含义**: 参数配置错误（如密钥、安全组等）

**解决方案**: 
- 检查配置参数
- 验证密钥存在: `aws ec2 describe-key-pairs --key-names fpga-f2-key`
- 验证安全组存在: `aws ec2 describe-security-groups --group-ids sg-03d27449f82b54360`

### RequestLimitExceeded

**含义**: API 请求频率过高

**解决方案**: 
- 脚本已添加延迟保护
- 如果仍然出现，等待几分钟后重试

## 使用示例

### 示例 1: 成功启动（第一个可用区）

```bash
$ ./launch_f1_ondemand.sh

尝试可用区: us-east-1a
  ✓ 成功在 us-east-1a 启动实例: i-0123456789abcdef0

✓ 实例 ID: i-0123456789abcdef0
✓ 可用区: us-east-1a
```

### 示例 2: 自动切换可用区

```bash
$ ./launch_f1_ondemand.sh

尝试可用区: us-east-1a
  ✗ us-east-1a 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1b
  ✗ us-east-1b 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1d
  ✓ 成功在 us-east-1d 启动实例: i-0123456789abcdef0

✓ 实例 ID: i-0123456789abcdef0
✓ 可用区: us-east-1d
```

### 示例 3: 所有可用区都失败

```bash
$ ./launch_f1_ondemand.sh

尝试可用区: us-east-1a
  ✗ us-east-1a 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1b
  ✗ us-east-1b 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1d
  ✗ us-east-1d 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1e
  ✗ us-east-1e 不可用
     原因: InsufficientInstanceCapacity
尝试可用区: us-east-1c
  ✗ us-east-1c 不可用
     原因: InsufficientInstanceCapacity

❌ 所有可用区都无法启动按需实例

可能的原因:
  1. 配额限制（F1 实例需要特殊配额）
  2. 账户权限问题
  3. F1 实例在该区域暂时不可用
  4. 安全组或密钥配置错误

建议:
  1. 检查 AWS 配额限制
  2. 请求增加 F1 配额
  3. 尝试其他区域（如 us-west-2）
  4. 联系 AWS 支持
```

## 手动指定可用区

如果需要手动指定可用区，可以修改脚本中的 `AVAILABILITY_ZONES` 数组：

```bash
# 只尝试 us-east-1a
AVAILABILITY_ZONES=("us-east-1a")

# 自定义顺序
AVAILABILITY_ZONES=("us-east-1d" "us-east-1a" "us-east-1b")
```

## 检查可用区状态

### 查看 F1 实例可用性

```bash
# 查看所有可用区
aws ec2 describe-availability-zones --region us-east-1

# 查看特定可用区
aws ec2 describe-availability-zones \
  --zone-names us-east-1a us-east-1b \
  --region us-east-1
```

### 查看实例类型可用性

```bash
# 查看 F1 实例在各可用区的可用性
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=f1.2xlarge \
  --region us-east-1 \
  --query 'InstanceTypeOfferings[*].[Location,InstanceType]' \
  --output table
```

### 查看 Spot 价格历史

```bash
# 查看各可用区的 Spot 价格
aws ec2 describe-spot-price-history \
  --instance-types f1.2xlarge \
  --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1 \
  --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice,Timestamp]' \
  --output table
```

## 最佳实践

1. **使用默认可用区列表**: 已按可用性优化排序
2. **不要频繁重试**: 如果失败，等待几分钟后再试
3. **监控成本**: 不同可用区的 Spot 价格可能不同
4. **保存实例信息**: 脚本会自动保存到 `.f1_instance_info`
5. **及时清理**: 使用完毕后立即停止实例

## 故障排除

### 问题: 脚本一直尝试但都失败

**检查步骤:**

1. 验证 AWS 凭证
```bash
aws sts get-caller-identity
```

2. 检查 F1 配额
```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-85EED4F7 \
  --region us-east-1
```

3. 验证密钥和安全组
```bash
aws ec2 describe-key-pairs --key-names fpga-f2-key --region us-east-1
aws ec2 describe-security-groups --group-ids sg-03d27449f82b54360 --region us-east-1
```

### 问题: Spot 请求一直处于 open 状态

**原因**: 当前没有可用的 Spot 容量

**解决方案**:
- 脚本会在 30 次尝试后超时
- 可以选择改用按需实例
- 或者稍后重试

### 问题: 按需实例也失败

**可能原因**:
- F1 配额为 0（需要申请）
- 账户权限不足
- 区域不支持 F1

**解决方案**:
- 申请 F1 配额
- 检查 IAM 权限
- 尝试其他区域（us-west-2）

## 相关文档

- [F1 启动选项说明](./F1_LAUNCH_OPTIONS.md)
- [F1 vs F2 快速参考](./F1_F2_QUICK_REFERENCE.md)
- [AWS 可用区文档](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)
