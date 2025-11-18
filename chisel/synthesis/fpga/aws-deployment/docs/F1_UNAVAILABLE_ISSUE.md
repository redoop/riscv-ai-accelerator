# F1 实例不可用问题

## 问题描述

在尝试启动 F1 实例时，所有可用区都返回 "Unsupported" 错误：

```
An error occurred (Unsupported) when calling the RunInstances operation: 
Your requested instance type (f1.2xlarge) is not supported in your requested 
Availability Zone (us-east-1x). Please retry your request by not specifying 
an Availability Zone or choosing...
```

## 根本原因

这个错误表明 **F1 实例当前在 us-east-1 区域的所有可用区都不可用**。

可能的原因：
1. **临时容量不足**: AWS 在该区域暂时没有 F1 实例容量
2. **区域限制**: F1 实例可能在某些区域有限制
3. **账户限制**: 某些新账户可能需要特殊批准才能使用 F1

## 诊断结果

运行 `diagnose_f1_launch.sh` 显示：
- ✓ AWS CLI 配置正确
- ✓ 凭证有效
- ✓ F1 配额充足 (96.0)
- ✓ 密钥和安全组存在
- ✓ AMI 可用
- ❌ 但实际启动时所有可用区都失败

## 解决方案

### 方案 1: 尝试其他区域（推荐）

F1 实例在以下区域通常更容易获得：

#### us-west-2 (俄勒冈)
```bash
# 修改脚本中的 REGION
REGION="us-west-2"

# 或使用环境变量
export AWS_DEFAULT_REGION=us-west-2
./launch_f1_ondemand.sh
```

#### eu-west-1 (爱尔兰)
```bash
REGION="eu-west-1"
```

#### ap-southeast-1 (新加坡)
```bash
REGION="ap-southeast-1"
```

### 方案 2: 使用 Spot 实例

Spot 实例有时比按需实例更容易获得：

```bash
./launch_f1_vivado.sh  # 使用 Spot 实例
```

### 方案 3: 稍后重试

F1 容量可能在几小时或几天后恢复。设置提醒稍后重试。

### 方案 4: 联系 AWS 支持

如果问题持续存在，联系 AWS 支持：

1. 访问 AWS Support Center
2. 创建案例
3. 选择 "Service Limit Increase"
4. 说明需要 F1 实例访问权限

## 修改脚本使用其他区域

### 临时修改

编辑 `launch_f1_ondemand.sh`:

```bash
# 将这一行
REGION="us-east-1"

# 改为
REGION="us-west-2"
```

同时修改可用区列表：

```bash
# us-west-2 的可用区
AVAILABILITY_ZONES=("us-west-2a" "us-west-2b" "us-west-2c" "us-west-2d")
```

### 创建区域特定脚本

为不同区域创建单独的启动脚本：

```bash
# 复制脚本
cp launch_f1_ondemand.sh launch_f1_ondemand_uswest2.sh

# 编辑新脚本
vim launch_f1_ondemand_uswest2.sh
# 修改 REGION 和 AVAILABILITY_ZONES
```

## 检查其他区域的 F1 可用性

```bash
# 检查 us-west-2
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=f1.2xlarge \
  --region us-west-2 \
  --query 'InstanceTypeOfferings[*].Location' \
  --output table

# 检查 eu-west-1
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=f1.2xlarge \
  --region eu-west-1 \
  --query 'InstanceTypeOfferings[*].Location' \
  --output table
```

## 验证新区域配置

在切换区域后，运行诊断：

```bash
# 修改 diagnose_f1_launch.sh 中的 REGION
vim diagnose_f1_launch.sh

# 运行诊断
./diagnose_f1_launch.sh
```

## 注意事项

### 1. AMI 区域特定

AMI ID 在不同区域是不同的。需要找到目标区域的 FPGA Developer AMI：

```bash
# 查找 us-west-2 的 FPGA Developer AMI
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=FPGA Developer AMI*" \
  --region us-west-2 \
  --query 'Images[*].[ImageId,Name,CreationDate]' \
  --output table | sort -k3 -r | head -5
```

### 2. 安全组和密钥对

安全组和密钥对也是区域特定的，需要在新区域创建：

```bash
# 在新区域创建密钥对
aws ec2 create-key-pair \
  --key-name fpga-f2-key \
  --region us-west-2 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/fpga-f2-key-uswest2.pem

chmod 400 ~/.ssh/fpga-f2-key-uswest2.pem

# 创建安全组
aws ec2 create-security-group \
  --group-name fpga-dev-sg \
  --description "FPGA Development Security Group" \
  --region us-west-2

# 添加 SSH 规则
aws ec2 authorize-security-group-ingress \
  --group-name fpga-dev-sg \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0 \
  --region us-west-2
```

### 3. 数据传输成本

跨区域数据传输会产生额外费用。如果需要频繁传输大文件，选择地理位置较近的区域。

## 当前状态

**us-east-1 区域 F1 实例状态: ❌ 不可用**

建议：
1. 切换到 us-west-2 区域
2. 或等待 us-east-1 容量恢复
3. 或联系 AWS 支持

## 相关文档

- [AWS 区域和可用区](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)
- [F1 实例](https://aws.amazon.com/ec2/instance-types/f1/)
- [FPGA Developer AMI](https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k)

## 更新日志

- 2024-11-18: 发现 us-east-1 所有可用区都不支持 F1
- 建议使用 us-west-2 作为替代方案
