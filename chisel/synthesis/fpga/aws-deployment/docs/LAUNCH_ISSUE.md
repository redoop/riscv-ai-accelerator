# F1 实例启动问题

**日期**：2025-11-16  
**问题**：无法在 us-east-1 区域启动 f1.2xlarge 实例

## 问题描述

尝试启动 F1 实例时遇到可用区不支持的错误，但错误信息自相矛盾。

### 错误信息

```
An error occurred (Unsupported) when calling the RunInstances operation: 
Your requested instance type (f1.2xlarge) is not supported in your requested 
Availability Zone (us-east-1X). Please retry your request by not specifying 
an Availability Zone or choosing us-east-1a, us-east-1b, us-east-1d, us-east-1e.
```

### 已尝试的可用区

- ✗ us-east-1a - 提示选择 us-east-1b, us-east-1c, us-east-1d, us-east-1e
- ✗ us-east-1b - 提示选择 us-east-1a, us-east-1c, us-east-1d, us-east-1e  
- ✗ us-east-1c - 提示选择 us-east-1a, us-east-1b, us-east-1d, us-east-1e
- ✗ us-east-1d - 提示选择 us-east-1a, us-east-1b, us-east-1c, us-east-1e
- ✗ us-east-1e - 提示选择 us-east-1a, us-east-1b, us-east-1c, us-east-1d

### 验证信息

```bash
$ aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters Name=instance-type,Values=f1.2xlarge \
    --region us-east-1

# 结果显示 F1 在所有可用区都可用
us-east-1a, us-east-1b, us-east-1c, us-east-1d, us-east-1e
```

```bash
$ aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-85EED4F7 \
    --region us-east-1

# 配额充足
Value: 96.0
```

## 可能的原因

1. **临时容量不足**：所有可用区当前都没有 F1 实例容量
2. **账户限制**：可能需要额外的权限或审批
3. **AMI 兼容性**：FPGA Developer AMI 可能有特殊要求
4. **区域问题**：us-east-1 可能暂时不可用

## 解决方案

### 方案 1：使用 AWS 控制台（推荐）

通过 AWS 控制台手动启动实例可能会提供更详细的错误信息：

1. 登录 [AWS EC2 控制台](https://console.aws.amazon.com/ec2/)
2. 点击 "Launch Instance"
3. 选择 AMI: ami-0cb1b6ae2ff99f8bf (FPGA Developer AMI 1.18.0)
4. 选择实例类型: f1.2xlarge
5. 配置网络和安全组
6. 启动实例

### 方案 2：尝试其他区域

F1 实例在以下区域可用：
- us-east-1 (弗吉尼亚北部) - 当前问题
- us-west-2 (俄勒冈) - 需要订阅该区域的 AMI
- eu-west-1 (爱尔兰)
- ap-southeast-2 (悉尼)

### 方案 3：使用 Spot 实例

Spot 实例可能有更多容量：

```bash
aws ec2 request-spot-instances \
  --spot-price "0.50" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json \
  --region us-east-1
```

### 方案 4：联系 AWS 支持

如果问题持续，可以：
1. 创建支持工单
2. 说明尝试启动 F1 实例但所有可用区都失败
3. 提供账户 ID: 052613181120
4. 提供错误信息和尝试的命令

### 方案 5：等待并重试

容量问题通常是临时的，可以：
1. 等待几小时后重试
2. 在非高峰时段尝试（如凌晨）
3. 设置自动重试脚本

## 临时替代方案

### 使用模拟执行

查看完整的模拟执行流程：
```bash
cat chisel/synthesis/fpga/docs/SIMULATION_RUN.md
```

### 继续本地开发

在等待 F1 实例可用期间，可以：
1. 继续优化 Chisel 设计
2. 完善测试用例
3. 准备文档
4. 优化 FPGA 约束文件

## 下一步行动

**推荐**：使用 AWS 控制台手动启动实例

1. 访问 https://console.aws.amazon.com/ec2/
2. 尝试启动 f1.2xlarge 实例
3. 记录任何额外的错误信息
4. 如果成功，记录实例 ID 并继续验证流程

## 参考信息

- **账户 ID**: 052613181120
- **区域**: us-east-1
- **AMI ID**: ami-0cb1b6ae2ff99f8bf
- **实例类型**: f1.2xlarge
- **密钥对**: fpga-dev-key
- **安全组**: sg-03d27449f82b54360
- **VPC**: vpc-0282f8e2e326aeef2

---

**状态**：等待解决  
**更新时间**：2025-11-16 16:00
