# 下一步操作指南

**当前状态**：阶段 1 已完成 ✅，准备进入阶段 2

## 当前进度总结

### ✅ 已完成

1. **本地环境准备**
   - Java 11+、sbt、AWS CLI 已安装
   - Verilog 代码已生成（3765 行）
   - 本地测试全部通过（20/20）

2. **FPGA 适配层**
   - 顶层封装、时钟生成、IO 适配完成
   - 约束文件准备完毕

3. **测试脚本和文档**
   - 功能测试脚本、性能测试脚本
   - 完整文档（快速开始、本地测试、AWS 部署等）

4. **AWS 基础配置**
   - AWS CLI 已配置（账户 052613181120）
   - 区域设置：us-east-1
   - F1 实例可用性已确认

## 🎯 下一步：启动 AWS F1 实例

### 选项 1：自动化启动（推荐）

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f1_instance.sh
```

这个脚本会自动：
- 检查 AWS 配置
- 创建密钥对（如果不存在）
- 查找 FPGA Developer AMI
- 配置安全组
- 启动 f1.2xlarge 实例
- 保存实例信息

**预计时间**：5-10 分钟  
**预计费用**：$1.65/小时

### 选项 2：手动启动

#### 步骤 1：创建密钥对

```bash
aws ec2 create-key-pair \
  --key-name fpga-dev-key \
  --region us-east-1 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/fpga-dev-key.pem

chmod 400 ~/.ssh/fpga-dev-key.pem
```

#### 步骤 2：查找 AMI

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=FPGA Developer AMI*" \
  --region us-east-1 \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text
```

#### 步骤 3：启动实例

```bash
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type f1.2xlarge \
  --key-name fpga-dev-key \
  --region us-east-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FPGA-Dev}]'
```

#### 步骤 4：获取 IP 地址

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=FPGA-Dev" \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

### 选项 3：使用 AWS 控制台

1. 登录 AWS 控制台
2. 进入 EC2 服务
3. 点击"启动实例"
4. 搜索"FPGA Developer AMI"
5. 选择 f1.2xlarge 实例类型
6. 配置网络和安全组
7. 创建或选择密钥对
8. 启动实例

## ⚠️ 重要提醒

### 成本控制

- **f1.2xlarge**：$1.65/小时
- **建议**：完成工作后立即停止实例
- **停止命令**：`aws ec2 stop-instances --instance-ids <instance-id>`
- **终止命令**：`aws ec2 terminate-instances --instance-ids <instance-id>`

### F1 实例配额

如果启动失败并提示配额不足：

1. 访问 [Service Quotas 控制台](https://console.aws.amazon.com/servicequotas/)
2. 搜索 "EC2 F1"
3. 请求增加配额（建议至少 1 个实例）
4. 等待审批（通常 1-2 个工作日）

### 首次使用 FPGA Developer AMI

如果是首次使用，需要在 AWS Marketplace 订阅：

1. 访问 [AWS Marketplace](https://aws.amazon.com/marketplace/)
2. 搜索 "FPGA Developer AMI"
3. 点击"Continue to Subscribe"
4. 接受条款

## 📝 实例启动后的操作

### 1. 连接到实例

```bash
# 等待 2-3 分钟让实例完全启动
ssh -i ~/.ssh/fpga-dev-key.pem centos@<instance-ip>
```

### 2. 配置 AWS FPGA 环境

```bash
# 克隆 AWS FPGA 仓库
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh
source hdk_setup.sh
```

### 3. 上传项目代码

在本地机器上：

```bash
# 打包项目
cd /Users/tongxiaojun/github/riscv-ai-accelerator
tar czf fpga-project.tar.gz chisel/synthesis/fpga chisel/generated

# 上传到 F1 实例
scp -i ~/.ssh/fpga-dev-key.pem fpga-project.tar.gz centos@<instance-ip>:~/
```

在 F1 实例上：

```bash
# 解压项目
tar xzf fpga-project.tar.gz
cd chisel/synthesis/fpga/aws-deployment

# 运行环境配置
./setup_aws.sh
```

### 4. 开始 FPGA 构建

```bash
cd ../
vivado -mode batch -source scripts/build_fpga.tcl
```

**预计时间**：2-4 小时

## 📊 时间和成本估算

| 阶段 | 时间 | 成本 |
|------|------|------|
| 启动实例 | 5-10 分钟 | $0.03 |
| 环境配置 | 10-15 分钟 | $0.04 |
| Vivado 构建 | 2-4 小时 | $3.30-$6.60 |
| AFI 创建 | 30-60 分钟 | $0.83-$1.65 |
| 测试验证 | 10-20 分钟 | $0.28-$0.55 |
| **总计** | **3-5 小时** | **$4.48-$8.87** |

## 🔍 故障排查

### 问题 1：无法启动 F1 实例

**错误**：`InsufficientInstanceCapacity`

**解决方案**：
- 尝试不同的可用区（us-east-1a, us-east-1b, us-east-1c）
- 稍后重试
- 联系 AWS 支持

### 问题 2：配额不足

**错误**：`InstanceLimitExceeded`

**解决方案**：
- 申请增加 F1 实例配额
- 使用 Spot 实例（可节省 70% 成本）

### 问题 3：SSH 连接失败

**错误**：`Connection refused` 或 `Connection timed out`

**解决方案**：
- 等待 2-3 分钟让实例完全启动
- 检查安全组是否允许 SSH（端口 22）
- 确认使用正确的密钥文件
- 确认使用 `centos` 用户名

## 📚 参考文档

- [AWS_FPGA_PLAN.md](AWS_FPGA_PLAN.md) - 完整验证方案
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - 详细环境搭建
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - FPGA 构建指南
- [TEST_GUIDE.md](TEST_GUIDE.md) - 测试指南

## ✅ 准备就绪检查

在启动 F1 实例前，确认：

- [ ] AWS CLI 已配置并测试
- [ ] 了解 F1 实例费用（$1.65/小时）
- [ ] 准备好停止/终止实例的命令
- [ ] 已阅读 AWS_FPGA_PLAN.md
- [ ] 本地 Verilog 代码已生成
- [ ] 本地测试已全部通过

## 🚀 开始执行

如果一切准备就绪，执行：

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f1_instance.sh
```

或者按照手动步骤操作。

---

**提示**：如果暂时不想使用 AWS F1（因为费用），可以继续在本地进行开发和测试。本地测试已经验证了设计的功能正确性。
