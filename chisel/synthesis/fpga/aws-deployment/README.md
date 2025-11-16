# AWS FPGA 部署包

本目录包含在 AWS F1 实例上部署 RISC-V AI 加速器所需的所有脚本和文档。

## 📁 目录内容

```
aws-deployment/
├── README.md              # 本文件
├── AWS_FPGA_PLAN.md      # 完整的 AWS FPGA 验证方案（必读）
├── setup_aws.sh          # AWS 环境配置脚本
└── create_afi.sh         # AFI 镜像创建脚本
```

## 🚀 快速开始

### 前提条件

1. **AWS 账户**：已配置 F1 实例访问权限
2. **F1 实例**：已启动并连接（推荐 f1.2xlarge）
3. **项目文件**：已上传到 F1 实例

### 部署步骤

#### 步骤 1：配置 AWS 环境

```bash
cd chisel/synthesis/fpga/aws-deployment
./setup_aws.sh
```

这个脚本会：
- 克隆 AWS FPGA 开发套件
- 配置环境变量
- 安装必要的工具
- 验证 Vivado 安装

#### 步骤 2：构建 FPGA 镜像

```bash
cd ..
vivado -mode batch -source scripts/build_fpga.tcl
```

构建时间：2-4 小时

#### 步骤 3：创建 AFI

```bash
cd aws-deployment
./create_afi.sh
```

这个脚本会：
- 创建 S3 bucket（如果不存在）
- 上传 DCP 文件到 S3
- 调用 AWS API 创建 AFI
- 保存 AFI 信息到 `../build/afi_info.txt`

AFI 生成时间：30-60 分钟

#### 步骤 4：等待 AFI 可用

```bash
# 检查 AFI 状态
AFI_ID=$(cat ../build/afi_info.txt | grep "AFI ID" | awk '{print $3}')
aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID

# 等待状态变为 "available"
# 可以使用以下命令持续监控：
watch -n 60 "aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID | grep State"
```

#### 步骤 5：加载并测试

```bash
# 加载 AFI
sudo fpga-load-local-image -S 0 -I $AFI_ID

# 验证加载
sudo fpga-describe-local-image -S 0 -H

# 运行测试
cd ../scripts
./test_processor_boot.sh
./test_compact_accel.sh
./benchmark_gops.sh
```

## 📚 文档说明

### AWS_FPGA_PLAN.md

这是最重要的文档，包含：

- **方案概述**：验证目标、AWS F1 优势、资源需求
- **实施方案**：开发流程、目录结构
- **技术方案**：时钟、复位、IO 映射、调试方案
- **实施步骤**：详细的操作步骤
- **测试计划**：功能测试、性能测试
- **成本估算**：详细的费用分析
- **风险与对策**：可能的问题和解决方案
- **时间计划**：5 周的详细计划

**建议**：在开始部署前，完整阅读此文档。

## 🔧 脚本说明

### setup_aws.sh

**功能**：配置 AWS F1 开发环境

**使用方法**：
```bash
./setup_aws.sh
source ~/.fpga_config  # 加载环境变量
```

**执行内容**：
- 检查系统要求
- 克隆 aws-fpga 仓库
- 配置 SDK 和 HDK
- 设置环境变量
- 验证工具安装

### create_afi.sh

**功能**：从 DCP 文件创建 AWS AFI 镜像

**使用方法**：
```bash
./create_afi.sh
```

**前提条件**：
- Vivado 构建已完成
- DCP 文件存在于 `../build/checkpoints/to_aws/SH_CL_routed.dcp`
- AWS CLI 已配置

**输出**：
- AFI ID 和 AGFI ID
- S3 bucket 信息
- 保存到 `../build/afi_info.txt`

## 💰 成本估算

### 实例费用

| 实例类型 | 价格（美东） | 推荐用途 |
|---------|------------|---------|
| f1.2xlarge | $1.65/小时 | 开发测试 |
| f1.4xlarge | $3.30/小时 | 性能测试 |
| f1.16xlarge | $13.20/小时 | 大规模设计 |

### 一次完整验证周期

| 阶段 | 时间 | 成本 |
|------|------|------|
| Vivado 构建 | 2-4 小时 | $3.30-$6.60 |
| AFI 创建 | 30-60 分钟 | $0.83-$1.65 |
| 测试验证 | 10-20 分钟 | $0.28-$0.55 |
| **总计** | **3-5 小时** | **$4.41-$8.80** |

### 存储费用

- S3 存储：约 $1/月
- EBS 卷（100GB）：约 $10/月

### 优化建议

1. **使用 Spot 实例**：可节省 70% 成本
2. **及时停止实例**：验证完成后立即停止
3. **复用 AFI**：同一设计的 AFI 可以重复使用
4. **批量验证**：积累多个修改后一次性验证

## ⚠️ 注意事项

### 权限要求

确保 AWS 账户有以下权限：
- EC2 F1 实例访问
- S3 bucket 创建和上传
- FPGA 镜像创建（ec2:CreateFpgaImage）

### 配额限制

检查 F1 实例配额：
```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-85EED4F7
```

如果配额为 0，需要在 AWS 控制台申请增加。

### 区域选择

推荐使用以下区域（F1 实例可用）：
- us-east-1（美国东部）
- us-west-2（美国西部）
- eu-west-1（欧洲）

### 数据安全

- DCP 文件包含设计信息，注意保护
- S3 bucket 建议设置为私有
- 使用完毕后可以删除 S3 中的文件

## 🐛 故障排查

### 问题 1：setup_aws.sh 失败

**症状**：
```
Error: aws-fpga repository not found
```

**解决方案**：
```bash
# 手动克隆
git clone https://github.com/aws/aws-fpga.git ~/aws-fpga
cd ~/aws-fpga
source sdk_setup.sh
source hdk_setup.sh
```

### 问题 2：create_afi.sh 找不到 DCP 文件

**症状**：
```
Error: DCP file not found
```

**解决方案**：
```bash
# 检查 DCP 文件位置
ls -la ../build/checkpoints/to_aws/

# 如果不存在，需要先运行 Vivado 构建
cd ..
vivado -mode batch -source scripts/build_fpga.tcl
```

### 问题 3：AFI 创建失败

**症状**：
```
AFI State: failed
```

**解决方案**：
```bash
# 查看 S3 日志
S3_BUCKET=$(cat ../build/afi_info.txt | grep "S3 Bucket" | awk '{print $3}')
aws s3 ls s3://$S3_BUCKET/logs/
aws s3 cp s3://$S3_BUCKET/logs/ ./logs/ --recursive
cat logs/*.log
```

### 问题 4：权限不足

**症状**：
```
UnauthorizedOperation: You are not authorized
```

**解决方案**：
```bash
# 检查 IAM 权限
aws iam get-user
aws iam list-attached-user-policies --user-name <your-username>

# 需要的权限：
# - AmazonEC2FullAccess
# - AmazonS3FullAccess
# - 或自定义策略包含 ec2:CreateFpgaImage
```

## 📞 获取帮助

### 文档资源

- **AWS_FPGA_PLAN.md**：完整的技术方案
- **../docs/BUILD_GUIDE.md**：构建指南
- **../docs/TEST_GUIDE.md**：测试指南
- **../docs/SETUP_GUIDE.md**：环境搭建

### 外部资源

- [AWS F1 官方文档](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/fpga-getting-started.html)
- [AWS FPGA GitHub](https://github.com/aws/aws-fpga)
- [AWS FPGA 论坛](https://forums.aws.amazon.com/forum.jspa?forumID=243)

### 命令速查

```bash
# 查看 AFI 状态
aws ec2 describe-fpga-images --fpga-image-ids <afi-id>

# 加载 AFI
sudo fpga-load-local-image -S 0 -I <afi-id>

# 查看 FPGA 状态
sudo fpga-describe-local-image -S 0 -H

# 清除 FPGA
sudo fpga-clear-local-image -S 0

# 查看实例类型
ec2-metadata --instance-type

# 查看可用区
ec2-metadata --availability-zone
```

## ✅ 检查清单

部署前检查：
- [ ] AWS 账户已配置
- [ ] F1 实例已启动
- [ ] 项目文件已上传
- [ ] AWS CLI 已安装并配置
- [ ] 有足够的 S3 存储空间
- [ ] 检查了 F1 实例配额

部署后验证：
- [ ] AFI 状态为 "available"
- [ ] FPGA 成功加载
- [ ] 处理器启动测试通过
- [ ] 加速器功能测试通过
- [ ] 性能达到目标（6.4 GOPS）

## 📊 预期结果

成功部署后，你应该看到：

```bash
$ sudo fpga-describe-local-image -S 0 -H
AFI          0       agfi-xxxxxxxxx  loaded            0        ok
AFIDEVICE    0       0x1d0f          0xf001      0000:00:1d.0

$ cd ../scripts
$ ./test_processor_boot.sh
✓ Processor started successfully
PASS: Processor boot test

$ ./benchmark_gops.sh
Performance: 6.4 GOPS
✓ PASS: Performance target met
```

---

**版本**：1.0  
**更新时间**：2025年11月16日  
**维护者**：redoop 团队
