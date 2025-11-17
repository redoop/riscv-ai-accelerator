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

1. **AWS 账户**：已配置 AWS CLI 和访问密钥
2. **SSH 密钥**：用于连接 F2 实例
3. **本地环境**：已安装 git 和 AWS CLI

### 自动化部署流程（推荐）

#### 步骤 1：启动 F2 实例（预装 Vivado）

```bash
cd chisel/synthesis/fpga/aws-deployment
./launch_f2_vivado.sh
```

这个脚本会：
- 请求 AWS F2 Spot 实例（节省 70% 成本）
- 使用预装 Vivado 2025.1 的 AMI
- 自动等待实例启动
- 保存实例信息到 `.f2_instance_info`（包含动态 IP）
- 显示连接命令

**输出示例**：
```
╔════════════════════════════════════════════════════════════╗
║         F2 实例启动成功（Vivado 预装）！                  ║
╚════════════════════════════════════════════════════════════╝

实例信息:
  实例 ID: i-0d552035c5aeda485
  公网 IP: 54.164.64.58
  
连接命令:
  ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@54.164.64.58
```

**重要**：实例信息会自动保存，后续脚本会自动读取，无需手动配置 IP。

#### 步骤 2：上传项目到 F2 实例

```bash
./upload_project.sh
```

这个脚本会：
- 自动读取实例 IP（从 `.f2_instance_info`）
- 打包项目文件（源码、生成的 Verilog、约束、脚本）
- 上传到 F2 实例
- 自动解压并验证

#### 步骤 3：启动 Vivado 构建

```bash
./start_build.sh
```

这个脚本会：
- 在 F2 实例上后台启动 Vivado 构建
- 构建时间：2-4 小时
- 生成 DCP 文件（用于创建 AFI）

#### 步骤 4：监控构建进度

```bash
# 智能监控（推荐）
./continuous_monitor.sh

# 或简单监控
./monitor_build.sh

# 或快速检查
./quick_status.sh
```

**智能监控功能**：
- 实时进度百分比
- 阶段自动检测（综合、布局、布线）
- 错误自动检测
- 完成自动退出

#### 步骤 5：创建 AFI

```bash
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

### 实例管理脚本

#### launch_f2_vivado.sh

**功能**：启动预装 Vivado 的 F2 Spot 实例

**使用方法**：
```bash
./launch_f2_vivado.sh
```

**特性**：
- 使用 Spot 实例（节省 70% 成本）
- 预装 Vivado 2025.1
- 自动保存实例信息到 `.f2_instance_info`
- 支持动态 IP 管理

**输出文件**：`.f2_instance_info`
```bash
INSTANCE_ID=i-0d552035c5aeda485
SPOT_REQUEST_ID=sir-b94fa4sm
PUBLIC_IP=54.164.64.58
KEY_NAME=fpga-f2-key
REGION=us-east-1
TIMESTAMP=2025-11-17 22:49:00
```

### 项目管理脚本

#### upload_project.sh

**功能**：上传项目到 F2 实例

**使用方法**：
```bash
./upload_project.sh
```

**自动功能**：
- 读取实例 IP（从 `.f2_instance_info`）
- 打包必要文件
- 上传并解压
- 验证项目结构

#### start_build.sh

**功能**：在 F2 实例上启动 Vivado 构建

**使用方法**：
```bash
./start_build.sh
```

**特性**：
- 后台运行（使用 nohup）
- 自动设置 Vivado 环境
- 生成构建日志

### 监控脚本

#### continuous_monitor.sh（推荐）

**功能**：智能持续监控构建进度

**使用方法**：
```bash
./continuous_monitor.sh
```

**特性**：
- 实时进度百分比和进度条
- 自动阶段检测（综合、布局、布线）
- 错误自动检测并退出
- 完成自动通知
- 记录关键里程碑

**输出示例**：
```
╔════════════════════════════════════════════════════════════╗
║          FPGA 构建持续监控 - 迭代 #42                     ║
╚════════════════════════════════════════════════════════════╝

⏱️  运行时间: 125 分 30 秒
🕐 当前时间: 16:45:30

📊 Vivado 进程: 1 个运行中

📈 构建进度: 65%
[████████████████████████████████░░░░░░░░░░░░░░░░░░] 65%

🔧 当前阶段: Placement Complete
   阶段用时: 45分12秒
```

#### monitor_build.sh

**功能**：简单监控构建状态

**使用方法**：
```bash
./monitor_build.sh
```

#### quick_status.sh

**功能**：快速检查构建状态

**使用方法**：
```bash
./quick_status.sh
```

#### check_status.sh

**功能**：单次状态检查

**使用方法**：
```bash
./check_status.sh
```

### AFI 管理脚本

#### create_afi.sh

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

### 环境配置脚本

#### setup_aws.sh

**功能**：配置 AWS FPGA 开发环境

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

#### setup_f2_environment.sh

**功能**：在 F2 实例上设置开发环境

**使用方法**：
```bash
# 在 F2 实例上运行
./setup_f2_environment.sh
```

**安装内容**：
- Java 11
- sbt（Scala Build Tool）
- 验证 Vivado

## 💰 成本估算

### F2 实例费用（推荐）

| 实例类型 | 按需价格 | Spot 价格 | 推荐用途 |
|---------|---------|----------|---------|
| f2.6xlarge | $3.30/小时 | ~$1.00/小时 | Vivado 构建 |
| f2.16xlarge | $13.20/小时 | ~$4.00/小时 | 大规模设计 |

**我们使用 Spot 实例，节省约 70% 成本！**

### F1 实例费用（测试用）

| 实例类型 | 价格（美东） | 推荐用途 |
|---------|------------|---------|
| f1.2xlarge | $1.65/小时 | 开发测试 |
| f1.4xlarge | $3.30/小时 | 性能测试 |

### 一次完整验证周期

| 阶段 | 实例类型 | 时间 | Spot 成本 |
|------|---------|------|----------|
| Vivado 构建 | F2.6xlarge | 2-4 小时 | $2.00-$4.00 |
| AFI 创建 | 无需实例 | 30-60 分钟 | $0 |
| 测试验证 | F1.2xlarge | 10-20 分钟 | $0.28-$0.55 |
| **总计** | - | **3-5 小时** | **$2.28-$4.55** |

**使用 Spot 实例比按需实例节省 ~$5！**

### 存储费用

- S3 存储：约 $0.50/月（DCP 文件）
- EBS 卷（100GB）：约 $10/月（可选）

### 优化建议

1. ✅ **使用 Spot 实例**：已集成在 `launch_f2_vivado.sh` 中
2. ✅ **自动化流程**：减少人工操作时间
3. ✅ **动态 IP 管理**：无需手动配置
4. **及时停止实例**：构建完成后立即停止
   ```bash
   aws ec2 terminate-instances --instance-ids <instance-id> --region us-east-1
   ```
5. **复用 AFI**：同一设计的 AFI 可以重复使用
6. **批量验证**：积累多个修改后一次性验证

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

### 问题 1：未找到实例信息文件

**症状**：
```
❌ 错误: 未找到实例信息文件
请先运行: ./launch_f2_vivado.sh
```

**原因**：脚本需要读取 `.f2_instance_info` 文件获取实例 IP

**解决方案**：
```bash
# 先启动 F2 实例
./launch_f2_vivado.sh

# 或者手动创建实例信息文件
cat > .f2_instance_info << EOF
INSTANCE_ID=i-xxxxxxxxx
PUBLIC_IP=xx.xx.xx.xx
KEY_NAME=fpga-f2-key
REGION=us-east-1
EOF
```

### 问题 2：SSH 连接失败

**症状**：
```
Permission denied (publickey)
```

**解决方案**：
```bash
# 检查密钥文件权限
chmod 400 ~/.ssh/fpga-f2-key.pem

# 检查密钥是否存在
ls -la ~/.ssh/fpga-f2-key.pem

# 如果密钥不存在，需要重新创建
aws ec2 delete-key-pair --key-name fpga-f2-key --region us-east-1
aws ec2 create-key-pair --key-name fpga-f2-key --region us-east-1 \
  --query 'KeyMaterial' --output text > ~/.ssh/fpga-f2-key.pem
chmod 400 ~/.ssh/fpga-f2-key.pem
```

### 问题 3：Spot 实例请求失败

**症状**：
```
❌ Spot 请求失败
```

**解决方案**：
```bash
# 检查 Spot 价格历史
aws ec2 describe-spot-price-history \
  --instance-types f2.6xlarge \
  --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --product-descriptions "Linux/UNIX" \
  --region us-east-1

# 提高 Spot 出价（在 launch_f2_vivado.sh 中修改 SPOT_PRICE）
# 或使用按需实例
```

### 问题 4：Vivado 构建失败

**症状**：
```
ERROR: [DRC NSTD-1] Unspecified I/O Standard
```

**原因**：这是正常的，我们只生成 DCP，不生成比特流

**解决方案**：
```bash
# 检查 DCP 文件是否生成
ssh -i ~/.ssh/fpga-f2-key.pem ubuntu@<ip> \
  'ls -lh ~/riscv-ai-accelerator/chisel/synthesis/fpga/build/checkpoints/to_aws/'

# 如果 DCP 存在，构建实际上是成功的
```

### 问题 5：create_afi.sh 找不到 DCP 文件

**症状**：
```
Error: DCP file not found
```

**解决方案**：
```bash
# 检查 DCP 文件位置
ls -la ../build/checkpoints/to_aws/

# 从 F2 实例下载 DCP
scp -i ~/.ssh/fpga-f2-key.pem \
  ubuntu@<ip>:~/riscv-ai-accelerator/chisel/synthesis/fpga/build/checkpoints/to_aws/SH_CL_routed.dcp \
  ../build/checkpoints/to_aws/
```

### 问题 6：AFI 创建失败

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

### 问题 7：权限不足

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
# - 或自定义策略包含 ec2:CreateFpgaImage, ec2:RequestSpotInstances
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
