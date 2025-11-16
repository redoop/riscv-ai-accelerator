# AWS FPGA 环境搭建指南

## 一、AWS 账户准备

### 1.1 创建 AWS 账户

1. 访问 https://aws.amazon.com/
2. 点击 "创建 AWS 账户"
3. 填写账户信息和支付方式
4. 完成身份验证

### 1.2 申请 F1 实例访问权限

F1 实例默认限额为 0，需要申请：

1. 登录 AWS 控制台
2. 进入 Service Quotas
3. 搜索 "EC2 F1"
4. 请求增加限额（建议至少 1 个实例）
5. 等待审批（通常 1-2 个工作日）

### 1.3 配置 IAM 权限

创建 IAM 用户并授予以下权限：
- EC2 完全访问
- S3 完全访问
- CloudWatch 日志访问

## 二、本地环境配置

### 2.1 安装 AWS CLI

```bash
# macOS
brew install awscli

# Linux
pip install awscli

# 验证安装
aws --version
```

### 2.2 配置 AWS 凭证

```bash
aws configure
# 输入：
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (us-east-1 或 us-west-2)
# - Default output format (json)
```

### 2.3 创建密钥对

```bash
# 创建 EC2 密钥对
aws ec2 create-key-pair \
  --key-name fpga-dev-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/fpga-dev-key.pem

# 设置权限
chmod 400 ~/.ssh/fpga-dev-key.pem
```

## 三、启动 F1 实例

### 3.1 选择 AMI

使用 AWS FPGA Developer AMI：
```bash
# 查找最新的 FPGA Developer AMI
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=FPGA Developer AMI*" \
  --query 'Images[0].ImageId' \
  --output text
```

### 3.2 启动实例

```bash
# 启动 f1.2xlarge 实例
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type f1.2xlarge \
  --key-name fpga-dev-key \
  --security-group-ids default \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FPGA-Dev}]'

# 获取实例 IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=FPGA-Dev" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

### 3.3 连接实例

```bash
ssh -i ~/.ssh/fpga-dev-key.pem centos@<instance-ip>
```

## 四、F1 实例环境配置

### 4.1 克隆 AWS FPGA 仓库

```bash
cd ~
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh
source hdk_setup.sh
```

### 4.2 安装开发工具

```bash
# 更新系统
sudo yum update -y

# 安装必要工具
sudo yum install -y git gcc make python3 python3-pip

# 安装 Python 依赖
pip3 install boto3 pyyaml
```

### 4.3 验证 Vivado

```bash
# 检查 Vivado 版本
vivado -version

# 应该显示 Vivado 2021.2 或更高版本
```

## 五、项目部署

### 5.1 上传项目代码

```bash
# 从本地上传
scp -i ~/.ssh/fpga-dev-key.pem -r \
  chisel/synthesis/fpga \
  centos@<instance-ip>:~/

# 或从 Git 克隆
ssh -i ~/.ssh/fpga-dev-key.pem centos@<instance-ip>
git clone <your-repo>
```

### 5.2 生成 Verilog

```bash
cd ~/chisel
sbt "runMain edgeai.SimpleEdgeAiSoCMain"
```

### 5.3 运行环境配置脚本

```bash
cd ~/chisel/synthesis/fpga
./scripts/setup_aws.sh
source ~/.fpga_config
```

## 六、故障排除

### 6.1 无法启动 F1 实例

**问题**：InsufficientInstanceCapacity 错误

**解决**：
- 尝试不同的可用区
- 使用 f1.2xlarge 而非 f1.16xlarge
- 联系 AWS 支持

### 6.2 Vivado 未找到

**问题**：vivado: command not found

**解决**：
```bash
source /opt/Xilinx/Vivado/2021.2/settings64.sh
```

### 6.3 权限错误

**问题**：Permission denied

**解决**：
```bash
chmod +x scripts/*.sh
sudo usermod -a -G fpga $USER
```

## 七、成本优化

### 7.1 使用 Spot 实例

```bash
# 请求 Spot 实例（可节省 70% 成本）
aws ec2 request-spot-instances \
  --spot-price "0.50" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

### 7.2 自动停止实例

```bash
# 设置自动停止（4 小时后）
aws ec2 stop-instances --instance-ids <instance-id>
```

### 7.3 删除不用的资源

```bash
# 删除旧的 AFI
aws ec2 delete-fpga-image --fpga-image-id afi-xxxxxxxxx

# 清理 S3 bucket
aws s3 rm s3://your-bucket --recursive
```

## 八、下一步

环境搭建完成后，请参考：
- `BUILD_GUIDE.md` - 构建 FPGA 镜像
- `TEST_GUIDE.md` - 运行测试
- `AWS_FPGA_PLAN.md` - 完整验证方案
