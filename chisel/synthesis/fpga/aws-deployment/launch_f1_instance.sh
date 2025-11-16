#!/bin/bash
# 启动 AWS F1 实例脚本

set -e

echo "=== AWS F1 实例启动脚本 ==="
echo ""

# 配置参数
INSTANCE_TYPE="f1.2xlarge"
KEY_NAME="fpga-dev-key"
REGION="us-east-1"
AMI_NAME="FPGA Developer AMI"

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    echo "安装命令: pip install awscli"
    exit 1
fi

echo "✓ AWS CLI 已安装"

# 检查 AWS 凭证
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS 凭证未配置"
    echo "配置命令: aws configure"
    exit 1
fi

echo "✓ AWS 凭证已配置"
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "  账户 ID: $AWS_ACCOUNT"
echo ""

# 检查密钥对
echo "检查密钥对..."
if aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    echo "✓ 密钥对 '$KEY_NAME' 已存在"
else
    echo "⚠ 密钥对 '$KEY_NAME' 不存在，正在创建..."
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $REGION \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "✓ 密钥对已创建: ~/.ssh/${KEY_NAME}.pem"
fi
echo ""

# 查找 FPGA Developer AMI
echo "查找 FPGA Developer AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon aws-marketplace \
    --filters "Name=name,Values=*FPGA Developer AMI*" \
    --region $REGION \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo "⚠ 未找到 FPGA Developer AMI，尝试使用已知的 AMI..."
    # 使用最新的 FPGA Developer AMI (Ubuntu) - 1.17.0
    AMI_ID="ami-01198b89d80ebfdd2"
fi

echo "✓ 找到 AMI: $AMI_ID"
echo ""

# 检查 F1 实例配额
echo "检查 F1 实例配额..."
# 注意：这个命令可能需要特定权限
QUOTA_CHECK=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-85EED4F7 \
    --region $REGION \
    --query 'Quota.Value' \
    --output text 2>&1 || echo "0")

if [ "$QUOTA_CHECK" == "0" ] || [ "$QUOTA_CHECK" == "0.0" ]; then
    echo "⚠ F1 实例配额为 0，需要申请增加配额"
    echo "申请地址: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
    echo ""
    read -p "是否继续尝试启动实例？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ F1 实例配额: $QUOTA_CHECK"
fi
echo ""

# 获取默认 VPC 和子网
echo "获取网络配置..."
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --region $REGION \
    --query 'Vpcs[0].VpcId' \
    --output text)

if [ -z "$VPC_ID" ] || [ "$VPC_ID" == "None" ]; then
    echo "❌ 未找到默认 VPC"
    exit 1
fi

# 选择支持 F1 的可用区（us-east-1a, us-east-1b, us-east-1d, us-east-1e）
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    --region $REGION \
    --query 'Subnets[?AvailabilityZone==`us-east-1a` || AvailabilityZone==`us-east-1b` || AvailabilityZone==`us-east-1d` || AvailabilityZone==`us-east-1e`] | [0].SubnetId' \
    --output text)

if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" == "None" ]; then
    echo "❌ 未找到子网"
    exit 1
fi

echo "✓ VPC: $VPC_ID"
echo "✓ Subnet: $SUBNET_ID"
echo ""

# 创建或获取安全组
echo "配置安全组..."
SG_NAME="fpga-dev-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --region $REGION \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>&1 || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "创建安全组..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for FPGA development" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)
    
    # 添加 SSH 规则
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    echo "✓ 安全组已创建: $SG_ID"
else
    echo "✓ 使用现有安全组: $SG_ID"
fi
echo ""

# 启动实例
echo "启动 F1 实例..."
echo "  实例类型: $INSTANCE_TYPE"
echo "  AMI: $AMI_ID"
echo "  密钥对: $KEY_NAME"
echo "  安全组: $SG_ID"
echo ""

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --subnet-id $SUBNET_ID \
    --region $REGION \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=FPGA-Dev},{Key=Project,Value=RISC-V-AI-Accelerator}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ 实例启动失败"
    exit 1
fi

echo "✓ 实例已启动: $INSTANCE_ID"
echo ""

# 等待实例运行
echo "等待实例启动..."
aws ec2 wait instance-running \
    --instance-ids $INSTANCE_ID \
    --region $REGION

echo "✓ 实例正在运行"
echo ""

# 获取公网 IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              F1 实例启动成功！                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  公网 IP: $PUBLIC_IP"
echo "  区域: $REGION"
echo "  类型: $INSTANCE_TYPE"
echo ""
echo "连接命令:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem centos@${PUBLIC_IP}"
echo ""
echo "保存实例信息到文件..."
cat > ../build/f1_instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Region: $REGION
Instance Type: $INSTANCE_TYPE
Key Name: $KEY_NAME
Launch Time: $(date)
EOF

echo "✓ 实例信息已保存到: ../build/f1_instance_info.txt"
echo ""
echo "下一步:"
echo "  1. 等待 2-3 分钟让实例完全启动"
echo "  2. 连接到实例: ssh -i ~/.ssh/${KEY_NAME}.pem centos@${PUBLIC_IP}"
echo "  3. 运行环境配置: ./setup_aws.sh"
echo ""
echo "停止实例命令:"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "终止实例命令:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
