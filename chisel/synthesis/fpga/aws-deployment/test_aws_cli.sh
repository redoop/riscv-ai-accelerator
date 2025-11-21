#!/bin/bash
# 测试 AWS CLI 配置和权限

echo "=== AWS CLI 配置测试 ==="
echo ""

# 检查 AWS CLI 是否安装
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    echo ""
    echo "安装方法:"
    echo "  pip install awscli"
    echo "  或"
    echo "  curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'"
    echo "  unzip awscliv2.zip"
    echo "  sudo ./aws/install"
    exit 1
fi

echo "✓ AWS CLI 已安装"
aws --version
echo ""

# 检查 AWS 凭证
echo "=== 检查 AWS 凭证 ==="
if aws sts get-caller-identity &> /dev/null; then
    echo "✓ AWS 凭证配置正确"
    aws sts get-caller-identity
else
    echo "❌ AWS 凭证未配置或无效"
    echo ""
    echo "配置方法:"
    echo "  aws configure"
    echo ""
    echo "需要提供:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    echo "  - Default region (us-east-1)"
    echo "  - Default output format (json)"
    exit 1
fi

echo ""

# 检查区域配置
echo "=== 检查区域配置 ==="
REGION=$(aws configure get region)
if [ -z "$REGION" ]; then
    echo "⚠️  未设置默认区域"
    echo "设置方法:"
    echo "  aws configure set region us-east-1"
else
    echo "✓ 默认区域: $REGION"
fi

echo ""

# 检查密钥对
echo "=== 检查密钥对 ==="
KEY_NAME="fpga-f2-key"
if aws ec2 describe-key-pairs --key-names $KEY_NAME --region us-east-1 &> /dev/null; then
    echo "✓ 密钥对存在: $KEY_NAME"
    
    # 检查本地密钥文件
    if [ -f ~/.ssh/${KEY_NAME}.pem ]; then
        echo "✓ 本地密钥文件存在: ~/.ssh/${KEY_NAME}.pem"
        
        # 检查权限
        PERMS=$(stat -c %a ~/.ssh/${KEY_NAME}.pem 2>/dev/null || stat -f %A ~/.ssh/${KEY_NAME}.pem 2>/dev/null)
        if [ "$PERMS" == "400" ] || [ "$PERMS" == "600" ]; then
            echo "✓ 密钥文件权限正确: $PERMS"
        else
            echo "⚠️  密钥文件权限不正确: $PERMS"
            echo "修复命令:"
            echo "  chmod 400 ~/.ssh/${KEY_NAME}.pem"
        fi
    else
        echo "⚠️  本地密钥文件不存在: ~/.ssh/${KEY_NAME}.pem"
        echo ""
        echo "如果密钥文件在其他位置，请移动到 ~/.ssh/"
        echo "如果密钥丢失，需要创建新密钥对"
    fi
else
    echo "❌ 密钥对不存在: $KEY_NAME"
    echo ""
    echo "创建密钥对:"
    echo "  aws ec2 create-key-pair --key-name $KEY_NAME --region us-east-1 --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem"
    echo "  chmod 400 ~/.ssh/${KEY_NAME}.pem"
fi

echo ""

# 检查安全组
echo "=== 检查安全组 ==="
SECURITY_GROUP="sg-03d27449f82b54360"
if aws ec2 describe-security-groups --group-ids $SECURITY_GROUP --region us-east-1 &> /dev/null; then
    echo "✓ 安全组存在: $SECURITY_GROUP"
    
    # 显示安全组规则
    echo ""
    echo "入站规则:"
    aws ec2 describe-security-groups \
        --group-ids $SECURITY_GROUP \
        --region us-east-1 \
        --query 'SecurityGroups[0].IpPermissions[*].[IpProtocol,FromPort,ToPort,IpRanges[0].CidrIp]' \
        --output table
else
    echo "❌ 安全组不存在: $SECURITY_GROUP"
    echo ""
    echo "请创建安全组或使用其他安全组"
fi

echo ""

# 检查 F2 配额
echo "=== 检查 F2 实例配额 ==="
echo "注意: AWS F1 实例已于 2024 年退役，现在使用 F2 实例"
echo ""
if command -v jq &> /dev/null; then
    QUOTA=$(aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code L-85EED4F7 \
        --region us-east-1 2>/dev/null | jq -r '.Quota.Value')
    
    if [ -n "$QUOTA" ] && [ "$QUOTA" != "null" ]; then
        if [ "$QUOTA" == "0" ] || [ "$QUOTA" == "0.0" ]; then
            echo "❌ F2 实例配额为 0"
            echo ""
            echo "需要申请 F2 配额:"
            echo "  https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-85EED4F7"
        else
            echo "✓ F2 实例配额: $QUOTA"
        fi
    else
        echo "⚠️  无法查询 F2 配额"
    fi
else
    echo "⚠️  jq 未安装，跳过配额检查"
    echo "安装 jq: sudo apt install jq"
fi

echo ""

# 测试 EC2 API 调用
echo "=== 测试 EC2 API 调用 ==="
echo "查询可用区..."
if aws ec2 describe-availability-zones --region us-east-1 --query 'AvailabilityZones[*].[ZoneName,State]' --output table &> /dev/null; then
    echo "✓ EC2 API 调用成功"
    echo ""
    echo "可用区列表:"
    aws ec2 describe-availability-zones --region us-east-1 --query 'AvailabilityZones[*].[ZoneName,State]' --output table
else
    echo "❌ EC2 API 调用失败"
    echo ""
    echo "可能的原因:"
    echo "  1. 网络连接问题"
    echo "  2. AWS 凭证权限不足"
    echo "  3. 区域配置错误"
fi

echo ""

# 检查 F2 实例类型可用性
echo "=== 检查 F2 实例类型可用性 ==="
echo "查询 f2.6xlarge 在各可用区的可用性..."
F2_ZONES=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters Name=instance-type,Values=f2.6xlarge \
    --region us-east-1 \
    --query 'InstanceTypeOfferings[*].Location' \
    --output text 2>/dev/null)

if [ -n "$F2_ZONES" ]; then
    echo "✓ F2 实例在以下可用区可用:"
    for zone in $F2_ZONES; do
        echo "  - $zone"
    done
else
    echo "⚠️  无法查询 F2 实例可用性"
fi

echo ""
echo "=== 测试完成 ==="
echo ""
echo "如果所有检查都通过，可以尝试启动 F2 实例:"
echo "  ./launch_f2_vivado.sh"
