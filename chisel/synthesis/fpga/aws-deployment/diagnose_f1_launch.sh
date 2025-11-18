#!/bin/bash
# 诊断 F1 实例启动问题

set -e

echo "=== F1 实例启动诊断工具 ==="
echo ""

# 配置参数（与 launch_f1_ondemand.sh 相同）
INSTANCE_TYPE="f1.2xlarge"
KEY_NAME="fpga-f2-key"
REGION="us-east-1"
AMI_ID="ami-092fc5deb8f3c0f7d"
SECURITY_GROUP="sg-03d27449f82b54360"
TEST_AZ="us-east-1a"

echo "测试配置:"
echo "  实例类型: $INSTANCE_TYPE"
echo "  AMI ID: $AMI_ID"
echo "  区域: $REGION"
echo "  测试可用区: $TEST_AZ"
echo "  密钥: $KEY_NAME"
echo "  安全组: $SECURITY_GROUP"
echo ""

# 1. 检查 AWS CLI
echo "=== 1. 检查 AWS CLI ==="
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    exit 1
fi

AWS_VERSION=$(aws --version 2>&1)
echo "✓ AWS CLI 已安装: $AWS_VERSION"
echo ""

# 2. 检查 AWS 凭证
echo "=== 2. 检查 AWS 凭证 ==="
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    echo "✓ AWS 凭证有效"
    echo "  账户 ID: $ACCOUNT_ID"
    echo "  用户 ARN: $USER_ARN"
else
    echo "❌ AWS 凭证无效或未配置"
    echo ""
    echo "配置 AWS 凭证:"
    echo "  aws configure"
    exit 1
fi
echo ""

# 3. 检查密钥对
echo "=== 3. 检查密钥对 ==="
if aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    echo "✓ 密钥对存在: $KEY_NAME"
    
    # 检查本地密钥文件
    if [ -f ~/.ssh/${KEY_NAME}.pem ]; then
        KEY_PERMS=$(stat -c %a ~/.ssh/${KEY_NAME}.pem 2>/dev/null || stat -f %A ~/.ssh/${KEY_NAME}.pem 2>/dev/null)
        echo "✓ 本地密钥文件存在: ~/.ssh/${KEY_NAME}.pem (权限: $KEY_PERMS)"
    else
        echo "⚠️  本地密钥文件不存在: ~/.ssh/${KEY_NAME}.pem"
    fi
else
    echo "❌ 密钥对不存在: $KEY_NAME"
    echo ""
    echo "创建密钥对:"
    echo "  aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem"
    echo "  chmod 400 ~/.ssh/${KEY_NAME}.pem"
    exit 1
fi
echo ""

# 4. 检查安全组
echo "=== 4. 检查安全组 ==="
if aws ec2 describe-security-groups --group-ids $SECURITY_GROUP --region $REGION &> /dev/null; then
    SG_NAME=$(aws ec2 describe-security-groups --group-ids $SECURITY_GROUP --region $REGION --query 'SecurityGroups[0].GroupName' --output text)
    VPC_ID=$(aws ec2 describe-security-groups --group-ids $SECURITY_GROUP --region $REGION --query 'SecurityGroups[0].VpcId' --output text)
    echo "✓ 安全组存在: $SECURITY_GROUP"
    echo "  名称: $SG_NAME"
    echo "  VPC: $VPC_ID"
else
    echo "❌ 安全组不存在: $SECURITY_GROUP"
    exit 1
fi
echo ""

# 5. 检查 AMI
echo "=== 5. 检查 AMI ==="
if aws ec2 describe-images --image-ids $AMI_ID --region $REGION &> /dev/null; then
    AMI_NAME=$(aws ec2 describe-images --image-ids $AMI_ID --region $REGION --query 'Images[0].Name' --output text)
    AMI_STATE=$(aws ec2 describe-images --image-ids $AMI_ID --region $REGION --query 'Images[0].State' --output text)
    echo "✓ AMI 存在: $AMI_ID"
    echo "  名称: $AMI_NAME"
    echo "  状态: $AMI_STATE"
else
    echo "❌ AMI 不存在或无权访问: $AMI_ID"
    exit 1
fi
echo ""

# 6. 检查 F1 配额
echo "=== 6. 检查 F1 实例配额 ==="
F1_QUOTA=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-85EED4F7 \
    --region $REGION \
    --query 'Quota.Value' \
    --output text 2>/dev/null || echo "0")

if [ "$F1_QUOTA" == "0" ] || [ -z "$F1_QUOTA" ]; then
    echo "❌ F1 实例配额为 0 或无法查询"
    echo ""
    echo "F1 实例需要特殊配额。请求配额:"
    echo "  1. 访问: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
    echo "  2. 搜索: Running On-Demand F instances"
    echo "  3. 请求增加配额到至少 1"
    echo ""
    echo "或使用 AWS CLI:"
    echo "  aws service-quotas request-service-quota-increase \\"
    echo "    --service-code ec2 \\"
    echo "    --quota-code L-85EED4F7 \\"
    echo "    --desired-value 1 \\"
    echo "    --region $REGION"
    exit 1
else
    echo "✓ F1 实例配额: $F1_QUOTA"
fi
echo ""

# 7. 检查可用区中的 F1 可用性
echo "=== 7. 检查 F1 实例在可用区的可用性 ==="
F1_AZS=$(aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters Name=instance-type,Values=$INSTANCE_TYPE \
    --region $REGION \
    --query 'InstanceTypeOfferings[*].Location' \
    --output text)

if [ -z "$F1_AZS" ]; then
    echo "❌ 该区域没有可用的 F1 实例"
    echo ""
    echo "尝试其他区域:"
    echo "  us-west-2"
    echo "  us-east-1"
    echo "  eu-west-1"
    exit 1
else
    echo "✓ F1 实例可用的可用区:"
    for az in $F1_AZS; do
        echo "  - $az"
    done
fi
echo ""

# 8. 测试实例启动（干运行）
echo "=== 8. 测试实例启动（干运行）==="
echo "执行 dry-run 测试..."

DRY_RUN_OUTPUT=$(aws ec2 run-instances \
    --dry-run \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --placement "AvailabilityZone=$TEST_AZ" \
    --region $REGION \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=F1-Test}]" 2>&1)

DRY_RUN_EXIT=$?

if [[ "$DRY_RUN_OUTPUT" == *"DryRunOperation"* ]]; then
    echo "✓ Dry-run 成功 - 所有权限和配置正确"
    echo ""
    echo "这意味着实际启动应该可以成功！"
elif [[ "$DRY_RUN_OUTPUT" == *"InsufficientInstanceCapacity"* ]]; then
    echo "⚠️  Dry-run 显示容量不足"
    echo ""
    echo "该可用区当前没有 F1 容量，但配置正确。"
    echo "建议尝试其他可用区或稍后重试。"
elif [[ "$DRY_RUN_OUTPUT" == *"Unsupported"* ]]; then
    echo "❌ 该可用区不支持 F1 实例"
    echo ""
    echo "请尝试其他可用区: $F1_AZS"
else
    echo "❌ Dry-run 失败"
    echo ""
    echo "错误信息:"
    echo "$DRY_RUN_OUTPUT"
    echo ""
    exit 1
fi
echo ""

# 9. 检查当前运行的实例
echo "=== 9. 检查当前运行的 F1 实例 ==="
RUNNING_F1=$(aws ec2 describe-instances \
    --filters "Name=instance-type,Values=$INSTANCE_TYPE" "Name=instance-state-name,Values=running,pending" \
    --region $REGION \
    --query 'Reservations[*].Instances[*].[InstanceId,State.Name,Placement.AvailabilityZone]' \
    --output text)

if [ -z "$RUNNING_F1" ]; then
    echo "✓ 当前没有运行的 F1 实例"
else
    echo "⚠️  发现运行中的 F1 实例:"
    echo "$RUNNING_F1"
    echo ""
    echo "如果达到配额限制，需要先停止现有实例"
fi
echo ""

# 10. 总结
echo "=== 诊断总结 ==="
echo ""
echo "✓ AWS CLI 配置正确"
echo "✓ 凭证有效"
echo "✓ 密钥对存在"
echo "✓ 安全组存在"
echo "✓ AMI 可用"

if [ "$F1_QUOTA" != "0" ] && [ -n "$F1_QUOTA" ]; then
    echo "✓ F1 配额充足"
else
    echo "❌ F1 配额不足 - 这是主要问题！"
fi

if [ -n "$F1_AZS" ]; then
    echo "✓ F1 实例在某些可用区可用"
else
    echo "❌ F1 实例在该区域不可用"
fi

echo ""
echo "=== 建议操作 ==="

if [ "$F1_QUOTA" == "0" ] || [ -z "$F1_QUOTA" ]; then
    echo ""
    echo "🔴 主要问题: F1 配额为 0"
    echo ""
    echo "解决方案:"
    echo "  1. 请求 F1 配额（需要 1-2 个工作日）:"
    echo "     https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
    echo ""
    echo "  2. 或者使用其他实例类型进行测试"
    echo ""
else
    echo ""
    echo "🟢 配置看起来正常"
    echo ""
    echo "如果启动仍然失败，可能是:"
    echo "  1. 临时容量不足 - 尝试其他可用区"
    echo "  2. 使用 Spot 实例而非按需实例"
    echo "  3. 稍后重试"
    echo ""
    echo "尝试启动实例:"
    echo "  ./launch_f1_ondemand.sh"
    echo ""
fi
