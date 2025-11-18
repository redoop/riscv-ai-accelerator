#!/bin/bash
# 检查 F1 实例可用性和配额

set -e

REGION="us-east-1"
INSTANCE_TYPE="f1.2xlarge"

echo "=== F1 实例可用性检查 ==="
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    exit 1
fi

echo "✓ AWS CLI 已安装"
echo ""

# 检查 AWS 凭证
echo "检查 AWS 凭证..."
if ! aws sts get-caller-identity &>/dev/null; then
    echo "❌ AWS 凭证未配置或无效"
    echo ""
    echo "配置方法:"
    echo "  aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
echo "✓ AWS 账户: $ACCOUNT_ID"
echo ""

# 检查区域中的可用区
echo "检查 $REGION 中的可用区..."
ZONES=$(aws ec2 describe-availability-zones \
    --region $REGION \
    --query 'AvailabilityZones[?State==`available`].ZoneName' \
    --output text)

echo "可用区: $ZONES"
echo ""

# 检查 F1 实例类型可用性
echo "检查 F1 实例类型可用性..."
for AZ in $ZONES; do
    AVAILABLE=$(aws ec2 describe-instance-type-offerings \
        --location-type availability-zone \
        --filters "Name=location,Values=$AZ" "Name=instance-type,Values=$INSTANCE_TYPE" \
        --region $REGION \
        --query 'InstanceTypeOfferings[0].InstanceType' \
        --output text 2>/dev/null)
    
    if [ "$AVAILABLE" == "$INSTANCE_TYPE" ]; then
        echo "  ✓ $AZ: 支持 $INSTANCE_TYPE"
    else
        echo "  ✗ $AZ: 不支持 $INSTANCE_TYPE"
    fi
done
echo ""

# 检查 vCPU 配额
echo "检查 vCPU 配额..."
F1_VCPU_QUOTA=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-74FC7D96 \
    --region $REGION \
    --query 'Quota.Value' \
    --output text 2>/dev/null || echo "0")

if [ "$F1_VCPU_QUOTA" == "0" ]; then
    echo "⚠️  无法查询 F1 vCPU 配额（可能需要权限）"
else
    echo "F1 vCPU 配额: $F1_VCPU_QUOTA"
    
    # F1.2xlarge 需要 8 vCPUs
    if (( $(echo "$F1_VCPU_QUOTA >= 8" | bc -l) )); then
        echo "✓ 配额足够（需要 8 vCPUs）"
    else
        echo "❌ 配额不足（需要 8 vCPUs，当前: $F1_VCPU_QUOTA）"
        echo ""
        echo "请求增加配额:"
        echo "  https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-74FC7D96"
    fi
fi
echo ""

# 检查 Spot 实例配额
echo "检查 Spot 实例配额..."
SPOT_QUOTA=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-34B43A08 \
    --region $REGION \
    --query 'Quota.Value' \
    --output text 2>/dev/null || echo "0")

if [ "$SPOT_QUOTA" == "0" ]; then
    echo "⚠️  无法查询 Spot 配额"
else
    echo "Spot vCPU 配额: $SPOT_QUOTA"
fi
echo ""

# 检查当前运行的实例
echo "检查当前运行的 F1 实例..."
RUNNING=$(aws ec2 describe-instances \
    --filters "Name=instance-type,Values=f1.*" "Name=instance-state-name,Values=running,pending" \
    --region $REGION \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone]' \
    --output text)

if [ -z "$RUNNING" ]; then
    echo "✓ 当前没有运行的 F1 实例"
else
    echo "当前运行的 F1 实例:"
    echo "$RUNNING"
fi
echo ""

# 测试 Spot 价格历史
echo "检查 Spot 价格历史..."
SPOT_PRICES=$(aws ec2 describe-spot-price-history \
    --instance-types $INSTANCE_TYPE \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
    --product-descriptions "Linux/UNIX" \
    --region $REGION \
    --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice,Timestamp]' \
    --output text | sort -k1,1 -k3,3r | awk '!seen[$1]++')

if [ -z "$SPOT_PRICES" ]; then
    echo "⚠️  无 Spot 价格数据（可能表示容量不足）"
else
    echo "最新 Spot 价格:"
    echo "$SPOT_PRICES" | while read az price time; do
        echo "  $az: \$$price/小时 (更新: $time)"
    done
fi
echo ""

# 建议
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "建议"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$F1_VCPU_QUOTA" == "0" ] || (( $(echo "$F1_VCPU_QUOTA < 8" | bc -l 2>/dev/null || echo "0") )); then
    echo "1. 请求增加 F1 vCPU 配额"
    echo "   https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-74FC7D96"
    echo ""
fi

if [ -z "$SPOT_PRICES" ]; then
    echo "2. Spot 容量不足，建议:"
    echo "   - 使用按需实例"
    echo "   - 尝试其他区域（us-west-2）"
    echo "   - 稍后重试"
    echo ""
fi

echo "3. 如果问题持续，联系 AWS 支持"
echo "   https://console.aws.amazon.com/support/home"
echo ""

echo "4. 考虑使用其他区域:"
echo "   - us-west-2 (Oregon)"
echo "   - eu-west-1 (Ireland)"
echo ""
