#!/bin/bash
# 使用预装 Vivado 的 AMI 启动 F2 Spot 实例

set -e

echo "=== 启动 F2 实例（Vivado 2025.1 预装）==="
echo ""

# 配置参数
INSTANCE_TYPE="f2.6xlarge"
KEY_NAME="fpga-f2-key"
REGION="us-east-1"
AMI_ID="ami-0b359c50bdba2aac0"  # Vivado 2025.1 预装
SPOT_PRICE="1.00"
SECURITY_GROUP="sg-03d27449f82b54360"
AVAILABILITY_ZONE="us-east-1b"

echo "配置信息:"
echo "  AMI: $AMI_ID (Vivado 2025.1 预装)"
echo "  实例类型: $INSTANCE_TYPE"
echo "  Spot 价格: $SPOT_PRICE/小时"
echo "  可用区: $AVAILABILITY_ZONE"
echo ""

# 请求 Spot 实例
echo "请求 Spot 实例..."
SPOT_REQUEST_ID=$(aws ec2 request-spot-instances \
    --spot-price "$SPOT_PRICE" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
        \"Placement\": {
            \"AvailabilityZone\": \"$AVAILABILITY_ZONE\"
        }
    }" \
    --region $REGION \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
    --output text)

echo "✓ Spot 请求 ID: $SPOT_REQUEST_ID"
echo ""

# 等待 Spot 请求完成
echo "等待 Spot 请求完成..."
for i in {1..30}; do
    STATUS=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids $SPOT_REQUEST_ID \
        --region $REGION \
        --query 'SpotInstanceRequests[0].State' \
        --output text)
    
    echo "  [$i/30] 状态: $STATUS"
    
    if [ "$STATUS" == "active" ]; then
        echo "✓ Spot 请求已激活"
        break
    elif [ "$STATUS" == "failed" ] || [ "$STATUS" == "cancelled" ]; then
        echo "❌ Spot 请求失败"
        exit 1
    fi
    
    sleep 5
done

# 获取实例 ID
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids $SPOT_REQUEST_ID \
    --region $REGION \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text)

echo "✓ 实例 ID: $INSTANCE_ID"
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

# 保存实例信息到文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f2_instance_info"

cat > "$INFO_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
SPOT_REQUEST_ID=$SPOT_REQUEST_ID
PUBLIC_IP=$PUBLIC_IP
KEY_NAME=$KEY_NAME
REGION=$REGION
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
EOF

echo "✓ 实例信息已保存到: $INFO_FILE"
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         F2 实例启动成功（Vivado 预装）！                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  Spot 请求 ID: $SPOT_REQUEST_ID"
echo "  公网 IP: $PUBLIC_IP"
echo "  AMI: Vivado 2025.1 (预装)"
echo ""
echo "连接命令:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "验证 Vivado:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'vivado -version'"
echo ""
echo "上传项目:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem fpga-project.tar.gz ubuntu@${PUBLIC_IP}:~/"
echo ""
echo "停止实例:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
