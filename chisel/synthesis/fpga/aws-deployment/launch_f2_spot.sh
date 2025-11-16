#!/bin/bash
# 启动 AWS F2 Spot 实例脚本

set -e

echo "=== AWS F2 Spot 实例启动脚本 ==="
echo ""

# 配置参数
INSTANCE_TYPE="f2.6xlarge"
KEY_NAME="fpga-dev-key"
REGION="us-east-1"
AMI_ID="ami-0cb1b6ae2ff99f8bf"  # FPGA Developer AMI 1.18.0
SPOT_PRICE="1.00"  # 最高出价 $1.00/小时（当前 Spot 价格 $0.50-$0.72）
SECURITY_GROUP="sg-03d27449f82b54360"
AVAILABILITY_ZONE="us-east-1b"  # Spot 价格最低的可用区

echo "配置信息:"
echo "  实例类型: $INSTANCE_TYPE"
echo "  AMI: $AMI_ID"
echo "  最高出价: $SPOT_PRICE/小时"
echo "  可用区: $AVAILABILITY_ZONE"
echo "  当前 Spot 价格: ~\$0.50-\$0.72/小时"
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    exit 1
fi
echo "✓ AWS CLI 已安装"

# 检查 AWS 凭证
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS 凭证未配置"
    exit 1
fi
echo "✓ AWS 凭证已配置"
echo ""

# 创建 Spot 实例请求
echo "请求 Spot 实例..."

SPOT_REQUEST=$(aws ec2 request-spot-instances \
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
        },
        \"TagSpecifications\": [{
            \"ResourceType\": \"instance\",
            \"Tags\": [
                {\"Key\": \"Name\", \"Value\": \"FPGA-Dev-F2\"},
                {\"Key\": \"Project\", \"Value\": \"RISC-V-AI-Accelerator\"},
                {\"Key\": \"Type\", \"Value\": \"Spot\"}
            ]
        }]
    }" \
    --region $REGION \
    --output json 2>&1)

if [ $? -ne 0 ]; then
    echo "❌ Spot 实例请求失败"
    echo "$SPOT_REQUEST"
    exit 1
fi

SPOT_REQUEST_ID=$(echo "$SPOT_REQUEST" | python3 -c "import json, sys; print(json.load(sys.stdin)['SpotInstanceRequests'][0]['SpotInstanceRequestId'])" 2>/dev/null)

if [ -z "$SPOT_REQUEST_ID" ]; then
    echo "❌ 无法获取 Spot 请求 ID"
    echo "$SPOT_REQUEST" | head -20
    exit 1
fi

echo "✓ Spot 请求已提交: $SPOT_REQUEST_ID"
echo ""

# 等待 Spot 请求完成
echo "等待 Spot 请求完成..."
MAX_WAIT=300  # 最多等待 5 分钟
WAIT_TIME=0

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    STATUS=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids $SPOT_REQUEST_ID \
        --region $REGION \
        --query 'SpotInstanceRequests[0].State' \
        --output text 2>/dev/null)
    
    if [ "$STATUS" == "active" ]; then
        echo "✓ Spot 请求已激活"
        break
    elif [ "$STATUS" == "failed" ] || [ "$STATUS" == "cancelled" ]; then
        echo "❌ Spot 请求失败: $STATUS"
        aws ec2 describe-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --region $REGION \
            --query 'SpotInstanceRequests[0].Status' \
            --output json
        exit 1
    fi
    
    echo "  状态: $STATUS (等待 ${WAIT_TIME}s)"
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "❌ 等待超时"
    exit 1
fi

# 获取实例 ID
INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids $SPOT_REQUEST_ID \
    --region $REGION \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
    echo "❌ 无法获取实例 ID"
    exit 1
fi

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

# 获取实际 Spot 价格
ACTUAL_PRICE=$(aws ec2 describe-spot-instance-requests \
    --spot-instance-request-ids $SPOT_REQUEST_ID \
    --region $REGION \
    --query 'SpotInstanceRequests[0].ActualBlockHourlyPrice' \
    --output text 2>/dev/null || echo "N/A")

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              F2 Spot 实例启动成功！                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  Spot 请求 ID: $SPOT_REQUEST_ID"
echo "  公网 IP: $PUBLIC_IP"
echo "  区域: $REGION"
echo "  可用区: $AVAILABILITY_ZONE"
echo "  实例类型: $INSTANCE_TYPE"
echo "  实际价格: \$$ACTUAL_PRICE/小时"
echo ""
echo "FPGA 信息:"
echo "  FPGA 型号: Xilinx Virtex UltraScale+ VU47P"
echo "  FPGA 数量: 1"
echo "  FPGA 内存: 80 GB"
echo "  逻辑单元: ~2.8M"
echo ""
echo "连接命令:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem rocky@${PUBLIC_IP}"
echo ""
echo "保存实例信息到文件..."
cat > ../build/f2_instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Spot Request ID: $SPOT_REQUEST_ID
Public IP: $PUBLIC_IP
Region: $REGION
Availability Zone: $AVAILABILITY_ZONE
Instance Type: $INSTANCE_TYPE
Actual Price: \$$ACTUAL_PRICE/hour
FPGA: Xilinx VU47P
Launch Time: $(date)
EOF

echo "✓ 实例信息已保存到: ../build/f2_instance_info.txt"
echo ""
echo "下一步:"
echo "  1. 等待 2-3 分钟让实例完全启动"
echo "  2. 连接到实例: ssh -i ~/.ssh/${KEY_NAME}.pem rocky@${PUBLIC_IP}"
echo "  3. 运行环境配置: ./setup_aws.sh"
echo ""
echo "成本估算:"
echo "  当前 Spot 价格: ~\$0.50-\$0.72/小时"
echo "  完整验证（3-5小时）: ~\$1.50-\$3.60"
echo "  比 F1 节省: ~60-70%"
echo ""
echo "停止实例命令:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "取消 Spot 请求:"
echo "  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID --region $REGION"
