#!/bin/bash
# 使用预装 Vivado 的 AMI 启动 F1 Spot 实例
# F1 实例使用 xcvu9p 设备，与 AWS AFI 服务兼容

set -e

echo "=== 启动 F1 实例（Vivado 2024.1 预装）==="
echo ""

# 配置参数
INSTANCE_TYPE="f1.2xlarge"
KEY_NAME="fpga-f2-key"
REGION="us-east-1"
AMI_ID="ami-092fc5deb8f3c0f7d"  # FPGA Developer AMI (Ubuntu) 1.16.1 - Vivado 2024.1
SPOT_PRICE="0.60"  # F1.2xlarge Spot 价格约 $0.50-0.60/小时
SECURITY_GROUP="sg-03d27449f82b54360"

# 可用区列表（按推荐顺序 - us-east-1a 不支持 F1）
AVAILABILITY_ZONES=("us-east-1b" "us-east-1c" "us-east-1d" "us-east-1e")

echo "配置信息:"
echo "  AMI: $AMI_ID"
echo "  AMI 名称: FPGA Developer AMI (Ubuntu) 1.16.1"
echo "  Vivado 版本: 2024.1"
echo "  实例类型: $INSTANCE_TYPE (xcvu9p - F1 兼容)"
echo "  Spot 价格: $SPOT_PRICE/小时"
echo "  可用区: 自动选择（${AVAILABILITY_ZONES[*]}）"
echo ""
echo "⚠️  重要: F1 实例使用 xcvu9p 设备，与 AWS AFI 服务兼容"
echo ""

# 尝试多个可用区
SPOT_REQUEST_ID=""
SELECTED_AZ=""

for AZ in "${AVAILABILITY_ZONES[@]}"; do
    echo "尝试可用区: $AZ"
    
    # 尝试请求 Spot 实例
    SPOT_OUTPUT=$(aws ec2 request-spot-instances \
        --spot-price "$SPOT_PRICE" \
        --instance-count 1 \
        --type "one-time" \
        --launch-specification "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"KeyName\": \"$KEY_NAME\",
            \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
            \"Placement\": {
                \"AvailabilityZone\": \"$AZ\"
            }
        }" \
        --region $REGION \
        --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
        --output text 2>&1)
    
    EXIT_CODE=$?
    
    # 检查是否成功创建请求
    if [ $EXIT_CODE -ne 0 ] || [ -z "$SPOT_OUTPUT" ] || [ "$SPOT_OUTPUT" == "None" ] || [[ "$SPOT_OUTPUT" == *"error"* ]] || [[ "$SPOT_OUTPUT" == *"Error"* ]] || [[ ! "$SPOT_OUTPUT" =~ ^sir- ]]; then
        ERROR_MSG=$(echo "$SPOT_OUTPUT" | grep -oP '(?<=\().*?(?=\))' | head -1)
        if [ -z "$ERROR_MSG" ]; then
            ERROR_MSG="$SPOT_OUTPUT"
        fi
        echo "  ✗ $AZ 不可用"
        echo "     原因: $ERROR_MSG"
        sleep 2  # 避免 API 限流
        continue
    fi
    
    SPOT_REQUEST_ID="$SPOT_OUTPUT"
    echo "  ✓ Spot 请求 ID: $SPOT_REQUEST_ID"
    
    # 等待请求状态
    sleep 3
    STATUS=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids $SPOT_REQUEST_ID \
        --region $REGION \
        --query 'SpotInstanceRequests[0].State' \
        --output text 2>/dev/null)
    
    if [ "$STATUS" == "failed" ] || [ "$STATUS" == "cancelled" ]; then
        # 获取失败原因
        FAULT=$(aws ec2 describe-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --region $REGION \
            --query 'SpotInstanceRequests[0].Fault.Message' \
            --output text 2>/dev/null)
        
        echo "  ✗ $AZ 请求失败"
        if [ -n "$FAULT" ] && [ "$FAULT" != "None" ]; then
            echo "     原因: $FAULT"
        fi
        
        # 取消失败的请求
        aws ec2 cancel-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --region $REGION &>/dev/null
        
        SPOT_REQUEST_ID=""
        sleep 2
        continue
    fi
    
    # 成功
    SELECTED_AZ="$AZ"
    echo "  ✓ 成功选择可用区: $AZ"
    break
done

if [ -z "$SPOT_REQUEST_ID" ]; then
    echo ""
    echo "❌ 所有可用区都无法启动 Spot 实例"
    echo ""
    echo "可能的原因:"
    echo "  1. Spot 容量不足"
    echo "  2. Spot 价格太低"
    echo "  3. 配额限制"
    echo ""
    echo "建议:"
    echo "  1. 提高 Spot 价格（当前: $SPOT_PRICE）"
    echo "  2. 使用按需实例（价格: \$1.65/小时）"
    echo "  3. 稍后重试"
    echo ""
    echo "按需实例命令:"
    echo "  aws ec2 run-instances \\"
    echo "    --image-id $AMI_ID \\"
    echo "    --instance-type $INSTANCE_TYPE \\"
    echo "    --key-name $KEY_NAME \\"
    echo "    --security-group-ids $SECURITY_GROUP \\"
    echo "    --region $REGION"
    exit 1
fi

echo ""
echo "✓ Spot 请求 ID: $SPOT_REQUEST_ID"
echo "✓ 可用区: $SELECTED_AZ"
echo ""

# 等待 Spot 请求完成
echo "等待 Spot 请求激活..."
ACTIVATED=false
for i in {1..30}; do
    STATUS=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids $SPOT_REQUEST_ID \
        --region $REGION \
        --query 'SpotInstanceRequests[0].State' \
        --output text)
    
    echo "  [$i/30] 状态: $STATUS"
    
    if [ "$STATUS" == "active" ]; then
        echo "✓ Spot 请求已激活"
        ACTIVATED=true
        break
    elif [ "$STATUS" == "failed" ] || [ "$STATUS" == "cancelled" ]; then
        echo "❌ Spot 请求失败"
        
        # 获取失败原因
        FAULT=$(aws ec2 describe-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --region $REGION \
            --query 'SpotInstanceRequests[0].Fault.Message' \
            --output text 2>/dev/null)
        
        if [ -n "$FAULT" ] && [ "$FAULT" != "None" ]; then
            echo "失败原因: $FAULT"
        fi
        
        # 取消请求
        aws ec2 cancel-spot-instance-requests \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --region $REGION &>/dev/null
        
        exit 1
    fi
    
    sleep 5
done

if [ "$ACTIVATED" = false ]; then
    echo ""
    echo "❌ Spot 请求超时（一直处于 open 状态）"
    echo ""
    echo "这通常意味着当前没有可用的 Spot 容量。"
    echo ""
    echo "建议："
    echo "  1. 使用按需实例（更可靠，但更贵）"
    echo "  2. 稍后重试 Spot 实例"
    echo "  3. 提高 Spot 出价"
    echo ""
    
    # 取消 Spot 请求
    echo "取消 Spot 请求..."
    aws ec2 cancel-spot-instance-requests \
        --spot-instance-request-ids $SPOT_REQUEST_ID \
        --region $REGION
    
    echo ""
    read -p "是否改用按需实例？(y/N): " use_ondemand
    
    if [[ $use_ondemand =~ ^[Yy]$ ]]; then
        echo ""
        echo "启动按需实例..."
        
        # 按需实例推荐的可用区（根据错误提示）
        ONDEMAND_AZS=("us-east-1a" "us-east-1b" "us-east-1d" "us-east-1e")
        
        INSTANCE_ID=""
        for AZ in "${ONDEMAND_AZS[@]}"; do
            echo "  尝试可用区: $AZ"
            
            INSTANCE_ID=$(aws ec2 run-instances \
                --image-id $AMI_ID \
                --instance-type $INSTANCE_TYPE \
                --key-name $KEY_NAME \
                --security-group-ids $SECURITY_GROUP \
                --placement "AvailabilityZone=$AZ" \
                --region $REGION \
                --query 'Instances[0].InstanceId' \
                --output text 2>&1)
            
            if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ] && [[ "$INSTANCE_ID" != *"error"* ]]; then
                SELECTED_AZ="$AZ"
                echo "  ✓ 成功在 $AZ 启动实例"
                break
            else
                echo "  ✗ $AZ 不可用"
                INSTANCE_ID=""
            fi
        done
        
        if [ -z "$INSTANCE_ID" ]; then
            echo ""
            echo "❌ 所有可用区都无法启动按需实例"
            echo ""
            echo "可能的原因:"
            echo "  1. 配额限制"
            echo "  2. 账户权限问题"
            echo "  3. F1 实例在该区域不可用"
            echo ""
            echo "建议:"
            echo "  1. 检查 AWS 配额限制"
            echo "  2. 尝试其他区域（如 us-west-2）"
            echo "  3. 联系 AWS 支持"
            exit 1
        fi
        
        echo "✓ 实例 ID: $INSTANCE_ID"
        echo "✓ 可用区: $SELECTED_AZ"
        SPOT_REQUEST_ID="on-demand"
    else
        echo "已取消"
        exit 1
    fi
else
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
fi

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
INFO_FILE="$SCRIPT_DIR/.f1_instance_info"

cat > "$INFO_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
SPOT_REQUEST_ID=$SPOT_REQUEST_ID
PUBLIC_IP=$PUBLIC_IP
KEY_NAME=$KEY_NAME
REGION=$REGION
INSTANCE_TYPE=$INSTANCE_TYPE
AVAILABILITY_ZONE=$SELECTED_AZ
DEVICE=xcvu9p
BILLING_TYPE=spot
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
EOF

# 验证文件是否创建成功
if [ ! -f "$INFO_FILE" ]; then
    echo ""
    echo "❌ 实例信息文件创建失败"
    echo "文件路径: $INFO_FILE"
    echo ""
    echo "可能的原因:"
    echo "  1. 目录权限不足"
    echo "  2. 磁盘空间不足"
    echo "  3. 文件系统错误"
    echo ""
    echo "请检查目录权限:"
    echo "  ls -la $SCRIPT_DIR"
    echo ""
    echo "实例已创建但信息未保存，请手动记录:"
    echo "  实例 ID: $INSTANCE_ID"
    echo "  Spot 请求 ID: $SPOT_REQUEST_ID"
    echo "  公网 IP: $PUBLIC_IP"
    echo "  可用区: $SELECTED_AZ"
    echo ""
    exit 1
fi

# 验证文件内容
if ! grep -q "INSTANCE_ID=$INSTANCE_ID" "$INFO_FILE"; then
    echo ""
    echo "❌ 实例信息文件内容不完整"
    echo ""
    echo "文件已创建但内容可能损坏，请手动记录:"
    echo "  实例 ID: $INSTANCE_ID"
    echo "  Spot 请求 ID: $SPOT_REQUEST_ID"
    echo "  公网 IP: $PUBLIC_IP"
    echo "  可用区: $SELECTED_AZ"
    echo ""
    exit 1
fi

echo "✓ 实例信息已保存到: $INFO_FILE"
echo ""

# 等待 SSH 可用
echo "等待 SSH 服务启动..."
for i in {1..30}; do
    if ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        ec2-user@${PUBLIC_IP} "echo 'SSH ready'" 2>/dev/null; then
        echo "✓ SSH 服务已就绪"
        break
    fi
    echo "  [$i/30] 等待 SSH..."
    sleep 10
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         F1 实例启动成功（Vivado 2024.1）！                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  实例类型: $INSTANCE_TYPE"
echo "  可用区: $SELECTED_AZ"
echo "  FPGA 设备: xcvu9p (F1 兼容)"
echo "  Spot 请求 ID: $SPOT_REQUEST_ID"
echo "  公网 IP: $PUBLIC_IP"
echo "  AMI: FPGA Developer AMI (Ubuntu) 1.16.1 - Vivado 2024.1"
echo ""
echo "连接命令:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "验证 Vivado:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'vivado -version'"
echo ""
echo "验证 AWS FPGA 环境:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} 'ls -la ~/aws-fpga'"
echo ""
echo "上传项目:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem fpga-project.tar.gz ec2-user@${PUBLIC_IP}:~/"
echo ""
echo "停止实例:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "成本估算:"
echo "  Spot 价格: ~\$0.50-0.60/小时"
echo "  按需价格: \$1.65/小时"
echo ""
echo "⚠️  重要提醒:"
echo "  • F1 实例使用 xcvu9p 设备"
echo "  • 与 AWS AFI 服务完全兼容"
echo "  • 构建完成后请立即清理实例以节省成本"
echo ""
echo "下一步:"
echo "  1. 上传项目: ./run_fpga_flow.sh aws-upload"
echo "  2. 启动构建: ./run_fpga_flow.sh aws-build"
echo "  3. 监控进度: ./run_fpga_flow.sh aws-monitor"
