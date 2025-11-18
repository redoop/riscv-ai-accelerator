#!/bin/bash
# 使用预装 Vivado 的 AMI 启动 F1 按需实例
# F1 实例使用 xcvu9p 设备，与 AWS AFI 服务兼容

set -e

echo "=== 启动 F1 按需实例（Vivado 2024.1 预装）==="
echo ""

# 配置参数
INSTANCE_TYPE="f1.2xlarge"
KEY_NAME="fpga-f2-key"
REGION="us-east-1"
AMI_ID="ami-092fc5deb8f3c0f7d"  # FPGA Developer AMI (Ubuntu) 1.16.1 - Vivado 2024.1
SECURITY_GROUP="sg-03d27449f82b54360"

# 可用区列表（按推荐顺序 - us-east-1a 不支持 F1）
AVAILABILITY_ZONES=("us-east-1b" "us-east-1c" "us-east-1d" "us-east-1e")

echo "配置信息:"
echo "  AMI: $AMI_ID"
echo "  AMI 名称: FPGA Developer AMI (Ubuntu) 1.16.1"
echo "  Vivado 版本: 2024.1"
echo "  实例类型: $INSTANCE_TYPE (xcvu9p - F1 兼容)"
echo "  计费方式: 按需实例（\$1.65/小时）"
echo "  可用区: 自动选择（${AVAILABILITY_ZONES[*]}）"
echo ""
echo "⚠️  重要: F1 实例使用 xcvu9p 设备，与 AWS AFI 服务兼容"
echo ""

# 尝试多个可用区
INSTANCE_ID=""
SELECTED_AZ=""

for AZ in "${AVAILABILITY_ZONES[@]}"; do
    echo "尝试可用区: $AZ"
    echo "  正在请求实例..."
    
    # 尝试启动实例，捕获输出和错误，添加超时
    INSTANCE_OUTPUT=$(timeout 60 aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SECURITY_GROUP \
        --placement "AvailabilityZone=$AZ" \
        --region $REGION \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=F1-FPGA-OnDemand},{Key=Project,Value=RISCV-AI-Accelerator},{Key=Type,Value=OnDemand}]" \
        --query 'Instances[0].InstanceId' \
        --output text 2>&1)
    
    EXIT_CODE=$?
    
    # 检查超时
    if [ $EXIT_CODE -eq 124 ]; then
        echo "  ✗ $AZ 请求超时（60秒）"
        echo "     原因: AWS API 响应超时"
        sleep 2
        continue
    fi
    
    # 调试输出
    echo "  调试: EXIT_CODE=$EXIT_CODE"
    echo "  调试: OUTPUT=${INSTANCE_OUTPUT:0:100}..."
    
    # 检查是否成功
    if [ $EXIT_CODE -eq 0 ] && [ -n "$INSTANCE_OUTPUT" ] && [ "$INSTANCE_OUTPUT" != "None" ] && [[ "$INSTANCE_OUTPUT" != *"error"* ]] && [[ "$INSTANCE_OUTPUT" != *"Error"* ]] && [[ "$INSTANCE_OUTPUT" =~ ^i-[0-9a-f]+ ]]; then
        INSTANCE_ID="$INSTANCE_OUTPUT"
        SELECTED_AZ="$AZ"
        echo "  ✓ 成功在 $AZ 启动实例: $INSTANCE_ID"
        break
    else
        # 提取错误信息（多种方式）
        ERROR_MSG=""
        
        # 方式1: 提取括号内容
        ERROR_MSG=$(echo "$INSTANCE_OUTPUT" | grep -oP '(?<=\().*?(?=\))' | head -1)
        
        # 方式2: 提取 "An error occurred" 后的内容
        if [ -z "$ERROR_MSG" ]; then
            ERROR_MSG=$(echo "$INSTANCE_OUTPUT" | grep -oP 'An error occurred.*' | head -1)
        fi
        
        # 方式3: 使用完整输出
        if [ -z "$ERROR_MSG" ]; then
            ERROR_MSG="$INSTANCE_OUTPUT"
        fi
        
        # 截断过长的错误信息
        if [ ${#ERROR_MSG} -gt 200 ]; then
            ERROR_MSG="${ERROR_MSG:0:200}..."
        fi
        
        echo "  ✗ $AZ 不可用"
        echo "     原因: $ERROR_MSG"
        INSTANCE_ID=""
        
        echo "  等待 2 秒后尝试下一个可用区..."
        sleep 2
    fi
done

if [ -z "$INSTANCE_ID" ]; then
    echo ""
    echo "❌ 所有可用区都无法启动按需实例"
    echo ""
    echo "可能的原因:"
    echo "  1. 配额限制（F1 实例需要特殊配额）"
    echo "  2. 账户权限问题"
    echo "  3. F1 实例在该区域暂时不可用"
    echo "  4. 安全组或密钥配置错误"
    echo ""
    echo "建议:"
    echo "  1. 检查 AWS 配额限制:"
    echo "     aws service-quotas get-service-quota \\"
    echo "       --service-code ec2 \\"
    echo "       --quota-code L-85EED4F7 \\"
    echo "       --region $REGION"
    echo ""
    echo "  2. 请求增加 F1 配额:"
    echo "     https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
    echo ""
    echo "  3. 尝试其他区域（如 us-west-2）"
    echo ""
    echo "  4. 联系 AWS 支持"
    exit 1
fi

echo ""
echo "✓ 实例 ID: $INSTANCE_ID"
echo "✓ 可用区: $SELECTED_AZ"
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

if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" == "None" ]; then
    echo "❌ 无法获取公网 IP"
    echo "实例可能没有分配公网 IP，请检查 VPC 配置"
    exit 1
fi

echo "✓ 公网 IP: $PUBLIC_IP"
echo ""

# 保存实例信息到文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f1_instance_info"

cat > "$INFO_FILE" << EOF
INSTANCE_ID=$INSTANCE_ID
SPOT_REQUEST_ID=on-demand
PUBLIC_IP=$PUBLIC_IP
KEY_NAME=$KEY_NAME
REGION=$REGION
INSTANCE_TYPE=$INSTANCE_TYPE
AVAILABILITY_ZONE=$SELECTED_AZ
DEVICE=xcvu9p
BILLING_TYPE=on-demand
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
    echo "  公网 IP: $PUBLIC_IP"
    echo "  可用区: $SELECTED_AZ"
    echo ""
    exit 1
fi

echo "✓ 实例信息已保存到: $INFO_FILE"
echo ""

# 等待 SSH 可用
echo "等待 SSH 服务启动（这可能需要 1-2 分钟）..."
SSH_READY=false
for i in {1..30}; do
    if ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        ubuntu@${PUBLIC_IP} "echo 'SSH ready'" 2>/dev/null; then
        echo "✓ SSH 服务已就绪"
        SSH_READY=true
        break
    fi
    echo "  [$i/30] 等待 SSH..."
    sleep 10
done

if [ "$SSH_READY" = false ]; then
    echo ""
    echo "⚠️  SSH 服务启动超时，但实例已运行"
    echo "请稍后手动尝试连接"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      F1 按需实例启动成功（Vivado 2024.1）！               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  实例类型: $INSTANCE_TYPE"
echo "  计费方式: 按需实例（\$1.65/小时）"
echo "  可用区: $SELECTED_AZ"
echo "  FPGA 设备: xcvu9p (F1 兼容)"
echo "  公网 IP: $PUBLIC_IP"
echo "  AMI: FPGA Developer AMI (Ubuntu) 1.16.1 - Vivado 2024.1"
echo ""
echo "连接命令:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "验证 Vivado:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'vivado -version'"
echo ""
echo "验证 AWS FPGA 环境:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'ls -la ~/aws-fpga'"
echo ""
echo "上传项目:"
echo "  scp -i ~/.ssh/${KEY_NAME}.pem fpga-project.tar.gz ubuntu@${PUBLIC_IP}:~/"
echo ""
echo "停止实例:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "成本估算:"
echo "  按需价格: \$1.65/小时"
echo "  4小时构建成本: \$6.60"
echo "  24小时成本: \$39.60"
echo ""
echo "⚠️  重要提醒:"
echo "  • F1 实例使用 xcvu9p 设备"
echo "  • 与 AWS AFI 服务完全兼容"
echo "  • 按需实例更可靠，但成本是 Spot 的 3 倍"
echo "  • 构建完成后请立即停止实例以节省成本"
echo ""
echo "下一步:"
echo "  1. 上传项目: ./run_fpga_flow.sh aws-upload"
echo "  2. 启动构建: ./run_fpga_flow.sh aws-build"
echo "  3. 监控进度: ./run_fpga_flow.sh aws-monitor"
echo "  4. 下载结果: ./run_fpga_flow.sh aws-download-dcp"
echo "  5. 创建 AFI: ./run_fpga_flow.sh aws-create-afi"
echo "  6. 清理实例: ./run_fpga_flow.sh aws-cleanup"
