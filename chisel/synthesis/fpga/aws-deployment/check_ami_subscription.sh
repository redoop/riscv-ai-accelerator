#!/bin/bash
# 检查 FPGA Developer AMI 订阅状态

set -e

echo "=== 检查 FPGA Developer AMI 订阅状态 ==="
echo ""

REGION="us-east-1"

# 测试 Rocky Linux 版本
echo "1. 检查 FPGA Developer AMI 1.18.0 (Rocky Linux)..."
AMI_ROCKY="ami-0cb1b6ae2ff99f8bf"
RESULT_ROCKY=$(aws ec2 run-instances --dry-run \
    --image-id $AMI_ROCKY \
    --instance-type f1.2xlarge \
    --key-name fpga-dev-key \
    --region $REGION 2>&1 || true)

if echo "$RESULT_ROCKY" | grep -q "DryRunOperation"; then
    echo "   ✅ 已订阅"
    SUBSCRIBED_AMI=$AMI_ROCKY
    SUBSCRIBED_NAME="FPGA Developer AMI 1.18.0 (Rocky Linux)"
elif echo "$RESULT_ROCKY" | grep -q "OptInRequired"; then
    echo "   ❌ 未订阅"
    echo "   订阅链接: https://aws.amazon.com/marketplace/pp?sku=dhd5uoidkh9mqmlv5jargxaqt"
else
    echo "   ⚠️  检查失败: $RESULT_ROCKY"
fi

echo ""

# 测试 Ubuntu 版本
echo "2. 检查 FPGA Developer AMI 1.17.0 (Ubuntu)..."
AMI_UBUNTU="ami-01198b89d80ebfdd2"
RESULT_UBUNTU=$(aws ec2 run-instances --dry-run \
    --image-id $AMI_UBUNTU \
    --instance-type f1.2xlarge \
    --key-name fpga-dev-key \
    --region $REGION 2>&1 || true)

if echo "$RESULT_UBUNTU" | grep -q "DryRunOperation"; then
    echo "   ✅ 已订阅"
    if [ -z "$SUBSCRIBED_AMI" ]; then
        SUBSCRIBED_AMI=$AMI_UBUNTU
        SUBSCRIBED_NAME="FPGA Developer AMI 1.17.0 (Ubuntu)"
    fi
elif echo "$RESULT_UBUNTU" | grep -q "OptInRequired"; then
    echo "   ❌ 未订阅"
    echo "   订阅链接: https://aws.amazon.com/marketplace/pp?sku=e4txuxx6uz6371b7tgmotozac"
else
    echo "   ⚠️  检查失败: $RESULT_UBUNTU"
fi

echo ""
echo "=== 检查结果 ==="

if [ -n "$SUBSCRIBED_AMI" ]; then
    echo "✅ 找到已订阅的 AMI"
    echo ""
    echo "AMI 信息:"
    echo "  AMI ID: $SUBSCRIBED_AMI"
    echo "  名称: $SUBSCRIBED_NAME"
    echo ""
    echo "可以继续启动 F1 实例:"
    echo "  ./launch_f1_instance.sh"
    echo ""
    
    # 保存到配置文件
    cat > ami_config.txt << EOF
SUBSCRIBED_AMI=$SUBSCRIBED_AMI
SUBSCRIBED_NAME=$SUBSCRIBED_NAME
SUBSCRIPTION_DATE=$(date)
EOF
    echo "✓ AMI 配置已保存到 ami_config.txt"
else
    echo "❌ 未找到已订阅的 FPGA Developer AMI"
    echo ""
    echo "请选择一个版本进行订阅:"
    echo ""
    echo "选项 1: Rocky Linux 版本（推荐）"
    echo "  订阅链接: https://aws.amazon.com/marketplace/pp?sku=dhd5uoidkh9mqmlv5jargxaqt"
    echo "  AMI ID: ami-0cb1b6ae2ff99f8bf"
    echo ""
    echo "选项 2: Ubuntu 版本"
    echo "  订阅链接: https://aws.amazon.com/marketplace/pp?sku=e4txuxx6uz6371b7tgmotozac"
    echo "  AMI ID: ami-01198b89d80ebfdd2"
    echo ""
    echo "订阅步骤:"
    echo "  1. 在浏览器中打开订阅链接"
    echo "  2. 点击 'Continue to Subscribe'"
    echo "  3. 接受条款"
    echo "  4. 等待 1-2 分钟"
    echo "  5. 重新运行此脚本验证"
    echo ""
    exit 1
fi
