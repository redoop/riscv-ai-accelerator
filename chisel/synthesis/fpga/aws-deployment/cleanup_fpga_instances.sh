#!/bin/bash
# 清理所有 AWS F1/F2 FPGA 实例和相关资源

set -e

REGION="${1:-us-east-1}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           清理 AWS FPGA 实例和资源                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "区域: $REGION"
echo "时间: $(date)"
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI 未安装"
    exit 1
fi

# 检查凭证
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS 凭证未配置"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS 账户: $ACCOUNT_ID"
echo ""

# 1. 查找所有 F1/F2 实例
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 查找 F1/F2 实例..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INSTANCE_IDS=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=instance-type,Values=f1.*,f2.*" \
              "Name=instance-state-name,Values=running,pending,stopped,stopping" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text)

if [ -z "$INSTANCE_IDS" ]; then
    echo "✓ 没有找到运行中的 F1/F2 实例"
else
    echo "找到以下实例:"
    aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $INSTANCE_IDS \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`].Value|[0]]' \
        --output table
    
    echo ""
    read -p "确认终止这些实例？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "终止实例..."
        aws ec2 terminate-instances \
            --instance-ids $INSTANCE_IDS \
            --region $REGION \
            --output table
        echo "✓ 实例已终止"
    else
        echo "⊘ 跳过实例终止"
    fi
fi
echo ""

# 2. 取消所有 Spot 请求
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. 取消 Spot 请求..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SPOT_IDS=$(aws ec2 describe-spot-instance-requests \
    --region $REGION \
    --filters "Name=state,Values=open,active" \
    --query 'SpotInstanceRequests[*].SpotInstanceRequestId' \
    --output text)

if [ -z "$SPOT_IDS" ]; then
    echo "✓ 没有找到活动的 Spot 请求"
else
    echo "找到以下 Spot 请求:"
    aws ec2 describe-spot-instance-requests \
        --region $REGION \
        --spot-instance-request-ids $SPOT_IDS \
        --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,State,InstanceType,CreateTime]' \
        --output table
    
    echo ""
    echo "取消 Spot 请求..."
    aws ec2 cancel-spot-instance-requests \
        --spot-instance-request-ids $SPOT_IDS \
        --region $REGION
    echo "✓ Spot 请求已取消"
fi
echo ""

# 3. 清理安全组（可选）
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 检查安全组..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SG_ID="sg-03d27449f82b54360"
if aws ec2 describe-security-groups --group-ids $SG_ID --region $REGION &> /dev/null; then
    echo "找到安全组: $SG_ID (fpga-dev-sg)"
    echo "⊘ 保留安全组（可用于下次部署）"
else
    echo "✓ 安全组不存在或已删除"
fi
echo ""

# 4. 最终验证
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. 最终验证..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

REMAINING_INSTANCES=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=instance-type,Values=f1.*,f2.*" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[*].Instances[*].InstanceId' \
    --output text)

REMAINING_SPOTS=$(aws ec2 describe-spot-instance-requests \
    --region $REGION \
    --filters "Name=state,Values=open,active" \
    --query 'SpotInstanceRequests[*].SpotInstanceRequestId' \
    --output text)

if [ -z "$REMAINING_INSTANCES" ] && [ -z "$REMAINING_SPOTS" ]; then
    echo "✅ 清理完成！没有剩余的 F1/F2 实例或 Spot 请求"
else
    echo "⚠️ 仍有资源存在:"
    [ -n "$REMAINING_INSTANCES" ] && echo "  实例: $REMAINING_INSTANCES"
    [ -n "$REMAINING_SPOTS" ] && echo "  Spot 请求: $REMAINING_SPOTS"
fi
echo ""

# 5. 生成清理报告
REPORT_FILE="../build/cleanup_report_$(date +%Y%m%d_%H%M%S).txt"
cat > $REPORT_FILE << EOFR
AWS FPGA 实例清理报告

时间: $(date)
区域: $REGION
账户: $ACCOUNT_ID

已清理:
  - F1/F2 实例: $(echo $INSTANCE_IDS | wc -w) 个
  - Spot 请求: $(echo $SPOT_IDS | wc -w) 个

保留资源:
  - 安全组: sg-03d27449f82b54360
  - SSH 密钥: fpga-f2-key
  - VPC: vpc-0282f8e2e326aeef2

下次部署:
  cd chisel/synthesis/fpga/aws-deployment
  ./launch_f2_vivado.sh
EOFR

echo "✓ 清理报告已保存: $REPORT_FILE"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  清理完成！                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
