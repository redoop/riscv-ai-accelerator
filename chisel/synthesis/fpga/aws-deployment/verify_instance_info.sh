#!/bin/bash
# 验证 F1 实例信息文件

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f1_instance_info"

echo "=== F1 实例信息验证工具 ==="
echo ""

# 检查文件是否存在
if [ ! -f "$INFO_FILE" ]; then
    echo "❌ 实例信息文件不存在"
    echo "文件路径: $INFO_FILE"
    echo ""
    echo "可能的原因:"
    echo "  1. 尚未启动 F1 实例"
    echo "  2. 实例启动失败"
    echo "  3. 文件创建失败"
    echo ""
    echo "启动 F1 实例:"
    echo "  ./launch_f1_vivado.sh      # Spot 实例"
    echo "  ./launch_f1_ondemand.sh    # 按需实例"
    echo "  ./launch_fpga_instance.sh  # 交互式选择"
    exit 1
fi

echo "✓ 实例信息文件存在"
echo ""

# 加载实例信息
source "$INFO_FILE"

# 验证必需字段
REQUIRED_FIELDS=(
    "INSTANCE_ID"
    "PUBLIC_IP"
    "KEY_NAME"
    "REGION"
    "INSTANCE_TYPE"
    "AVAILABILITY_ZONE"
    "DEVICE"
    "TIMESTAMP"
)

MISSING_FIELDS=()

for field in "${REQUIRED_FIELDS[@]}"; do
    if [ -z "${!field}" ]; then
        MISSING_FIELDS+=("$field")
    fi
done

if [ ${#MISSING_FIELDS[@]} -gt 0 ]; then
    echo "❌ 实例信息文件不完整"
    echo ""
    echo "缺少字段:"
    for field in "${MISSING_FIELDS[@]}"; do
        echo "  - $field"
    done
    echo ""
    echo "请重新启动实例"
    exit 1
fi

echo "✓ 所有必需字段都存在"
echo ""

# 显示实例信息
echo "=== 实例信息 ==="
echo "实例 ID:      $INSTANCE_ID"
echo "实例类型:     $INSTANCE_TYPE"
echo "公网 IP:      $PUBLIC_IP"
echo "可用区:       $AVAILABILITY_ZONE"
echo "区域:         $REGION"
echo "FPGA 设备:    $DEVICE"
echo "密钥名称:     $KEY_NAME"

if [ -n "$BILLING_TYPE" ]; then
    echo "计费类型:     $BILLING_TYPE"
fi

if [ -n "$SPOT_REQUEST_ID" ]; then
    echo "Spot 请求 ID: $SPOT_REQUEST_ID"
fi

echo "创建时间:     $TIMESTAMP"
echo ""

# 检查实例状态
echo "=== 检查 AWS 实例状态 ==="

if ! command -v aws &> /dev/null; then
    echo "⚠️  AWS CLI 未安装，跳过状态检查"
    exit 0
fi

INSTANCE_STATE=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "❌ 无法查询实例状态"
    echo ""
    echo "可能的原因:"
    echo "  1. AWS 凭证未配置"
    echo "  2. 实例已被删除"
    echo "  3. 网络连接问题"
    exit 1
fi

case "$INSTANCE_STATE" in
    running)
        echo "✓ 实例状态: 运行中 (running)"
        ;;
    pending)
        echo "⏳ 实例状态: 启动中 (pending)"
        ;;
    stopping)
        echo "⏸️  实例状态: 停止中 (stopping)"
        ;;
    stopped)
        echo "⏹️  实例状态: 已停止 (stopped)"
        ;;
    terminated)
        echo "❌ 实例状态: 已终止 (terminated)"
        echo ""
        echo "实例已被删除，请删除信息文件:"
        echo "  rm $INFO_FILE"
        exit 1
        ;;
    *)
        echo "⚠️  实例状态: $INSTANCE_STATE"
        ;;
esac

echo ""

# 测试 SSH 连接
echo "=== 测试 SSH 连接 ==="

if [ ! -f ~/.ssh/${KEY_NAME}.pem ]; then
    echo "⚠️  密钥文件不存在: ~/.ssh/${KEY_NAME}.pem"
    echo "请确保密钥文件存在并有正确的权限"
    exit 0
fi

# 检查密钥权限
KEY_PERMS=$(stat -c %a ~/.ssh/${KEY_NAME}.pem 2>/dev/null || stat -f %A ~/.ssh/${KEY_NAME}.pem 2>/dev/null)
if [ "$KEY_PERMS" != "400" ] && [ "$KEY_PERMS" != "600" ]; then
    echo "⚠️  密钥文件权限不正确: $KEY_PERMS"
    echo "修复权限:"
    echo "  chmod 400 ~/.ssh/${KEY_NAME}.pem"
fi

echo "测试 SSH 连接到 ubuntu@${PUBLIC_IP}..."

if ssh -i ~/.ssh/${KEY_NAME}.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    ubuntu@${PUBLIC_IP} "echo 'SSH connection successful'" 2>/dev/null; then
    echo "✓ SSH 连接成功"
else
    echo "❌ SSH 连接失败"
    echo ""
    echo "可能的原因:"
    echo "  1. 实例尚未完全启动"
    echo "  2. 安全组规则不允许 SSH"
    echo "  3. 密钥不匹配"
    echo ""
    echo "手动测试连接:"
    echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
fi

echo ""
echo "=== 验证完成 ==="
