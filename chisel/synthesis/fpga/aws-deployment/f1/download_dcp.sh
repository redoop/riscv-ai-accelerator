#!/bin/bash
# F1 实例 DCP 下载脚本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== F1 DCP 下载 ==="
echo ""

# 检查实例信息文件
if [ ! -f "$PARENT_DIR/.f1_instance_info" ]; then
    echo "❌ 未找到 F1 实例信息"
    echo ""
    echo "请先启动 F1 实例:"
    echo "  bash launch.sh"
    exit 1
fi

# 加载实例信息
source "$PARENT_DIR/.f1_instance_info"

echo "实例信息:"
echo "  实例 ID: $INSTANCE_ID"
echo "  公网 IP: $PUBLIC_IP"
echo "  设备: $DEVICE"
echo ""

# 创建本地目录
BUILD_DIR="$PARENT_DIR/../build/checkpoints/to_aws"
mkdir -p "$BUILD_DIR"

# F1 实例的可能路径
REMOTE_PATHS=(
    "~/aws-fpga/hdk/cl/examples/*/build/checkpoints/to_aws/*.SH_CL_routed.dcp"
    "~/fpga-project/build/checkpoints/to_aws/SH_CL_routed.dcp"
    "~/aws-fpga-build/*/build/checkpoints/to_aws/*.SH_CL_routed.dcp"
)

echo "搜索 DCP 文件..."

FOUND_PATH=""
for pattern in "${REMOTE_PATHS[@]}"; do
    RESULT=$(ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} \
        "ls $pattern 2>/dev/null | head -1" || echo "")
    
    if [ -n "$RESULT" ]; then
        FOUND_PATH="$RESULT"
        echo "✓ 找到 DCP: $FOUND_PATH"
        break
    fi
done

if [ -z "$FOUND_PATH" ]; then
    echo "❌ 未找到 DCP 文件"
    echo ""
    echo "手动搜索:"
    echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
    echo "  find ~ -name '*.SH_CL_routed.dcp' 2>/dev/null"
    exit 1
fi

# 获取文件大小
REMOTE_SIZE=$(ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP} \
    "ls -lh '$FOUND_PATH' | awk '{print \$5}'")

echo "文件大小: $REMOTE_SIZE"
echo ""

# 下载文件
echo "开始下载..."
scp -i ~/.ssh/${KEY_NAME}.pem \
    ec2-user@${PUBLIC_IP}:"$FOUND_PATH" \
    "$BUILD_DIR/SH_CL_routed.dcp"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ DCP 下载成功"
    echo "位置: $BUILD_DIR/SH_CL_routed.dcp"
    
    LOCAL_SIZE=$(ls -lh "$BUILD_DIR/SH_CL_routed.dcp" | awk '{print $5}')
    echo "本地大小: $LOCAL_SIZE"
    
    # 验证设备
    echo ""
    echo "验证 DCP 设备..."
    if command -v unzip &> /dev/null; then
        DEVICE_CHECK=$(unzip -p "$BUILD_DIR/SH_CL_routed.dcp" dcp.xml 2>/dev/null | grep -o 'xcvu[0-9]*p' | head -1)
        if [ "$DEVICE_CHECK" == "xcvu9p" ]; then
            echo "✓ 设备验证通过: $DEVICE_CHECK (F1 兼容)"
        else
            echo "⚠️  设备: $DEVICE_CHECK (可能不兼容 AFI)"
        fi
    fi
    
    echo ""
    echo "下一步:"
    echo "  1. 创建 AFI: bash create_afi.sh"
    echo "  2. 清理实例: bash cleanup.sh"
else
    echo "❌ 下载失败"
    exit 1
fi
