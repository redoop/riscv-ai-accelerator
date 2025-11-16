#!/bin/bash
# 验证并创建 AFI - 确保 manifest 正确

set -e

REGION="us-east-1"
BUILD_DIR="chisel/synthesis/fpga/build_results"
OUTPUT_DIR="chisel/synthesis/fpga/aws-deployment/output"
DCP_FILE="$BUILD_DIR/SH_CL_routed.dcp"
MANIFEST_FILE="$BUILD_DIR/manifest"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
AFI_NAME="riscv-ai-${TIMESTAMP}"
S3_BUCKET="fpga-afi-${TIMESTAMP}"
TAR_FILE="$OUTPUT_DIR/${AFI_NAME}.tar"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         验证并创建 AFI                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 验证文件存在
if [ ! -f "$DCP_FILE" ]; then
    echo "❌ DCP 文件不存在: $DCP_FILE"
    exit 1
fi

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "❌ Manifest 文件不存在: $MANIFEST_FILE"
    exit 1
fi

echo "✓ DCP: $(du -h $DCP_FILE | cut -f1)"
echo "✓ Manifest 存在"
echo ""

# 显示 manifest 内容
echo "📄 Manifest 内容:"
cat "$MANIFEST_FILE"
echo ""

# 创建临时目录用于打包
TEMP_DIR=$(mktemp -d)
echo "📦 准备打包到临时目录: $TEMP_DIR"
cp "$DCP_FILE" "$TEMP_DIR/"
cp "$MANIFEST_FILE" "$TEMP_DIR/manifest"  # 重命名为 manifest（无扩展名）

# 创建 tar（从临时目录，确保文件在根目录）
echo "📦 创建 tar..."
tar -cvf "$TAR_FILE" -C "$TEMP_DIR" SH_CL_routed.dcp manifest
rm -rf "$TEMP_DIR"

echo ""
echo "✓ Tar 创建完成: $(du -h $TAR_FILE | cut -f1)"
echo ""
echo "📋 Tar 内容:"
tar -tf "$TAR_FILE"
echo ""

# 验证 tar 内容
echo "🔍 验证 tar 内容..."
tar -xf "$TAR_FILE" -C /tmp manifest
echo "Manifest 第一行:"
head -1 /tmp/manifest
rm /tmp/manifest

# 创建 S3 bucket
echo ""
echo "📦 创建 S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null || echo "Bucket 已存在"

# 上传
echo ""
echo "📤 上传到 S3..."
S3_KEY="dcp/${AFI_NAME}.tar"
aws s3 cp "$TAR_FILE" s3://$S3_BUCKET/$S3_KEY --region $REGION
echo "✓ 上传完成: s3://$S3_BUCKET/$S3_KEY"
echo ""

# 验证上传的文件
echo "🔍 验证 S3 上的文件..."
aws s3 cp s3://$S3_BUCKET/$S3_KEY /tmp/verify.tar --region $REGION
tar -tf /tmp/verify.tar
rm /tmp/verify.tar
echo ""

# 创建 AFI
echo "🔨 创建 AFI..."
AFI_ID=$(aws ec2 create-fpga-image \
    --region $REGION \
    --name $AFI_NAME \
    --description "RISC-V AI Accelerator" \
    --input-storage-location Bucket=$S3_BUCKET,Key=$S3_KEY \
    --logs-storage-location Bucket=$S3_BUCKET,Key=logs \
    --query 'FpgaImageId' \
    --output text 2>&1)

if [ -z "$AFI_ID" ] || [[ "$AFI_ID" != afi-* ]]; then
    echo "❌ 创建失败:"
    echo "$AFI_ID"
    exit 1
fi

echo "✓ AFI ID: $AFI_ID"
sleep 5

AGFI_ID=$(aws ec2 describe-fpga-images \
    --region $REGION \
    --fpga-image-ids $AFI_ID \
    --query 'FpgaImages[0].FpgaImageGlobalId' \
    --output text)

echo "✓ AGFI ID: $AGFI_ID"
echo ""

# 立即检查状态
echo "🔍 检查初始状态..."
RESULT=$(aws ec2 describe-fpga-images \
    --region $REGION \
    --fpga-image-ids $AFI_ID \
    --query 'FpgaImages[0].[State.Code,State.Message]' \
    --output text)

STATUS=$(echo "$RESULT" | cut -f1)
MSG=$(echo "$RESULT" | cut -f2-)

echo "状态: $STATUS"
if [ "$MSG" != "None" ]; then
    echo "消息: $MSG"
fi
echo ""

if [ "$STATUS" == "failed" ]; then
    echo "❌ AFI 立即失败"
    echo ""
    echo "检查日志:"
    echo "  aws s3 ls s3://$S3_BUCKET/logs/ --recursive --region $REGION"
    exit 1
fi

# 保存信息
cat > "$OUTPUT_DIR/afi_info_${TIMESTAMP}.txt" << EOF
AFI 信息
========
时间: $(date)
AFI ID: $AFI_ID
AGFI ID: $AGFI_ID
S3 Bucket: $S3_BUCKET
S3 Key: $S3_KEY

检查状态:
  aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --region $REGION

加载到 F1:
  sudo fpga-load-local-image -S 0 -I $AGFI_ID
  sudo fpga-describe-local-image -S 0 -H
EOF

echo "✓ 信息已保存到: $OUTPUT_DIR/afi_info_${TIMESTAMP}.txt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "AFI 创建成功，开始监控（预计 30-60 分钟）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ITERATION=0
START=$(date +%s)

while true; do
    ITERATION=$((ITERATION + 1))
    ELAPSED=$((($(date +%s) - START) / 60))
    
    RESULT=$(aws ec2 describe-fpga-images \
        --region $REGION \
        --fpga-image-ids $AFI_ID \
        --query 'FpgaImages[0].[State.Code,State.Message]' \
        --output text)
    
    STATUS=$(echo "$RESULT" | cut -f1)
    MSG=$(echo "$RESULT" | cut -f2-)
    
    echo "[$(date +%H:%M:%S)] #$ITERATION (${ELAPSED}分钟) - $STATUS"
    [ "$MSG" != "None" ] && echo "  └─ $MSG"
    
    if [ "$STATUS" == "available" ]; then
        echo ""
        echo "🎉 AFI 可用！(${ELAPSED}分钟)"
        echo "AGFI: $AGFI_ID"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo ""
        echo "❌ 失败"
        echo "查看日志:"
        echo "  aws s3 ls s3://$S3_BUCKET/logs/ --recursive --region $REGION"
        exit 1
    fi
    
    sleep 60
done
