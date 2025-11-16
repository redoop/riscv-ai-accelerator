#!/bin/bash
# 简化的 AFI 创建脚本

set -e

REGION="us-east-1"
BUILD_DIR="chisel/synthesis/fpga/build_results"
OUTPUT_DIR="chisel/synthesis/fpga/aws-deployment/output"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
AFI_NAME="riscv-ai-${TIMESTAMP}"
S3_BUCKET="fpga-afi-${TIMESTAMP}"
TAR_FILE="$OUTPUT_DIR/${AFI_NAME}.tar"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         创建 AFI                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 验证文件
if [ ! -f "$BUILD_DIR/SH_CL_routed.dcp" ]; then
    echo "❌ DCP 文件不存在"
    exit 1
fi

if [ ! -f "$BUILD_DIR/manifest" ]; then
    echo "❌ manifest 文件不存在"
    exit 1
fi

echo "✓ DCP: $(du -h $BUILD_DIR/SH_CL_routed.dcp | cut -f1)"
echo "✓ manifest 存在"
echo ""

# 显示 manifest
echo "📄 Manifest:"
cat "$BUILD_DIR/manifest"
echo ""

# 创建 tar
echo "📦 创建 tar..."
tar -cvf "$TAR_FILE" -C "$BUILD_DIR" SH_CL_routed.dcp manifest
echo ""
echo "✓ Tar: $(du -h $TAR_FILE | cut -f1)"
echo ""
echo "📋 Tar 内容:"
tar -tf "$TAR_FILE"
echo ""

# 创建 S3 bucket
echo "📦 创建 S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null || echo "Bucket 已存在"
echo ""

# 上传
echo "📤 上传到 S3..."
S3_KEY="dcp/${AFI_NAME}.tar"
aws s3 cp "$TAR_FILE" s3://$S3_BUCKET/$S3_KEY --region $REGION
echo "✓ 上传完成"
echo ""

# 验证上传
echo "🔍 验证 S3 文件..."
aws s3 cp s3://$S3_BUCKET/$S3_KEY /tmp/verify.tar --region $REGION --quiet
tar -tf /tmp/verify.tar
rm /tmp/verify.tar
echo "✓ 验证通过"
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
    --output text)

if [[ "$AFI_ID" != afi-* ]]; then
    echo "❌ 创建失败: $AFI_ID"
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

# 检查初始状态
echo "🔍 检查状态..."
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
EOF

echo "✓ 信息已保存到: $OUTPUT_DIR/afi_info_${TIMESTAMP}.txt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ AFI 创建成功！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "AFI ID: $AFI_ID"
echo "AGFI ID: $AGFI_ID"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "监控命令（预计 30-60 分钟）:"
echo "  watch -n 60 'aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --region $REGION --query \"FpgaImages[0].[State.Code,State.Message]\" --output text'"
