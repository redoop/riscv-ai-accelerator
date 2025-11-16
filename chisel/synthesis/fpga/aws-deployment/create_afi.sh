#!/bin/bash
# 使用 AWS 官方 manifest 格式创建 AFI

set -e

REGION="us-east-1"
DCP_FILE="../build_results/SH_CL_routed.dcp"
OUTPUT_DIR="output"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
AFI_NAME="riscv-ai-${TIMESTAMP}"
S3_BUCKET="fpga-afi-${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         使用 AWS 官方格式创建 AFI                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$DCP_FILE" ]; then
    echo "❌ DCP 不存在"
    exit 1
fi

echo "✓ DCP: $(du -h $DCP_FILE | cut -f1)"

# 创建临时目录
TEMP_DIR="$OUTPUT_DIR/afi_temp_${TIMESTAMP}"
mkdir -p $TEMP_DIR
cp $DCP_FILE $TEMP_DIR/
echo "✓ 临时目录: $TEMP_DIR"
echo ""

# 创建 manifest - 文件名必须是 "manifest" (无扩展名)
echo "📝 创建 manifest..."

# 计算 hash
DCP_HASH=$(md5 -q $TEMP_DIR/SH_CL_routed.dcp)
DATE_STR=$(date +%Y/%m/%d)

# 创建 manifest (无扩展名，键值对格式，LF 换行)
cat > $TEMP_DIR/manifest << EOF
manifest_format_version=2
pci_vendor_id=0x1D0F
pci_device_id=0xF000
subsystem_id=0x1D51
subsystem_vendor_id=0xFEDD
dcp_hash=${DCP_HASH}
shell_version=0x04261818
dcp_file_name=SH_CL_routed.dcp
hdk_version=1.4.23
date=${DATE_STR}
clock_main_a0=250
clock_extra_b0=125
clock_extra_c0=375
EOF

# 确保 LF 换行符（移除可能的 CRLF）
sed -i '' -e $'s/\r$//' $TEMP_DIR/manifest 2>/dev/null || sed -i -e $'s/\r$//' $TEMP_DIR/manifest

echo "Manifest 内容:"
cat $TEMP_DIR/manifest
echo ""
echo "文件信息:"
file $TEMP_DIR/manifest
echo ""

# 创建 tar (manifest 必须在根目录)
echo "📦 创建 tar..."
TAR_FILE="$(pwd)/$OUTPUT_DIR/${AFI_NAME}.tar"
(cd $TEMP_DIR && tar -cvf "$TAR_FILE" SH_CL_routed.dcp manifest)
echo ""
echo "✓ Tar: $(du -h $TAR_FILE | cut -f1)"

# 验证 tar
echo ""
echo "验证 tar 内容:"
tar -tvf $TAR_FILE
echo ""

# 提取并验证 manifest
echo "从 tar 中提取 manifest 验证:"
tar -xOf $TAR_FILE manifest | head -5
echo ""

# 清理临时目录
rm -rf $TEMP_DIR
echo "✓ 临时目录已清理"
echo ""

# 创建 S3 bucket
echo "📦 创建 S3 bucket..."
if aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null; then
    echo "✓ Bucket: $S3_BUCKET"
else
    echo "✓ Bucket 已存在"
fi

# 上传
echo ""
echo "📤 上传到 S3..."
S3_KEY="dcp/${AFI_NAME}.tar"
aws s3 cp $TAR_FILE s3://$S3_BUCKET/$S3_KEY
echo "✓ 上传完成"
echo ""

# 创建 AFI
echo "🔨 创建 AFI..."
AFI_ID=$(aws ec2 create-fpga-image \
    --region $REGION \
    --name $AFI_NAME \
    --description "RISC-V AI Accelerator FPGA Image" \
    --input-storage-location Bucket=$S3_BUCKET,Key=$S3_KEY \
    --logs-storage-location Bucket=$S3_BUCKET,Key=logs \
    --query 'FpgaImageId' \
    --output text 2>&1)

if [[ "$AFI_ID" != afi-* ]]; then
    echo "❌ 创建失败: $AFI_ID"
    exit 1
fi

echo "✓ AFI ID: $AFI_ID"
echo ""

# 获取 AGFI
echo "获取 AGFI..."
sleep 10
AGFI_ID=$(aws ec2 describe-fpga-images \
    --region $REGION \
    --fpga-image-ids $AFI_ID \
    --query 'FpgaImages[0].FpgaImageGlobalId' \
    --output text)

echo "✓ AGFI ID: $AGFI_ID"
echo ""

# 保存信息
AFI_INFO="$OUTPUT_DIR/afi_info_${TIMESTAMP}.txt"
cat > $AFI_INFO << EOF
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

echo "✓ 信息已保存: $AFI_INFO"
echo ""

# 立即检查状态
echo "检查初始状态..."
RESULT=$(aws ec2 describe-fpga-images \
    --region $REGION \
    --fpga-image-ids $AFI_ID \
    --query 'FpgaImages[0].[State.Code,State.Message]' \
    --output text)

STATUS=$(echo "$RESULT" | cut -f1)
MSG=$(echo "$RESULT" | cut -f2-)

echo "状态: $STATUS"
if [ "$MSG" != "None" ] && [ -n "$MSG" ]; then
    echo "消息: $MSG"
fi
echo ""

if [ "$STATUS" == "failed" ]; then
    echo "❌ AFI 立即失败"
    echo ""
    echo "查看详细日志:"
    echo "  aws s3 ls s3://$S3_BUCKET/logs/ --recursive --region $REGION"
    echo "  aws s3 cp s3://$S3_BUCKET/logs/afi-${AFI_ID}/State - --region $REGION"
    echo ""
    echo "尝试下载并检查 tar 包:"
    echo "  aws s3 cp s3://$S3_BUCKET/$S3_KEY /tmp/test.tar --region $REGION"
    echo "  tar -xOf /tmp/test.tar manifest.txt"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              AFI 创建请求成功！                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 监控
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "监控 AFI 创建进度（30-60 分钟）"
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
    if [ "$MSG" != "None" ] && [ -n "$MSG" ]; then
        echo "  └─ $MSG"
    fi
    
    if [ "$STATUS" == "available" ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║              🎉 AFI 可用！                                ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "总耗时: ${ELAPSED} 分钟"
        echo "AGFI ID: $AGFI_ID"
        echo ""
        echo "下一步: ./launch_f1_for_testing.sh"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo ""
        echo "❌ AFI 创建失败"
        echo "日志: s3://$S3_BUCKET/logs"
        exit 1
    fi
    
    sleep 60
done
