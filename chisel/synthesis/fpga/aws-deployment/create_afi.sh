#!/bin/bash
# 使用 AWS 官方 manifest 格式创建 AFI

set -e

REGION="us-east-1"
# 尝试多个可能的 DCP 文件路径
DCP_PATHS=(
    "../build/checkpoints/to_aws/SH_CL_routed.dcp"
    "../build_results/SH_CL_routed.dcp"
    "./SH_CL_routed.dcp"
)

DCP_FILE=""
for path in "${DCP_PATHS[@]}"; do
    if [ -f "$path" ]; then
        DCP_FILE="$path"
        break
    fi
done

OUTPUT_DIR="output"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
AFI_NAME="riscv-ai-${TIMESTAMP}"
# 使用固定的 S3 bucket，在其下创建子目录
S3_BUCKET="riscv-fpga-afi"
S3_PREFIX="builds/${TIMESTAMP}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         使用 AWS 官方格式创建 AFI                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$DCP_FILE" ]; then
    echo "❌ DCP 文件不存在"
    echo ""
    echo "已检查的路径:"
    for path in "${DCP_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""
    echo "请先下载 DCP 文件:"
    echo "  cd .. && ./run_fpga_flow.sh aws-download-dcp"
    exit 1
fi

echo "✓ 找到 DCP: $DCP_FILE"
echo "✓ 文件大小: $(du -h $DCP_FILE | cut -f1)"

# 创建临时目录
TEMP_DIR="$OUTPUT_DIR/afi_temp_${TIMESTAMP}"
mkdir -p $TEMP_DIR
cp $DCP_FILE $TEMP_DIR/
echo "✓ 临时目录: $TEMP_DIR"
echo ""

# 创建 manifest.txt - 文件名必须是 "manifest.txt"
echo "📝 创建 manifest.txt..."

# 计算 SHA256 hash（兼容 macOS 和 Linux）
if command -v sha256sum &> /dev/null; then
    # Linux
    DCP_HASH=$(sha256sum $TEMP_DIR/SH_CL_routed.dcp | awk '{print $1}')
elif command -v shasum &> /dev/null; then
    # macOS
    DCP_HASH=$(shasum -a 256 $TEMP_DIR/SH_CL_routed.dcp | awk '{print $1}')
else
    echo "❌ 错误: 未找到 sha256sum 或 shasum 命令"
    exit 1
fi

# 获取日期（格式：YY_MM_DD-HHMMSS）
DATE_STR=$(date +%y_%m_%d-%H%M%S)

# 获取 shell version（从 AWS FPGA 仓库）
SHELL_VERSION="0x04261818"
if [ -f "../aws-fpga/hdk/common/shell_stable/shell_version.txt" ]; then
    SHELL_VERSION=$(cat ../aws-fpga/hdk/common/shell_stable/shell_version.txt | tr -d '\n\r')
fi

# 获取 HDK version
HDK_VERSION="1.4.23"
if [ -f "../aws-fpga/release_version.txt" ]; then
    HDK_VERSION=$(cat ../aws-fpga/release_version.txt | tr -d '\n\r')
fi

# 创建 manifest.txt（键值对格式，LF 换行）
cat > $TEMP_DIR/manifest.txt << 'MANIFEST_EOF'
manifest_format_version=2
pci_vendor_id=0x1D0F
pci_device_id=0xF000
pci_subsystem_id=0x1D51
pci_subsystem_vendor_id=0xFEDD
MANIFEST_EOF

# 追加动态内容
cat >> $TEMP_DIR/manifest.txt << MANIFEST_EOF
dcp_hash=${DCP_HASH}
shell_version=${SHELL_VERSION}
dcp_file_name=${TIMESTAMP}.SH_CL_routed.dcp
hdk_version=${HDK_VERSION}
tool_version=v2024.1
date=${DATE_STR}
clock_recipe_a=A1
clock_recipe_b=B0
clock_recipe_c=C0
clock_recipe_hbm=H0
MANIFEST_EOF

# 确保 LF 换行符（移除可能的 CRLF）
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e 's/\r$//' $TEMP_DIR/manifest.txt
else
    sed -i -e 's/\r$//' $TEMP_DIR/manifest.txt
fi

# 创建 to_aws 目录结构（AWS 要求）
echo "📦 创建 AWS 标准目录结构..."
TO_AWS_DIR="$TEMP_DIR/to_aws"
mkdir -p $TO_AWS_DIR

# 移动文件到 to_aws 目录，使用 AWS 标准命名
mv $TEMP_DIR/SH_CL_routed.dcp $TO_AWS_DIR/${TIMESTAMP}.SH_CL_routed.dcp
mv $TEMP_DIR/manifest.txt $TO_AWS_DIR/${TIMESTAMP}.manifest.txt

echo "Manifest 内容:"
cat $TO_AWS_DIR/${TIMESTAMP}.manifest.txt
echo ""
echo "文件信息:"
file $TO_AWS_DIR/${TIMESTAMP}.manifest.txt 2>/dev/null || echo "manifest.txt: ASCII text"
echo ""

# 创建 tar（打包 to_aws 目录）
echo "📦 创建 tar..."
TAR_FILE="$(pwd)/$OUTPUT_DIR/${AFI_NAME}.tar"
(cd $TEMP_DIR && tar -cvf "$TAR_FILE" to_aws/)
echo ""
echo "✓ Tar: $(du -h $TAR_FILE | cut -f1)"

# 验证 tar
echo ""
echo "验证 tar 内容:"
tar -tvf $TAR_FILE
echo ""

# 提取并验证 manifest.txt
echo "从 tar 中提取 manifest.txt 验证:"
tar -xOf $TAR_FILE to_aws/${TIMESTAMP}.manifest.txt
echo ""

# 清理临时目录
rm -rf $TEMP_DIR
echo "✓ 临时目录已清理"
echo ""

# 确保 S3 bucket 存在
echo "📦 检查 S3 bucket..."
if ! aws s3 ls s3://$S3_BUCKET --region $REGION 2>/dev/null; then
    echo "创建 S3 bucket: $S3_BUCKET"
    aws s3 mb s3://$S3_BUCKET --region $REGION
    echo "✓ Bucket 已创建"
else
    echo "✓ Bucket 已存在: $S3_BUCKET"
fi

# 上传到子目录
echo ""
echo "📤 上传到 S3..."
S3_DCP_KEY="${S3_PREFIX}/dcp/${AFI_NAME}.tar"
S3_LOGS_KEY="${S3_PREFIX}/logs"
aws s3 cp $TAR_FILE s3://$S3_BUCKET/$S3_DCP_KEY
echo "✓ 上传完成: s3://$S3_BUCKET/$S3_DCP_KEY"
echo ""

# 创建 AFI
echo "🔨 创建 AFI..."
AFI_ID=$(aws ec2 create-fpga-image \
    --region $REGION \
    --name $AFI_NAME \
    --description "RISC-V AI Accelerator FPGA Image" \
    --input-storage-location Bucket=$S3_BUCKET,Key=$S3_DCP_KEY \
    --logs-storage-location Bucket=$S3_BUCKET,Key=$S3_LOGS_KEY \
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
S3 Bucket: s3://$S3_BUCKET
S3 DCP: s3://$S3_BUCKET/$S3_DCP_KEY
S3 Logs: s3://$S3_BUCKET/$S3_LOGS_KEY

检查状态:
  aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --region $REGION

查看日志:
  aws s3 ls s3://$S3_BUCKET/$S3_LOGS_KEY/ --recursive --region $REGION

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
    echo "  aws s3 ls s3://$S3_BUCKET/$S3_LOGS_KEY/ --recursive --region $REGION"
    echo ""
    echo "尝试下载并检查 tar 包:"
    echo "  aws s3 cp s3://$S3_BUCKET/$S3_DCP_KEY /tmp/test.tar --region $REGION"
    echo "  tar -tf /tmp/test.tar"
    echo "  tar -xOf /tmp/test.tar to_aws/*.manifest.txt"
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
        echo "日志: s3://$S3_BUCKET/$S3_LOGS_KEY"
        echo ""
        echo "查看详细错误:"
        echo "  aws s3 ls s3://$S3_BUCKET/$S3_LOGS_KEY/ --recursive --region $REGION"
        exit 1
    fi
    
    sleep 60
done
