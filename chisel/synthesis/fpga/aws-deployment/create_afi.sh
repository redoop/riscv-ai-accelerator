#!/bin/bash
# 创建 AWS AFI (Amazon FPGA Image)

set -e

# 加载配置
if [ -f "$HOME/.fpga_config" ]; then
    source $HOME/.fpga_config
else
    echo "错误：未找到 FPGA 配置，请先运行 setup_aws.sh"
    exit 1
fi

echo "=========================================="
echo "创建 AWS AFI"
echo "=========================================="

# 检查 DCP 文件
DCP_FILE="./build/checkpoints/to_aws/SH_CL_routed.dcp"
if [ ! -f "$DCP_FILE" ]; then
    echo "错误：未找到 DCP 文件，请先运行 build_fpga.tcl"
    exit 1
fi

# 设置参数
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
AFI_NAME="riscv_ai_accel_${TIMESTAMP}"
S3_BUCKET=${AFI_BUCKET:-"fpga-afi-bucket"}
S3_DCP_KEY="dcp/${AFI_NAME}.dcp"
S3_LOGS_KEY="logs/${AFI_NAME}"

echo "AFI 名称：$AFI_NAME"
echo "S3 Bucket：$S3_BUCKET"

# 创建 S3 bucket（如果不存在）
aws s3 mb s3://$S3_BUCKET 2>/dev/null || echo "Bucket 已存在"

# 上传 DCP 到 S3
echo "上传 DCP 文件到 S3..."
aws s3 cp $DCP_FILE s3://$S3_BUCKET/$S3_DCP_KEY

# 创建 AFI
echo "创建 AFI（这可能需要 30-60 分钟）..."
AFI_ID=$(aws ec2 create-fpga-image \
    --name $AFI_NAME \
    --description "RISC-V AI Accelerator FPGA Image" \
    --input-storage-location Bucket=$S3_BUCKET,Key=$S3_DCP_KEY \
    --logs-storage-location Bucket=$S3_BUCKET,Key=$S3_LOGS_KEY \
    --query 'FpgaImageId' \
    --output text)

echo "AFI ID: $AFI_ID"

# 获取 AGF ID（全局唯一标识符）
sleep 5
AGFI_ID=$(aws ec2 describe-fpga-images \
    --fpga-image-ids $AFI_ID \
    --query 'FpgaImages[0].FpgaImageGlobalId' \
    --output text)

echo "AGFI ID: $AGFI_ID"

# 保存 AFI 信息
AFI_INFO_FILE="./build/afi_info.txt"
cat > $AFI_INFO_FILE << EOF
AFI 创建信息
============
创建时间：$TIMESTAMP
AFI ID：$AFI_ID
AGFI ID：$AGFI_ID
S3 Bucket：$S3_BUCKET
DCP 路径：s3://$S3_BUCKET/$S3_DCP_KEY
日志路径：s3://$S3_BUCKET/$S3_LOGS_KEY

加载命令：
sudo fpga-load-local-image -S 0 -I $AGFI_ID

查看状态：
aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID
EOF

cat $AFI_INFO_FILE

echo ""
echo "=========================================="
echo "AFI 创建请求已提交！"
echo "=========================================="
echo "AFI ID: $AFI_ID"
echo "AGFI ID: $AGFI_ID"
echo ""
echo "检查 AFI 状态："
echo "  aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID"
echo ""
echo "等待状态变为 'available'（约 30-60 分钟）"
echo ""
echo "AFI 信息已保存到：$AFI_INFO_FILE"
echo ""

# 监控 AFI 状态
echo "开始监控 AFI 状态（按 Ctrl+C 退出）..."
while true; do
    STATUS=$(aws ec2 describe-fpga-images \
        --fpga-image-ids $AFI_ID \
        --query 'FpgaImages[0].State.Code' \
        --output text)
    
    echo "[$(date +%H:%M:%S)] AFI 状态: $STATUS"
    
    if [ "$STATUS" == "available" ]; then
        echo ""
        echo "=========================================="
        echo "AFI 创建成功！"
        echo "=========================================="
        echo "可以使用以下命令加载 AFI："
        echo "  sudo fpga-load-local-image -S 0 -I $AGFI_ID"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo ""
        echo "错误：AFI 创建失败"
        echo "请检查日志：s3://$S3_BUCKET/$S3_LOGS_KEY"
        exit 1
    fi
    
    sleep 60
done
