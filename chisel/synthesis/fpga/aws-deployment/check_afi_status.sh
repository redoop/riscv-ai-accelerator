#!/bin/bash
# 检查 AFI 创建状态

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}AFI 状态检查${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 查找最新的 AFI 信息文件
LATEST_AFI_INFO=$(ls -t "$OUTPUT_DIR"/afi_info_*.txt 2>/dev/null | head -1)

if [ -z "$LATEST_AFI_INFO" ] || [ ! -f "$LATEST_AFI_INFO" ]; then
    echo -e "${YELLOW}○${NC} 未找到 AFI 记录"
    echo ""
    echo -e "${BLUE}创建 AFI:${NC}"
    echo -e "  ${CYAN}./run_fpga_flow.sh aws-create-afi${NC}"
    exit 0
fi

# 提取 AFI 信息
AFI_ID=$(grep "AFI ID:" "$LATEST_AFI_INFO" | awk '{print $3}')
AGFI_ID=$(grep "AGFI ID:" "$LATEST_AFI_INFO" | awk '{print $3}')
CREATE_TIME=$(grep "时间:" "$LATEST_AFI_INFO" | cut -d: -f2- | xargs)

echo -e "${GREEN}✓${NC} 找到 AFI 记录"
echo -e "  文件:     $(basename $LATEST_AFI_INFO)"
echo -e "  AFI ID:   ${CYAN}$AFI_ID${NC}"
echo -e "  AGFI ID:  ${CYAN}$AGFI_ID${NC}"
echo -e "  创建时间: $CREATE_TIME"
echo ""

# 检查 AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} AWS CLI 未安装，无法查询状态"
    echo ""
    echo -e "${BLUE}手动查询命令:${NC}"
    echo -e "  ${CYAN}aws ec2 describe-fpga-images --fpga-image-ids $AFI_ID --region us-east-1${NC}"
    exit 0
fi

# 查询 AFI 状态
echo -e "${BLUE}查询 AWS 状态...${NC}"

RESULT=$(aws ec2 describe-fpga-images \
    --fpga-image-ids "$AFI_ID" \
    --region us-east-1 \
    --query 'FpgaImages[0].[State.Code,State.Message,CreateTime,UpdateTime]' \
    --output text 2>/dev/null)

if [ -z "$RESULT" ]; then
    echo -e "${RED}✗${NC} 无法查询 AFI 状态"
    echo ""
    echo -e "${BLUE}可能的原因:${NC}"
    echo "  1. AWS 凭证未配置"
    echo "  2. AFI ID 不存在"
    echo "  3. 网络连接问题"
    echo ""
    echo -e "${BLUE}检查 AWS 配置:${NC}"
    echo -e "  ${CYAN}aws configure list${NC}"
    exit 1
fi

STATUS=$(echo "$RESULT" | cut -f1)
MESSAGE=$(echo "$RESULT" | cut -f2)
AWS_CREATE_TIME=$(echo "$RESULT" | cut -f3)
AWS_UPDATE_TIME=$(echo "$RESULT" | cut -f4)

echo ""

# 根据状态显示不同信息
case "$STATUS" in
    "available")
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  🎉 AFI 已就绪！可以加载到 F1 实例                       ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  状态: ${GREEN}✓ $STATUS${NC}"
        echo ""
        echo -e "${BLUE}下一步操作:${NC}"
        echo ""
        echo -e "${BLUE}1. 启动 F1 实例${NC}"
        echo -e "   ${CYAN}aws ec2 run-instances \\${NC}"
        echo -e "   ${CYAN}  --image-id ami-0c55b159cbfafe1f0 \\${NC}"
        echo -e "   ${CYAN}  --instance-type f1.2xlarge \\${NC}"
        echo -e "   ${CYAN}  --key-name your-key \\${NC}"
        echo -e "   ${CYAN}  --region us-east-1${NC}"
        echo ""
        echo -e "${BLUE}2. 在 F1 实例上加载 AFI${NC}"
        echo -e "   ${CYAN}sudo fpga-clear-local-image -S 0${NC}"
        echo -e "   ${CYAN}sudo fpga-load-local-image -S 0 -I $AGFI_ID${NC}"
        echo -e "   ${CYAN}sudo fpga-describe-local-image -S 0 -H${NC}"
        echo ""
        echo -e "${BLUE}3. 运行测试${NC}"
        echo -e "   ${CYAN}# 编译并运行你的测试程序${NC}"
        ;;
        
    "pending")
        echo -e "  状态: ${YELLOW}⏳ $STATUS${NC} (生成中)"
        echo ""
        
        # 计算已用时间
        if [ -n "$AWS_CREATE_TIME" ]; then
            CREATE_EPOCH=$(date -d "$AWS_CREATE_TIME" +%s 2>/dev/null || echo "0")
            NOW_EPOCH=$(date +%s)
            
            if [ "$CREATE_EPOCH" != "0" ]; then
                ELAPSED_MIN=$(( (NOW_EPOCH - CREATE_EPOCH) / 60 ))
                REMAINING_MIN=$(( 45 - ELAPSED_MIN ))
                
                if [ $REMAINING_MIN -lt 0 ]; then
                    REMAINING_MIN=0
                fi
                
                echo -e "  已用时间: ${CYAN}$ELAPSED_MIN${NC} 分钟"
                echo -e "  预计剩余: ${YELLOW}$REMAINING_MIN${NC} 分钟 (通常 30-60 分钟)"
                
                # 进度条
                PROGRESS=$(( ELAPSED_MIN * 100 / 60 ))
                if [ $PROGRESS -gt 100 ]; then
                    PROGRESS=100
                fi
                
                FILLED=$(( PROGRESS / 2 ))
                EMPTY=$(( 50 - FILLED ))
                
                echo -n "  进度: ["
                for ((i=0; i<$FILLED; i++)); do echo -n "█"; done
                for ((i=0; i<$EMPTY; i++)); do echo -n "░"; done
                echo "] $PROGRESS%"
            fi
        fi
        
        echo ""
        echo -e "${YELLOW}⏳ AFI 正在生成中，请稍候...${NC}"
        echo ""
        echo -e "${BLUE}持续监控:${NC}"
        echo -e "  ${CYAN}watch -n 60 './run_fpga_flow.sh status'${NC}"
        echo ""
        echo -e "  或使用:"
        echo -e "  ${CYAN}watch -n 60 'bash $0'${NC}"
        ;;
        
    "failed")
        echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ❌ AFI 创建失败                                          ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  状态: ${RED}✗ $STATUS${NC}"
        
        if [ "$MESSAGE" != "None" ] && [ -n "$MESSAGE" ]; then
            echo -e "  错误: ${RED}$MESSAGE${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}查看详细日志:${NC}"
        
        S3_LOGS=$(grep "S3 Logs:" "$LATEST_AFI_INFO" | awk '{print $3}')
        if [ -n "$S3_LOGS" ]; then
            echo ""
            echo -e "${BLUE}1. 列出日志文件${NC}"
            echo -e "   ${CYAN}aws s3 ls $S3_LOGS/ --recursive --region us-east-1${NC}"
            echo ""
            echo -e "${BLUE}2. 下载 Vivado 日志${NC}"
            echo -e "   ${CYAN}aws s3 cp $S3_LOGS/afi-*/\*_vivado.log vivado.log --region us-east-1${NC}"
            echo ""
            echo -e "${BLUE}3. 查看错误${NC}"
            echo -e "   ${CYAN}grep -i error vivado.log${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}常见问题:${NC}"
        echo "  • Vivado 版本不匹配 → 使用 Vivado 2024.1 重新构建"
        echo "  • 时序违例 → 优化设计或降低时钟频率"
        echo "  • 资源超限 → 减少设计规模"
        ;;
        
    *)
        echo -e "  状态: ${YELLOW}$STATUS${NC}"
        if [ "$MESSAGE" != "None" ] && [ -n "$MESSAGE" ]; then
            echo -e "  消息: $MESSAGE"
        fi
        ;;
esac

echo ""
echo -e "${BLUE}AWS 时间戳:${NC}"
echo -e "  创建: $AWS_CREATE_TIME"
echo -e "  更新: $AWS_UPDATE_TIME"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# 返回状态码
case "$STATUS" in
    "available") exit 0 ;;
    "pending") exit 2 ;;
    "failed") exit 1 ;;
    *) exit 3 ;;
esac
