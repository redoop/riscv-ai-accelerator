#!/bin/bash
# 持续监控 FPGA 构建 - 智能版本
# 特性：
# - 跟踪最新变化
# - 检测错误并自动退出
# - 显示进度百分比
# - 记录关键里程碑

# 加载实例信息
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f2_instance_info"

if [ ! -f "$INFO_FILE" ]; then
    echo "❌ 错误: 未找到实例信息文件"
    echo "请先运行: ./launch_f2_vivado.sh"
    exit 1
fi

source "$INFO_FILE"

INSTANCE_IP="$PUBLIC_IP"
KEY="~/.ssh/${KEY_NAME}.pem"
USER="ubuntu"
LOG_FILE="monitor_$(date +%Y%m%d_%H%M%S).log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 记录开始时间
START_TIME=$(date +%s)

# 上次日志行数
LAST_LINE_COUNT=0

# 阶段跟踪
CURRENT_STAGE=""
STAGE_START_TIME=0

echo "🚀 启动持续监控 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo "========================================"
echo ""

# 函数：获取远程日志
get_remote_log() {
    ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -100 fpga-project/build/logs/vivado_build.log 2>/dev/null'
}

# 函数：检查错误
check_errors() {
    local log_content="$1"
    
    # 检查严重错误
    if echo "$log_content" | grep -q "ERROR:.*failed"; then
        return 1
    fi
    
    if echo "$log_content" | grep -q "CRITICAL WARNING.*failed"; then
        return 1
    fi
    
    if echo "$log_content" | grep -q "Failed runs"; then
        return 1
    fi
    
    return 0
}

# 函数：提取当前阶段
extract_stage() {
    local log_content="$1"
    
    if echo "$log_content" | tail -20 | grep -q "Starting synthesis"; then
        echo "Synthesis"
    elif echo "$log_content" | tail -20 | grep -q "Finished Synthesis"; then
        echo "Synthesis Complete"
    elif echo "$log_content" | tail -20 | grep -q "Starting Placer"; then
        echo "Placement"
    elif echo "$log_content" | tail -20 | grep -q "Finished Placer"; then
        echo "Placement Complete"
    elif echo "$log_content" | tail -20 | grep -q "Starting Routing"; then
        echo "Routing"
    elif echo "$log_content" | tail -20 | grep -q "Finished Routing"; then
        echo "Routing Complete"
    elif echo "$log_content" | tail -20 | grep -q "write_bitstream"; then
        echo "Bitstream Generation"
    elif echo "$log_content" | tail -20 | grep -q "Bitstream.*complete"; then
        echo "Build Complete"
    else
        echo "Unknown"
    fi
}

# 函数：计算进度百分比
calculate_progress() {
    local stage="$1"
    
    case "$stage" in
        "Synthesis") echo "15" ;;
        "Synthesis Complete") echo "30" ;;
        "Placement") echo "50" ;;
        "Placement Complete") echo "65" ;;
        "Routing") echo "80" ;;
        "Routing Complete") echo "90" ;;
        "Bitstream Generation") echo "95" ;;
        "Build Complete") echo "100" ;;
        *) echo "0" ;;
    esac
}

# 主监控循环
ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    # 清屏
    clear
    
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          FPGA 构建持续监控 - 迭代 #$ITERATION${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "⏱️  运行时间: $(($ELAPSED / 60)) 分 $(($ELAPSED % 60)) 秒"
    echo "🕐 当前时间: $(date '+%H:%M:%S')"
    echo ""
    
    # 检查进程
    PROC_COUNT=$(ssh -i $KEY ${USER}@${INSTANCE_IP} 'ps aux | grep "[v]ivado" | wc -l' 2>/dev/null)
    
    if [ "$PROC_COUNT" -eq 0 ]; then
        echo -e "${RED}⚠️  警告: 没有 Vivado 进程运行！${NC}"
        echo ""
        echo "检查最后的日志..."
        LOG_CONTENT=$(get_remote_log)
        
        if check_errors "$LOG_CONTENT"; then
            echo -e "${GREEN}✅ 构建可能已完成${NC}"
        else
            echo -e "${RED}❌ 构建失败${NC}"
            echo ""
            echo "最后 20 行日志:"
            echo "$LOG_CONTENT" | tail -20
        fi
        
        exit 0
    fi
    
    echo -e "${GREEN}📊 Vivado 进程: $PROC_COUNT 个运行中${NC}"
    echo ""
    
    # 获取日志
    LOG_CONTENT=$(get_remote_log)
    
    # 检查错误
    if ! check_errors "$LOG_CONTENT"; then
        echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║                  ❌ 检测到错误！                          ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "错误日志:"
        echo "$LOG_CONTENT" | grep -A 5 -B 5 "ERROR\|CRITICAL WARNING\|Failed"
        echo ""
        echo -e "${YELLOW}监控已停止。请检查完整日志。${NC}"
        exit 1
    fi
    
    # 提取当前阶段
    STAGE=$(extract_stage "$LOG_CONTENT")
    PROGRESS=$(calculate_progress "$STAGE")
    
    # 检测阶段变化
    if [ "$STAGE" != "$CURRENT_STAGE" ] && [ "$STAGE" != "Unknown" ]; then
        STAGE_ELAPSED=$((CURRENT_TIME - STAGE_START_TIME))
        if [ "$CURRENT_STAGE" != "" ]; then
            echo -e "${GREEN}✅ $CURRENT_STAGE 完成 (用时: $(($STAGE_ELAPSED / 60))分$(($STAGE_ELAPSED % 60))秒)${NC}" | tee -a "$LOG_FILE"
        fi
        CURRENT_STAGE="$STAGE"
        STAGE_START_TIME=$CURRENT_TIME
        echo -e "${BLUE}🔄 进入新阶段: $STAGE${NC}" | tee -a "$LOG_FILE"
    fi
    
    # 显示进度条
    echo "📈 构建进度: $PROGRESS%"
    BAR_LENGTH=50
    FILLED=$((PROGRESS * BAR_LENGTH / 100))
    EMPTY=$((BAR_LENGTH - FILLED))
    printf "["
    printf "%${FILLED}s" | tr ' ' '█'
    printf "%${EMPTY}s" | tr ' ' '░'
    printf "] $PROGRESS%%\n"
    echo ""
    
    # 显示当前阶段
    if [ "$STAGE" != "Unknown" ]; then
        STAGE_ELAPSED=$((CURRENT_TIME - STAGE_START_TIME))
        echo -e "${YELLOW}🔧 当前阶段: $STAGE${NC}"
        echo "   阶段用时: $(($STAGE_ELAPSED / 60))分$(($STAGE_ELAPSED % 60))秒"
    fi
    echo ""
    
    # 显示最新日志（只显示新增的行）
    LINE_COUNT=$(echo "$LOG_CONTENT" | wc -l)
    if [ $LINE_COUNT -gt $LAST_LINE_COUNT ]; then
        NEW_LINES=$((LINE_COUNT - LAST_LINE_COUNT))
        echo "📝 最新日志 (新增 $NEW_LINES 行):"
        echo "────────────────────────────────────────────────────────────"
        echo "$LOG_CONTENT" | tail -$NEW_LINES | grep -E "INFO:|Phase|Starting|Finished|完成|Time \(s\):" | tail -10
        LAST_LINE_COUNT=$LINE_COUNT
    else
        echo "📝 等待新日志..."
    fi
    
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "⏱️  下次更新: 30秒后 | 按 Ctrl+C 停止"
    
    # 如果构建完成，退出
    if [ "$STAGE" = "Build Complete" ]; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║              🎉 构建成功完成！                            ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "总用时: $(($ELAPSED / 60)) 分钟"
        exit 0
    fi
    
    sleep 30
done
