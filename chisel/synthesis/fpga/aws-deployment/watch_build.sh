#!/bin/bash
# 持续监控构建进度

INSTANCE_IP="54.81.161.62"
KEY="~/.ssh/fpga-f2-key.pem"
USER="ubuntu"

echo "🔄 持续监控 FPGA 构建"
echo "===================="
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "📊 构建状态 - $(date '+%H:%M:%S')"
    echo "================================"
    echo ""
    
    # 检查进程
    PROC_COUNT=$(ssh -i $KEY ${USER}@${INSTANCE_IP} 'ps aux | grep "[v]ivado" | wc -l')
    echo "Vivado 进程数: $PROC_COUNT"
    
    # 最新日志
    echo ""
    echo "📝 最新进度:"
    ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -15 fpga-project/build/logs/vivado_build.log 2>/dev/null | grep -E "INFO:|开始|完成|finished|Starting|Finished|ERROR|WARNING" | tail -10'
    
    echo ""
    echo "⏱️  下次更新: 30秒后..."
    sleep 30
done
