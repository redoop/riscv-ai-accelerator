#!/bin/bash
# 快速检查当前构建状态（单次检查）

INSTANCE_IP="54.81.161.62"
KEY="~/.ssh/fpga-f2-key.pem"
USER="ubuntu"

echo "🔍 FPGA 构建状态检查"
echo "===================="
echo ""

# 检查进程
PROC_COUNT=$(ssh -i $KEY ${USER}@${INSTANCE_IP} 'ps aux | grep "[v]ivado" | wc -l' 2>/dev/null)
echo "📊 Vivado 进程: $PROC_COUNT"

if [ "$PROC_COUNT" -eq 0 ]; then
    echo "⚠️  没有 Vivado 进程运行"
    echo ""
    echo "检查最后的日志..."
    ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -30 fpga-project/build/logs/vivado_build.log 2>/dev/null | grep -E "ERROR|完成|finished|Complete"'
    exit 0
fi

echo ""
echo "📝 当前阶段:"
ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -50 fpga-project/build/logs/vivado_build.log 2>/dev/null' | \
    grep -E "Starting|Finished|Phase|开始|完成" | tail -5

echo ""
echo "⏱️  最新进度:"
ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -20 fpga-project/build/logs/vivado_build.log 2>/dev/null' | \
    grep -E "Time \(s\):|INFO:" | tail -3

echo ""
echo "💡 启动持续监控: ./continuous_monitor.sh"
