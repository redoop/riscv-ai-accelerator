#!/bin/bash
# 监控 F2 实例上的 Vivado 构建进度

INSTANCE_IP="54.81.161.62"
KEY_FILE="~/.ssh/fpga-f2-key.pem"
USER="ubuntu"

echo "=== 监控 FPGA 构建进度 ==="
echo ""

# 检查 Vivado 进程
echo "📊 Vivado 进程状态："
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} 'ps aux | grep "[v]ivado" | head -3'

echo ""
echo "📁 构建目录："
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} 'ls -lh fpga-project/build/ 2>/dev/null || echo "构建目录尚未创建"'

echo ""
echo "📝 最新日志（最后 30 行）："
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} 'tail -30 fpga-project/build/logs/vivado_build.log 2>/dev/null || echo "日志文件为空或尚未创建"'

echo ""
echo "💾 磁盘使用："
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} 'df -h | grep -E "Filesystem|/$"'

echo ""
echo "🔄 持续监控："
echo "  watch -n 30 ./monitor_build.sh"
echo ""
echo "📊 实时日志："
echo "  ssh -i $KEY_FILE ${USER}@${INSTANCE_IP}"
echo "  tail -f fpga-project/build/logs/vivado_build.log"
echo ""
