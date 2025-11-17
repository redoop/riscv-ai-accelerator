#!/bin/bash
# 快速检查构建状态

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

echo "🔍 快速状态检查"
echo "================"
echo ""

# 检查 Vivado 进程
echo "📊 Vivado 进程:"
ssh -i $KEY ${USER}@${INSTANCE_IP} 'ps aux | grep "[v]ivado" | wc -l' | \
  xargs -I {} echo "  运行中的进程数: {}"

# 检查最新日志
echo ""
echo "📝 最新日志 (最后 10 行):"
ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -10 fpga-project/build/logs/vivado_build.log 2>/dev/null || echo "  日志文件为空"'

echo ""
echo "💡 提示:"
echo "  完整日志: ssh -i $KEY ${USER}@${INSTANCE_IP} 'tail -f fpga-project/build/logs/vivado_build.log'"
echo "  详细监控: ./monitor_build.sh"
