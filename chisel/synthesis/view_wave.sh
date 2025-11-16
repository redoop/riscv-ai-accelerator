#!/bin/bash
# 波形查看便捷脚本 - 从 synthesis 目录调用 waves 工具

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAVES_DIR="$SCRIPT_DIR/waves"

# 检查 waves 目录是否存在
if [ ! -d "$WAVES_DIR" ]; then
    echo "❌ 错误: waves 目录不存在: $WAVES_DIR"
    exit 1
fi

# 调用 waves 目录中的 view_wave.sh
exec "$WAVES_DIR/view_wave.sh" "$@"
