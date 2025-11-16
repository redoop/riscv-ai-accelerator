#!/bin/bash
# 在 F2 实例上设置 Vivado 环境

echo "=== 设置 Vivado 2025.1 环境 ==="

# Vivado 路径
export VIVADO_PATH="/tools/Xilinx/2025.1/Vivado"
export PATH="$VIVADO_PATH/bin:$PATH"

# 设置环境脚本
if [ -f "$VIVADO_PATH/settings64.sh" ]; then
    source "$VIVADO_PATH/settings64.sh"
    echo "✓ Vivado 环境已加载"
else
    echo "⚠ 未找到 settings64.sh，仅添加到 PATH"
fi

# 验证
echo ""
echo "Vivado 版本:"
vivado -version | head -3

echo ""
echo "✓ 环境设置完成！"
echo ""
echo "使用方法:"
echo "  source ~/setup_vivado_env.sh"
echo "  vivado -mode batch -source your_script.tcl"
