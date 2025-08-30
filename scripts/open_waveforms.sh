#!/bin/bash
# 快速打开RTL波形文件

echo "🌊 RTL波形查看器"
echo "================"

cd verification/simple_rtl

# 检查VCD文件
if [ -f "test_simple_tpu_mac.vcd" ]; then
    echo "📊 找到主要波形文件: test_simple_tpu_mac.vcd"
    echo "🚀 使用GTKWave打开波形..."
    gtkwave test_simple_tpu_mac.vcd &
    echo "✅ GTKWave已启动"
else
    echo "❌ 未找到波形文件，先运行RTL仿真..."
    echo "🔧 运行: python3 ../../rtl_hardware_backend.py"
fi

# 列出所有VCD文件
echo ""
echo "📈 所有可用的波形文件:"
ls -la *.vcd 2>/dev/null || echo "  (无VCD文件)"