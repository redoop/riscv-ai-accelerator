#!/bin/bash

# TPU测试脚本 - 使用iverilog
echo "🔧 运行TPU MAC单元测试 (使用iverilog)"

# 编译RTL和测试文件
echo "📝 编译RTL代码..."
iverilog -g2012 -I../../rtl -I../../rtl/accelerators \
    -o tpu_mac_test \
    ../../rtl/accelerators/tpu_mac_unit.sv \
    test_tpu_mac_simple.sv

if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
    
    # 运行仿真
    echo "🚀 运行仿真..."
    ./tpu_mac_test
    
    if [ $? -eq 0 ]; then
        echo "✅ TPU MAC测试完成"
        
        # 检查是否生成了波形文件
        if [ -f "tpu_mac_test.vcd" ]; then
            echo "📊 波形文件已生成: tpu_mac_test.vcd"
            echo "   可以使用 gtkwave tpu_mac_test.vcd 查看波形"
        fi
    else
        echo "❌ 仿真运行失败"
        exit 1
    fi
else
    echo "❌ 编译失败"
    exit 1
fi

echo "🎉 TPU测试完成!"