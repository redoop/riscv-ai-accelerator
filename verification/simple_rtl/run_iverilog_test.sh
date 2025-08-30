#!/bin/bash

echo "🔨 编译简单 MAC RTL 测试 (使用 Icarus Verilog)..."

# 编译 SystemVerilog 代码
iverilog -g2012 -o simple_mac_test simple_mac_test.sv

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "🚀 运行 RTL 仿真..."
    
    # 运行仿真
    vvp simple_mac_test
    
    if [ $? -eq 0 ]; then
        echo "🎉 RTL 仿真完成!"
        
        # 检查是否生成了波形文件
        if [ -f "simple_mac_test.vcd" ]; then
            echo "📊 波形文件已生成: simple_mac_test.vcd"
            echo "💡 使用 GTKWave 查看波形: gtkwave simple_mac_test.vcd"
        fi
    else
        echo "❌ RTL 仿真失败!"
        exit 1
    fi
else
    echo "❌ 编译失败!"
    exit 1
fi