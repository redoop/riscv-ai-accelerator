#!/bin/bash

echo "🔬 编译真实 TPU MAC RTL 测试..."

# 编译包含真实 RTL 模块的测试
iverilog -g2012 -I../../rtl/config -I../../rtl/accelerators \
    -o test_real_tpu_mac \
    ../../rtl/accelerators/tpu_mac_unit.sv \
    test_real_tpu_mac.sv

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "🚀 运行真实 TPU MAC RTL 仿真..."
    
    # 运行仿真
    vvp test_real_tpu_mac
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 真实 RTL 硬件代码执行完成!"
        echo "📊 波形文件: test_real_tpu_mac.vcd"
        echo "💡 查看波形: gtkwave test_real_tpu_mac.vcd"
        echo ""
        echo "🔍 总结:"
        echo "  ✅ 成功调用了项目中的真实 RTL 代码"
        echo "  ✅ TPU MAC 单元正常工作"
        echo "  ✅ 硬件乘加运算验证通过"
    else
        echo "❌ RTL 仿真失败!"
        exit 1
    fi
else
    echo "❌ 编译失败!"
    exit 1
fi