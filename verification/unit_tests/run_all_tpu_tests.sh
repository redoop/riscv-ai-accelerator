#!/bin/bash

# 运行所有TPU测试的脚本
echo "🔧 运行完整的TPU测试套件"
echo "=========================="

# 测试计数器
total_tests=0
passed_tests=0
failed_tests=0

# 测试结果数组
declare -a test_results

# 运行单个测试的函数
run_test() {
    local test_name=$1
    local test_file=$2
    local rtl_files=$3
    
    echo ""
    echo "🧪 运行测试: $test_name"
    echo "----------------------------"
    
    total_tests=$((total_tests + 1))
    
    # 编译测试
    echo "📝 编译 $test_file..."
    iverilog -g2012 -I../../rtl -I../../rtl/accelerators \
        -o ${test_name}_sim \
        $rtl_files \
        $test_file
    
    if [ $? -eq 0 ]; then
        echo "✅ 编译成功"
        
        # 运行仿真
        echo "🚀 运行仿真..."
        ./${test_name}_sim
        
        if [ $? -eq 0 ]; then
            echo "✅ $test_name 测试通过"
            passed_tests=$((passed_tests + 1))
            test_results+=("✅ $test_name: PASSED")
        else
            echo "❌ $test_name 测试失败"
            failed_tests=$((failed_tests + 1))
            test_results+=("❌ $test_name: FAILED")
        fi
    else
        echo "❌ $test_name 编译失败"
        failed_tests=$((failed_tests + 1))
        test_results+=("❌ $test_name: COMPILE_FAILED")
    fi
}

# 清理之前的编译文件
echo "🧹 清理之前的编译文件..."
rm -f *_sim *.vcd

# 测试1: TPU MAC简单测试
run_test "tpu_mac_simple" \
    "test_tpu_mac_simple.sv" \
    "../../rtl/accelerators/tpu_mac_unit.sv"

# 测试2: TPU MAC数组测试 (修复版)
if [ -f "test_tpu_mac_array_fixed.sv" ]; then
    run_test "tpu_mac_array_fixed" \
        "test_tpu_mac_array_fixed.sv" \
        "../../rtl/accelerators/tpu_mac_unit.sv"
fi

# 测试3: TPU计算数组测试 (修复版)
if [ -f "test_tpu_compute_array_fixed.sv" ]; then
    run_test "tpu_compute_array_fixed" \
        "test_tpu_compute_array_fixed.sv" \
        ""
fi

# 测试4: TPU控制器和缓存测试 (修复版)
if [ -f "test_tpu_controller_cache_fixed.sv" ]; then
    run_test "tpu_controller_cache_fixed" \
        "test_tpu_controller_cache_fixed.sv" \
        ""
fi

# 显示测试总结
echo ""
echo "🎯 TPU测试套件总结"
echo "=================="
echo "总测试数: $total_tests"
echo "通过测试: $passed_tests"
echo "失败测试: $failed_tests"
echo ""

echo "📊 详细结果:"
for result in "${test_results[@]}"; do
    echo "  $result"
done

echo ""
if [ $failed_tests -eq 0 ]; then
    echo "🎉 所有TPU测试都通过了!"
    echo "✨ TPU硬件验证成功!"
else
    echo "⚠️  有 $failed_tests 个测试失败"
    echo "🔧 请检查失败的测试并修复问题"
fi

# 清理编译文件
echo ""
echo "🧹 清理编译文件..."
rm -f *_sim

echo "🏁 TPU测试套件完成!"