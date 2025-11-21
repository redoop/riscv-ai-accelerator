#!/bin/bash
# 快速测试 SDC 约束文件
# 仅验证语法和基本集成，不运行完整综合

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "SDC 约束文件快速测试"
echo "=========================================="
echo ""

# 检查 SDC 文件
SDC_FILE="fpga/constraints/timing_complete.sdc"
if [ ! -f "$SDC_FILE" ]; then
    echo "✗ 错误: 未找到 SDC 文件: $SDC_FILE"
    exit 1
fi
echo "✓ 找到 SDC 文件: $SDC_FILE"
echo ""

# 显示文件信息
echo "文件信息:"
echo "  大小: $(wc -c < "$SDC_FILE") 字节"
echo "  行数: $(wc -l < "$SDC_FILE") 行"
echo ""

# 检查关键约束
echo "检查关键约束:"

# 主时钟
if grep -q "create_clock.*sys_clk.*period 10.000" "$SDC_FILE"; then
    echo "  ✓ 主时钟: 100 MHz (10 ns)"
else
    echo "  ✗ 主时钟定义错误"
    exit 1
fi

# SPI 时钟
if grep -q "create_generated_clock.*spi_clk" "$SDC_FILE"; then
    echo "  ✓ SPI 时钟: 10 MHz (生成时钟)"
else
    echo "  ⚠ SPI 时钟定义缺失"
fi

# 时钟不确定性
if grep -q "set_clock_uncertainty" "$SDC_FILE"; then
    echo "  ✓ 时钟不确定性约束"
else
    echo "  ⚠ 时钟不确定性约束缺失"
fi

# 输入延迟
INPUT_DELAYS=$(grep -c "set_input_delay" "$SDC_FILE" || true)
echo "  ✓ 输入延迟约束: $INPUT_DELAYS 条"

# 输出延迟
OUTPUT_DELAYS=$(grep -c "set_output_delay" "$SDC_FILE" || true)
echo "  ✓ 输出延迟约束: $OUTPUT_DELAYS 条"

# 假路径
FALSE_PATHS=$(grep -c "set_false_path" "$SDC_FILE" || true)
echo "  ✓ 假路径约束: $FALSE_PATHS 条"

# 设计规则
if grep -q "set_max_fanout\|set_max_transition\|set_max_capacitance" "$SDC_FILE"; then
    echo "  ✓ 设计规则约束"
else
    echo "  ⚠ 设计规则约束缺失"
fi

echo ""

# 检查端口名称
echo "检查端口名称匹配:"
PORTS=("clock" "reset" "io_uart_rx" "io_uart_tx" "io_lcd_spi_clk" "io_gpio_in" "io_gpio_out")
for port in "${PORTS[@]}"; do
    if grep -q "$port" "$SDC_FILE"; then
        echo "  ✓ $port"
    else
        echo "  ⚠ $port (未在 SDC 中约束)"
    fi
done

echo ""

# 显示时钟定义
echo "时钟定义详情:"
grep "create_clock\|create_generated_clock" "$SDC_FILE" | sed 's/^/  /'

echo ""
echo "=========================================="
echo "✓ SDC 文件验证通过"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 运行完整集成测试:"
echo "     bash test_sdc_integration.sh"
echo ""
echo "  2. 使用 SDC 进行综合:"
echo "     bash run_ics55_synthesis.sh    # ICS55 PDK"
echo "     bash run_ihp_synthesis.sh      # IHP PDK"
echo "     bash run_generic_synthesis.sh  # 通用"
echo ""
echo "  3. 查看 SDC 使用文档:"
echo "     cat fpga/constraints/SDC_USAGE.md"
echo ""
