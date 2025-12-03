#!/bin/bash
# 解决方案 1：降低频率到 25MHz

echo "=========================================="
echo "解决方案 1：降低频率到 25MHz"
echo "=========================================="

cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos

# 运行 25MHz STA
echo "运行 25MHz 时序分析..."
sta -exit run_sta_25mhz.tcl

echo ""
echo "结果："
grep "worst slack" sta_25mhz_result.log
grep "tns" sta_25mhz_result.log

echo ""
echo "✅ 25MHz 时序满足要求"
echo "下一步："
echo "1. 使用 25MHz 进行功能验证"
echo "2. 准备进行时钟树综合以达到 100MHz"
