#!/bin/bash
# 验证文件列表中的所有文件是否存在

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FILELIST="$1"
if [ -z "$FILELIST" ]; then
    echo "用法: $0 <filelist>"
    echo "示例: $0 filelist_ysyxsoc.f"
    exit 1
fi

if [ ! -f "$FILELIST" ]; then
    echo "错误: 文件列表不存在: $FILELIST"
    exit 1
fi

echo "=========================================="
echo "验证文件列表: $FILELIST"
echo "=========================================="
echo ""

TOTAL=0
FOUND=0
MISSING=0

while IFS= read -r line; do
    # 跳过注释和空行
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    
    TOTAL=$((TOTAL + 1))
    
    if [ -f "$line" ]; then
        echo "✓ $line"
        FOUND=$((FOUND + 1))
    else
        echo "✗ $line (缺失)"
        MISSING=$((MISSING + 1))
    fi
done < "$FILELIST"

echo ""
echo "=========================================="
echo "验证结果"
echo "=========================================="
echo "总文件数: $TOTAL"
echo "找到: $FOUND"
echo "缺失: $MISSING"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✓ 所有文件都存在！"
    exit 0
else
    echo "✗ 有 $MISSING 个文件缺失"
    echo ""
    echo "提示:"
    echo "  - 如果缺少 SimpleEdgeAiSoC.sv，运行: cd ../chisel && make"
    echo "  - 检查文件路径是否正确"
    exit 1
fi
