#!/bin/bash

# 修复 RiscvAiChip.sv 的综合问题
# 1. 将 PicoRV32BlackBox 替换为 picorv32
# 2. 移除资源清单行

echo "🔧 修复 RiscvAiChip.sv 综合问题..."

INPUT_FILE="generated/RiscvAiChip.sv"
OUTPUT_FILE="generated/RiscvAiChip_fixed.sv"
BACKUP_FILE="generated/RiscvAiChip_backup.sv"

# 备份原文件
cp "$INPUT_FILE" "$BACKUP_FILE"
echo "✓ 已备份原文件到 $BACKUP_FILE"

# 修复步骤
echo "📝 应用修复..."

# 1. 替换 PicoRV32BlackBox 为 picorv32
# 2. 移除资源清单部分
sed -e 's/PicoRV32BlackBox/picorv32/g' \
    -e '/^\/\/ ----- 8< ----- FILE "firrtl_black_box_resource_files.f"/,$ d' \
    "$INPUT_FILE" > "$OUTPUT_FILE"

# 检查修复结果
if [ -f "$OUTPUT_FILE" ]; then
    ORIG_LINES=$(wc -l < "$INPUT_FILE")
    FIXED_LINES=$(wc -l < "$OUTPUT_FILE")
    
    echo "✓ 修复完成"
    echo "  原文件行数: $ORIG_LINES"
    echo "  修复后行数: $FIXED_LINES"
    echo "  输出文件: $OUTPUT_FILE"
    
    # 验证关键修改
    echo ""
    echo "🔍 验证修复..."
    
    # 检查是否还有 PicoRV32BlackBox
    if grep -q "PicoRV32BlackBox" "$OUTPUT_FILE"; then
        echo "⚠️  警告: 仍然存在 PicoRV32BlackBox"
    else
        echo "✓ PicoRV32BlackBox 已全部替换为 picorv32"
    fi
    
    # 检查是否还有资源清单
    if grep -q "firrtl_black_box_resource_files" "$OUTPUT_FILE"; then
        echo "⚠️  警告: 仍然存在资源清单标记"
    else
        echo "✓ 资源清单标记已移除"
    fi
    
    # 检查 picorv32 模块是否存在
    if grep -q "^module picorv32" "$OUTPUT_FILE"; then
        echo "✓ picorv32 模块定义存在"
    else
        echo "❌ 错误: picorv32 模块定义缺失"
    fi
    
    echo ""
    echo "✅ 修复完成！"
    echo ""
    echo "📁 文件位置:"
    echo "  原文件: $INPUT_FILE"
    echo "  备份: $BACKUP_FILE"
    echo "  修复后: $OUTPUT_FILE"
    echo ""
    echo "💡 使用修复后的文件进行综合:"
    echo "  yosys -p \"read_verilog $OUTPUT_FILE; ...\""
    
else
    echo "❌ 修复失败"
    exit 1
fi
