#!/bin/bash
# 快速查看波形 - 生成静态 HTML 页面

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
VCD_FILE="post_syn.vcd"
MAX_SIGNALS=15
MAX_POINTS=2000
OUTPUT_FILE=""

# 显示帮助
show_help() {
    cat << EOF
用法: $0 [选项]

快速生成静态波形 HTML 页面

选项:
    -f, --file FILE         VCD 文件路径 (默认: post_syn.vcd)
    -o, --output FILE       输出 HTML 文件路径 (默认: 自动生成)
    -s, --signals NUM       最大信号数量 (默认: 15)
    -p, --points NUM        最大采样点数 (默认: 2000)
    -h, --help              显示此帮助信息

示例:
    $0                                      # 使用默认参数
    $0 -f my.vcd                            # 指定 VCD 文件
    $0 -s 20 -p 3000                        # 更多信号和采样点
    $0 -f my.vcd -o my_wave.html            # 指定输出文件

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            VCD_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--signals)
            MAX_SIGNALS="$2"
            shift 2
            ;;
        -p|--points)
            MAX_POINTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查 VCD 文件
if [ ! -f "$VCD_FILE" ]; then
    echo "❌ 错误: VCD 文件不存在: $VCD_FILE"
    exit 1
fi

# 检查依赖
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo "正在安装依赖..."
    python3 -m pip install matplotlib --user
fi

# 构建命令
CMD="python3 \"$SCRIPT_DIR/generate_static_wave.py\" \"$VCD_FILE\" --max-signals $MAX_SIGNALS --max-points $MAX_POINTS"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD -o \"$OUTPUT_FILE\""
fi

# 执行生成
echo "正在生成静态波形页面..."
echo ""
eval $CMD

# 获取输出文件名
if [ -z "$OUTPUT_FILE" ]; then
    BASENAME=$(basename "$VCD_FILE" .vcd)
    OUTPUT_FILE="waveform_${BASENAME}.html"
fi

# 提示打开
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✓ 生成完成！"
    echo ""
    echo "查看波形:"
    echo "  方法 1: 在浏览器中打开 $OUTPUT_FILE"
    echo "  方法 2: 运行命令: xdg-open $OUTPUT_FILE"
    echo ""
    
    # 尝试自动打开（如果有桌面环境）
    if command -v xdg-open &> /dev/null; then
        read -p "是否现在打开? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            xdg-open "$OUTPUT_FILE" &
        fi
    fi
fi
