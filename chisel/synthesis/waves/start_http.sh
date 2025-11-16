#!/bin/bash
# 启动 HTTP 服务器查看波形文件

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT=8000

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口 $PORT 已被占用，尝试使用端口 $((PORT+1))"
    PORT=$((PORT+1))
fi

echo "启动 HTTP 服务器..."
echo ""

# 切换到脚本目录并启动服务器
cd "$SCRIPT_DIR"
python3 serve_wave.py -p $PORT -d .

# 如果失败，尝试其他端口
if [ $? -ne 0 ]; then
    echo ""
    echo "尝试使用端口 8080..."
    python3 serve_wave.py -p 8080 -d .
fi
