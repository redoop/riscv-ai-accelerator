#!/bin/bash
# 启动 HTTP 服务器查看波形文件

PORT=8000

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口 $PORT 已被占用，尝试使用端口 $((PORT+1))"
    PORT=$((PORT+1))
fi

echo "启动 HTTP 服务器..."
echo ""

# 启动服务器
python3 serve_wave.py -p $PORT

# 如果失败，尝试其他端口
if [ $? -ne 0 ]; then
    echo ""
    echo "尝试使用端口 8080..."
    python3 serve_wave.py -p 8080
fi
