#!/bin/bash
# 启动 Web 波形查看器

echo "============================================================"
echo "启动 Web 波形查看器"
echo "============================================================"

# 检查 Flask 是否安装
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask 未安装，正在安装..."
    python3 -m pip install flask --user
fi

# 启动服务器
python3 wave_viewer.py --port 5000 --host 0.0.0.0

echo ""
echo "服务器已停止"
