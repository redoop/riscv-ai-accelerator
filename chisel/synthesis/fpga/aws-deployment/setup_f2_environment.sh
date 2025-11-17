#!/bin/bash
# 在 AWS F2 实例上设置 FPGA 开发环境

set -e

echo "=== 设置 AWS F2 FPGA 开发环境 ==="
echo ""

# 更新系统
echo "[1/5] 更新系统包..."
sudo apt-get update -qq

# 安装 Java
echo "[2/5] 安装 Java..."
if ! command -v java &> /dev/null; then
    sudo apt-get install -y openjdk-11-jdk
    echo "✓ Java 已安装"
else
    echo "✓ Java 已存在: $(java -version 2>&1 | head -n 1)"
fi

# 安装 sbt
echo "[3/5] 安装 sbt..."
if ! command -v sbt &> /dev/null; then
    echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
    sudo apt-get update -qq
    sudo apt-get install -y sbt
    echo "✓ sbt 已安装"
else
    echo "✓ sbt 已存在: $(sbt --version 2>&1 | grep 'sbt version')"
fi

# 验证 Vivado
echo "[4/5] 验证 Vivado..."
if command -v vivado &> /dev/null; then
    echo "✓ Vivado 已安装: $(vivado -version 2>&1 | head -n 1)"
else
    echo "⚠ Vivado 未找到，请检查安装"
fi

# 克隆或更新项目
echo "[5/5] 准备项目..."
if [ ! -d "~/riscv-ai-accelerator" ]; then
    echo "克隆项目..."
    cd ~
    git clone https://github.com/redoop/riscv-ai-accelerator.git
    cd riscv-ai-accelerator
else
    echo "✓ 项目已存在"
    cd ~/riscv-ai-accelerator
    git pull
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         环境设置完成！                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "下一步："
echo "  cd ~/riscv-ai-accelerator/chisel/synthesis/fpga"
echo "  ./run_fpga_flow.sh prepare    # 生成 Verilog"
echo "  ./run_fpga_flow.sh aws        # 运行完整 AWS 流程"
echo ""
