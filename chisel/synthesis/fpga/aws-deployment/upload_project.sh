#!/bin/bash
# 上传 FPGA 项目到 F2 实例

set -e

# 加载实例信息
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFO_FILE="$SCRIPT_DIR/.f2_instance_info"

if [ ! -f "$INFO_FILE" ]; then
    echo "❌ 错误: 未找到实例信息文件"
    echo "请先运行: ./launch_f2_vivado.sh"
    exit 1
fi

source "$INFO_FILE"

# 实例信息
INSTANCE_IP="$PUBLIC_IP"
KEY_FILE="~/.ssh/${KEY_NAME}.pem"
USER="ubuntu"
REMOTE_DIR="~/fpga-project"

echo "=== 上传 FPGA 项目到 F2 实例 ==="
echo ""
echo "实例 IP: $INSTANCE_IP"
echo "用户: $USER"
echo ""

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FPGA_DIR="$SCRIPT_DIR/.."
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# 创建临时打包目录
TEMP_DIR=$(mktemp -d)
PROJECT_DIR="$TEMP_DIR/fpga-project"
mkdir -p "$PROJECT_DIR"

echo "📦 准备项目文件..."

# 复制生成的 Verilog（必需）
echo "  - 复制生成的 Verilog..."
mkdir -p "$PROJECT_DIR/generated"
if [ -d "$REPO_ROOT/chisel/generated/simple_edgeaisoc" ]; then
    cp -r "$REPO_ROOT/chisel/generated/simple_edgeaisoc" "$PROJECT_DIR/generated/"
    echo "    ✓ Verilog 文件已复制"
else
    echo "    ⚠️  警告: 未找到生成的 Verilog"
    echo "    请先运行: cd $REPO_ROOT/chisel && sbt 'runMain riscv.ai.SimpleEdgeAiSoCMain'"
fi

# 复制约束文件
echo "  - 复制约束文件..."
if [ -d "$FPGA_DIR/constraints" ]; then
    cp -r "$FPGA_DIR/constraints" "$PROJECT_DIR/"
    echo "    ✓ 约束文件已复制"
else
    echo "    ⚠️  警告: 未找到约束文件目录"
fi

# 复制脚本
echo "  - 复制脚本..."
if [ -d "$FPGA_DIR/scripts" ]; then
    cp -r "$FPGA_DIR/scripts" "$PROJECT_DIR/"
    echo "    ✓ 脚本已复制"
else
    echo "    ⚠️  警告: 未找到脚本目录"
fi

# 复制源码（如果存在）
if [ -d "$FPGA_DIR/src" ]; then
    echo "  - 复制源码..."
    cp -r "$FPGA_DIR/src" "$PROJECT_DIR/"
    echo "    ✓ 源码已复制"
fi

# 复制测试文件（如果存在）
if [ -d "$FPGA_DIR/testbench" ]; then
    echo "  - 复制测试文件..."
    cp -r "$FPGA_DIR/testbench" "$PROJECT_DIR/"
    echo "    ✓ 测试文件已复制"
fi

# 复制文档（如果存在）
echo "  - 复制文档..."
mkdir -p "$PROJECT_DIR/docs"
[ -f "$FPGA_DIR/README.md" ] && cp "$FPGA_DIR/README.md" "$PROJECT_DIR/"
[ -f "$FPGA_DIR/docs/BUILD_GUIDE.md" ] && cp "$FPGA_DIR/docs/BUILD_GUIDE.md" "$PROJECT_DIR/docs/" 2>/dev/null

# 复制环境脚本
if [ -f "$SCRIPT_DIR/setup_vivado_env.sh" ]; then
    echo "  - 复制环境脚本..."
    cp "$SCRIPT_DIR/setup_vivado_env.sh" "$PROJECT_DIR/"
    echo "    ✓ 环境脚本已复制"
fi

# 创建项目结构说明
cat > "$PROJECT_DIR/README.txt" << 'EOF'
FPGA 项目结构
=============

src/                    - FPGA 顶层和适配器
  fpga_top.v           - FPGA 顶层模块
  clock_gen.v          - 时钟生成
  io_adapter.v         - IO 适配器

generated/              - Chisel 生成的 Verilog
  simple_edgeaisoc/
    SimpleEdgeAiSoC.sv - SoC 核心设计

constraints/            - 约束文件
  timing.xdc           - 时序约束
  pins.xdc             - 引脚约束
  physical.xdc         - 物理约束

scripts/                - 构建和测试脚本
  build_fpga.tcl       - Vivado 构建脚本
  run_tests.sh         - 测试脚本

testbench/              - 测试平台
  tb_fpga_top.sv       - 顶层测试
  test_vectors/        - 测试向量

快速开始
========

1. 设置 Vivado 环境:
   source setup_vivado_env.sh

2. 运行 Vivado 构建:
   cd scripts
   vivado -mode batch -source build_fpga.tcl

3. 查看构建结果:
   ls -lh ../build/

EOF

# 打包
echo ""
echo "📦 打包项目..."
cd "$TEMP_DIR"
tar czf fpga-project.tar.gz fpga-project/
PROJECT_SIZE=$(du -h fpga-project.tar.gz | cut -f1)
echo "  ✓ 打包完成: $PROJECT_SIZE"

# 上传
echo ""
echo "📤 上传到 F2 实例..."
scp -i $KEY_FILE fpga-project.tar.gz ${USER}@${INSTANCE_IP}:~/

# 解压
echo ""
echo "📂 解压项目..."
ssh -i $KEY_FILE ${USER}@${INSTANCE_IP} << 'ENDSSH'
echo "解压 fpga-project.tar.gz..."
tar xzf fpga-project.tar.gz
echo "✓ 解压完成"
echo ""
echo "项目结构:"
ls -lh fpga-project/
echo ""
echo "查看 README:"
cat fpga-project/README.txt
ENDSSH

# 清理
rm -rf "$TEMP_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              项目上传成功！                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "连接到实例:"
echo "  ssh -i $KEY_FILE ${USER}@${INSTANCE_IP}"
echo ""
echo "开始构建:"
echo "  cd fpga-project"
echo "  source setup_vivado_env.sh"
echo "  cd scripts"
echo "  vivado -mode batch -source build_fpga.tcl"
echo ""
