# RISC-V AI Accelerator Chisel Project Makefile
# 设置Java环境
JAVA_HOME := /Library/Java/JavaVirtualMachines/jdk-11.0.16.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)

# SBT命令前缀
SBT := sbt

# 默认目标
.PHONY: all test clean compile help generate

all: compile test

# 编译项目
compile:
	@echo "📦 编译Chisel项目..."
	$(SBT) compile

# 编译测试代码
compile-test:
	@echo "📦 编译测试代码..."
	$(SBT) Test/compile

# 运行所有测试
test:
	@echo "🧪 运行所有测试..."
	$(SBT) test

# 运行 SimpleEdgeAiSoC 测试
test-soc:
	@echo "🧪 运行 SimpleEdgeAiSoC 测试..."
	$(SBT) 'testOnly riscv.ai.SimpleEdgeAiSoCTest'

# 运行 PicoRV32 核心测试
test-picorv32:
	@echo "🧪 运行 PicoRV32 核心测试..."
	$(SBT) 'testOnly riscv.ai.PicoRV32CoreTest'

# 运行 BitNet 加速器测试
test-bitnet:
	@echo "🧪 运行 BitNet 加速器调试测试..."
	$(SBT) 'testOnly riscv.ai.BitNetAccelDebugTest'

# 运行 Compact 加速器测试
test-compact:
	@echo "🧪 运行 Compact 加速器调试测试..."
	$(SBT) 'testOnly riscv.ai.SimpleCompactAccelDebugTest'

# 运行 RISC-V 指令测试
test-riscv:
	@echo "🧪 运行 RISC-V 指令测试..."
	$(SBT) 'testOnly riscv.ai.SimpleRiscvInstructionTests'

# 运行测试并显示详细输出
test-verbose:
	@echo "🧪 运行测试（详细输出）..."
	$(SBT) 'set Test / testOptions += Tests.Argument("-oD")' test

# 清理构建文件
clean:
	@echo "🧹 清理构建文件..."
	$(SBT) clean
	@echo "🧹 删除测试运行目录..."
	rm -rf test_run_dir
	@echo "🧹 删除生成的文件..."
	rm -rf generated

# 生成 SimpleEdgeAiSoC Verilog
generate:
	@echo "🔧 生成 SimpleEdgeAiSoC SystemVerilog..."
	$(SBT) 'runMain riscv.ai.SimpleEdgeAiSoCMain'

# 生成所有版本的 Verilog
generate-all:
	@echo "🔧 生成所有版本的 SystemVerilog..."
	$(SBT) 'runMain riscv.ai.VerilogGenerator'

# 快速测试（仅运行 SoC 测试）
quick-test: compile test-soc

# 完整流程
full: compile test generate
	@echo "✅ 完整流程完成！"

# 显示帮助信息
help:
	@echo "RISC-V AI Accelerator - 可用的 make 目标："
	@echo ""
	@echo "📦 编译相关："
	@echo "  compile          - 编译项目"
	@echo "  compile-test     - 编译测试代码"
	@echo ""
	@echo "🧪 测试相关："
	@echo "  test             - 运行所有测试"
	@echo "  test-soc         - 运行 SimpleEdgeAiSoC 测试"
	@echo "  test-picorv32    - 运行 PicoRV32 核心测试"
	@echo "  test-bitnet      - 运行 BitNet 加速器测试"
	@echo "  test-compact     - 运行 Compact 加速器测试"
	@echo "  test-riscv       - 运行 RISC-V 指令测试"
	@echo "  test-verbose     - 运行测试（详细输出）"
	@echo "  quick-test       - 快速测试（仅 SoC）"
	@echo ""
	@echo "🔧 生成相关："
	@echo "  generate         - 生成 SimpleEdgeAiSoC SystemVerilog"
	@echo "  generate-all     - 生成所有版本的 SystemVerilog"
	@echo ""
	@echo "🧹 清理相关："
	@echo "  clean            - 清理所有构建文件"
	@echo ""
	@echo "🚀 工作流："
	@echo "  all              - 编译并运行所有测试"
	@echo "  full             - 完整流程（编译+测试+生成）"
	@echo "  help             - 显示此帮助信息"
	@echo ""
	@echo "💡 示例："
	@echo "  make quick-test  - 快速验证 SoC 功能"
	@echo "  make full        - 完整的开发流程"
	@echo "  make generate    - 仅生成 Verilog 文件"
	@echo ""
	@echo "🔧 环境变量："
	@echo "  JAVA_HOME=$(JAVA_HOME)"