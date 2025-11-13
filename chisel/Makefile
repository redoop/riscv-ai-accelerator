# Chisel项目Makefile
# 设置Java环境
JAVA_HOME := /Library/Java/JavaVirtualMachines/jdk-11.0.16.jdk/Contents/Home
export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)

# SBT命令前缀
SBT := sbt

# 默认目标
.PHONY: all test clean compile help

all: compile test

# 编译项目
compile:
	@echo "编译Chisel项目..."
	$(SBT) compile

# 编译测试代码
compile-test:
	@echo "编译测试代码..."
	$(SBT) Test/compile

# 运行所有测试
test:
	@echo "运行所有测试..."
	$(SBT) test

# 运行特定的矩阵乘法测试
test-matrix:
	@echo "运行MatrixMultiplier测试..."
	$(SBT) 'testOnly MatrixMultiplierTest'

# 运行4x4矩阵乘法的10个随机测试用例
test-4x4-random:
	@echo "运行4x4矩阵乘法的10个随机测试用例..."
	$(SBT) 'testOnly MatrixMultiplierTest -- -z "perform 4x4 matrix multiplication with 10 random test cases"'

# 运行基本2x2矩阵乘法测试
test-2x2-basic:
	@echo "运行基本2x2矩阵乘法测试..."
	$(SBT) 'testOnly MatrixMultiplierTest -- -z "perform basic 2x2 matrix multiplication"'

# 运行大规模矩阵测试
test-large-matrices:
	@echo "运行大规模矩阵测试..."
	$(SBT) 'testOnly MatrixMultiplierTest -- -z "support large matrix multiplication"'

# 运行测试并显示详细输出
test-verbose:
	@echo "运行测试（详细输出）..."
	$(SBT) test -v

# 运行测试并只显示警告
test-warn:
	@echo "运行测试（仅警告）..."
	$(SBT) test --warn

# 清理构建文件
clean:
	@echo "清理构建文件..."
	$(SBT) clean
	@echo "删除测试运行目录..."
	rm -rf test_run_dir

# 生成Verilog代码
verilog:
	@echo "生成Verilog代码..."
	$(SBT) 'runMain riscv.ai.Main'

# 显示帮助信息
help:
	@echo "可用的make目标："
	@echo "  all              - 编译并运行所有测试"
	@echo "  compile          - 编译项目"
	@echo "  compile-test     - 编译测试代码"
	@echo "  test             - 运行所有测试"
	@echo "  test-matrix      - 运行MatrixMultiplier测试"
	@echo "  test-4x4-random  - 运行4x4矩阵乘法的10个随机测试用例"
	@echo "  test-2x2-basic   - 运行基本2x2矩阵乘法测试"
	@echo "  test-large-matrices - 运行大规模矩阵测试"
	@echo "  test-verbose     - 运行测试（详细输出）"
	@echo "  test-warn        - 运行测试（仅警告）"
	@echo "  clean            - 清理构建文件"
	@echo "  verilog          - 生成Verilog代码"
	@echo "  help             - 显示此帮助信息"
	@echo ""
	@echo "环境变量："
	@echo "  JAVA_HOME=$(JAVA_HOME)"