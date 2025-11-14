# 逻辑综合后网表仿真 Makefile

# 工具配置
VCS = vcs
VERDI = verdi
VERILATOR = verilator

# 目录配置
NETLIST_DIR = ./netlist
TB_DIR = ./testbench
SIM_DIR = ./sim
WAVE_DIR = ./waves

# 设计配置
DESIGN = SimpleEdgeAiSoC
NETLIST = $(NETLIST_DIR)/$(DESIGN)_syn.v
TB_BASIC = $(TB_DIR)/post_syn_tb.sv
TB_ADVANCED = $(TB_DIR)/advanced_post_syn_tb.sv

# VCS 编译选项
VCS_OPTS = -full64 \
           -sverilog \
           -timescale=1ns/1ps \
           +v2k \
           -debug_all \
           -kdb \
           -lca \
           -LDFLAGS -Wl,--no-as-needed

# 仿真选项
SIM_OPTS = +vcs+finish+100000000 \
           -ucli

# 默认目标
.PHONY: all
all: help

# 帮助信息
.PHONY: help
help:
	@echo "=========================================="
	@echo "逻辑综合后网表仿真 Makefile"
	@echo "=========================================="
	@echo ""
	@echo "可用目标:"
	@echo "  compile_basic    - 编译基本测试平台"
	@echo "  compile_advanced - 编译高级测试平台"
	@echo "  sim_basic        - 运行基本仿真"
	@echo "  sim_advanced     - 运行高级仿真"
	@echo "  wave             - 查看波形"
	@echo "  clean            - 清理生成文件"
	@echo "  report           - 生成测试报告"
	@echo ""
	@echo "完整流程:"
	@echo "  make compile_advanced"
	@echo "  make sim_advanced"
	@echo "  make wave"
	@echo ""

# 创建目录
$(SIM_DIR) $(WAVE_DIR):
	mkdir -p $@

# 检查网表文件
.PHONY: check_netlist
check_netlist:
	@if [ ! -f "$(NETLIST)" ]; then \
		echo "错误: 网表文件不存在: $(NETLIST)"; \
		echo "请先运行逻辑综合生成网表"; \
		exit 1; \
	fi
	@echo "✓ 找到网表文件: $(NETLIST)"

# 编译基本测试平台
.PHONY: compile_basic
compile_basic: check_netlist $(SIM_DIR)
	@echo "编译基本测试平台..."
	$(VCS) $(VCS_OPTS) \
		-f $(TB_DIR)/filelist.f \
		$(NETLIST) \
		$(TB_BASIC) \
		-o $(SIM_DIR)/simv_basic \
		-l $(SIM_DIR)/compile_basic.log
	@echo "✓ 编译完成"

# 编译高级测试平台
.PHONY: compile_advanced
compile_advanced: check_netlist $(SIM_DIR)
	@echo "编译高级测试平台..."
	$(VCS) $(VCS_OPTS) \
		-f $(TB_DIR)/filelist.f \
		$(NETLIST) \
		$(TB_ADVANCED) \
		-o $(SIM_DIR)/simv_advanced \
		-l $(SIM_DIR)/compile_advanced.log
	@echo "✓ 编译完成"

# 运行基本仿真
.PHONY: sim_basic
sim_basic: $(WAVE_DIR)
	@echo "运行基本仿真..."
	@if [ ! -f "$(SIM_DIR)/simv_basic" ]; then \
		echo "错误: 仿真可执行文件不存在，请先运行 make compile_basic"; \
		exit 1; \
	fi
	cd $(SIM_DIR) && ./simv_basic $(SIM_OPTS) -l sim_basic.log
	@echo "✓ 仿真完成"
	@echo "查看日志: $(SIM_DIR)/sim_basic.log"

# 运行高级仿真
.PHONY: sim_advanced
sim_advanced: $(WAVE_DIR)
	@echo "运行高级仿真..."
	@if [ ! -f "$(SIM_DIR)/simv_advanced" ]; then \
		echo "错误: 仿真可执行文件不存在，请先运行 make compile_advanced"; \
		exit 1; \
	fi
	cd $(SIM_DIR) && ./simv_advanced $(SIM_OPTS) -l sim_advanced.log
	@echo "✓ 仿真完成"
	@echo "查看日志: $(SIM_DIR)/sim_advanced.log"
	@echo "查看报告: $(SIM_DIR)/detailed_report.txt"

# 使用 Verdi 查看波形
.PHONY: wave
wave:
	@if [ -f "$(WAVE_DIR)/advanced_post_syn.vcd" ]; then \
		$(VERDI) -ssf $(WAVE_DIR)/advanced_post_syn.vcd &; \
	elif [ -f "$(WAVE_DIR)/post_syn.vcd" ]; then \
		$(VERDI) -ssf $(WAVE_DIR)/post_syn.vcd &; \
	else \
		echo "错误: 未找到波形文件"; \
		exit 1; \
	fi

# 使用 GTKWave 查看波形（备选）
.PHONY: wave_gtk
wave_gtk:
	@if [ -f "$(WAVE_DIR)/advanced_post_syn.vcd" ]; then \
		gtkwave $(WAVE_DIR)/advanced_post_syn.vcd &; \
	elif [ -f "$(WAVE_DIR)/post_syn.vcd" ]; then \
		gtkwave $(WAVE_DIR)/post_syn.vcd &; \
	else \
		echo "错误: 未找到波形文件"; \
		exit 1; \
	fi

# 生成测试报告
.PHONY: report
report:
	@echo "=========================================="
	@echo "测试报告"
	@echo "=========================================="
	@echo ""
	@if [ -f "$(SIM_DIR)/detailed_report.txt" ]; then \
		cat $(SIM_DIR)/detailed_report.txt; \
	elif [ -f "$(SIM_DIR)/post_syn_report.txt" ]; then \
		cat $(SIM_DIR)/post_syn_report.txt; \
	else \
		echo "未找到测试报告，请先运行仿真"; \
	fi

# 清理
.PHONY: clean
clean:
	@echo "清理生成文件..."
	rm -rf $(SIM_DIR)
	rm -rf $(WAVE_DIR)
	rm -rf csrc
	rm -rf simv*
	rm -rf ucli.key
	rm -rf vc_hdrs.h
	rm -rf DVEfiles
	rm -rf *.log
	@echo "✓ 清理完成"

# 完整流程
.PHONY: full
full: compile_advanced sim_advanced report
	@echo ""
	@echo "=========================================="
	@echo "完整流程完成"
	@echo "=========================================="
	@echo ""
	@echo "下一步:"
	@echo "  make wave      - 查看波形"
	@echo "  make report    - 查看报告"
	@echo ""

# Verilator 仿真（备选方案）
.PHONY: verilator
verilator: check_netlist
	@echo "使用 Verilator 进行仿真..."
	$(VERILATOR) --cc --exe --build \
		--trace \
		-Wall \
		$(NETLIST) \
		$(TB_BASIC) \
		-o $(SIM_DIR)/Vsim
	@echo "✓ Verilator 编译完成"
	$(SIM_DIR)/Vsim
	@echo "✓ Verilator 仿真完成"

.PHONY: info
info:
	@echo "配置信息:"
	@echo "  设计名称: $(DESIGN)"
	@echo "  网表文件: $(NETLIST)"
	@echo "  测试平台: $(TB_DIR)"
	@echo "  仿真目录: $(SIM_DIR)"
	@echo "  波形目录: $(WAVE_DIR)"
	@echo ""
	@echo "工具:"
	@echo "  VCS: $(shell which $(VCS) 2>/dev/null || echo '未安装')"
	@echo "  Verdi: $(shell which $(VERDI) 2>/dev/null || echo '未安装')"
	@echo "  Verilator: $(shell which $(VERILATOR) 2>/dev/null || echo '未安装')"
