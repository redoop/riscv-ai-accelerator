#!/usr/bin/env python3
"""
逻辑综合后网表仿真自动化脚本
支持多种仿真工具和测试场景
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

class PostSynthesisSimulator:
    def __init__(self, design_name="SimpleEdgeAiSoC"):
        self.design_name = design_name
        self.root_dir = Path(__file__).parent
        self.netlist_dir = self.root_dir / "netlist"
        self.tb_dir = self.root_dir / "testbench"
        self.sim_dir = self.root_dir / "sim"
        self.wave_dir = self.root_dir / "waves"
        
        # 创建目录
        self.sim_dir.mkdir(exist_ok=True)
        self.wave_dir.mkdir(exist_ok=True)
    
    def print_header(self):
        """打印标题"""
        print("=" * 60)
        print("逻辑综合后网表仿真")
        print("=" * 60)
        print(f"设计: {self.design_name}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()
    
    def check_netlist(self):
        """检查网表文件是否存在"""
        netlist_file = self.netlist_dir / f"{self.design_name}_syn.v"
        if not netlist_file.exists():
            print(f"❌ 错误: 网表文件不存在: {netlist_file}")
            print("请先运行逻辑综合生成网表")
            return False
        print(f"✓ 找到网表文件: {netlist_file}")
        return True
    
    def check_tool(self, tool_name):
        """检查工具是否可用"""
        try:
            subprocess.run([tool_name, "--version"], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def run_vcs_simulation(self, testbench="advanced"):
        """使用 VCS 运行仿真"""
        print("\n使用 VCS 进行仿真...")
        print("-" * 60)
        
        if not self.check_tool("vcs"):
            print("❌ VCS 未安装或不在 PATH 中")
            return False
        
        netlist = self.netlist_dir / f"{self.design_name}_syn.v"
        if testbench == "basic":
            tb_file = self.tb_dir / "post_syn_tb.sv"
            simv = self.sim_dir / "simv_basic"
        else:
            tb_file = self.tb_dir / "advanced_post_syn_tb.sv"
            simv = self.sim_dir / "simv_advanced"
        
        # 编译
        print("1. 编译...")
        compile_cmd = [
            "vcs",
            "-full64",
            "-sverilog",
            "-timescale=1ns/1ps",
            "+v2k",
            "-debug_all",
            "-kdb",
            "-lca",
            f"-f", str(self.tb_dir / "filelist.f"),
            str(netlist),
            str(tb_file),
            "-o", str(simv),
            "-l", str(self.sim_dir / f"compile_{testbench}.log")
        ]
        
        try:
            subprocess.run(compile_cmd, check=True)
            print("✓ 编译成功")
        except subprocess.CalledProcessError:
            print("❌ 编译失败")
            return False
        
        # 仿真
        print("2. 运行仿真...")
        sim_cmd = [
            str(simv),
            "+vcs+finish+100000000",
            "-l", str(self.sim_dir / f"sim_{testbench}.log")
        ]
        
        try:
            subprocess.run(sim_cmd, check=True, cwd=self.sim_dir)
            print("✓ 仿真成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ 仿真失败")
            return False
    
    def run_verilator_simulation(self):
        """使用 Verilator 运行仿真"""
        print("\n使用 Verilator 进行仿真...")
        print("-" * 60)
        
        if not self.check_tool("verilator"):
            print("❌ Verilator 未安装或不在 PATH 中")
            return False
        
        netlist = self.netlist_dir / f"{self.design_name}_syn.v"
        tb_file = self.tb_dir / "post_syn_tb.sv"
        
        # 编译和仿真
        print("1. 编译和仿真...")
        cmd = [
            "verilator",
            "--cc",
            "--exe",
            "--build",
            "--trace",
            "-Wall",
            str(netlist),
            str(tb_file),
            "-o", str(self.sim_dir / "Vsim")
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Verilator 编译成功")
            
            # 运行仿真
            subprocess.run([str(self.sim_dir / "Vsim")], check=True)
            print("✓ Verilator 仿真成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ Verilator 失败")
            return False
    
    def view_waveform(self, tool="verdi"):
        """查看波形"""
        print(f"\n使用 {tool} 查看波形...")
        
        # 查找波形文件
        wave_files = list(self.wave_dir.glob("*.vcd")) + \
                    list(self.wave_dir.glob("*.fsdb"))
        
        if not wave_files:
            print("❌ 未找到波形文件")
            return False
        
        wave_file = wave_files[0]
        print(f"波形文件: {wave_file}")
        
        if tool == "verdi":
            if self.check_tool("verdi"):
                subprocess.Popen(["verdi", "-ssf", str(wave_file)])
                print("✓ Verdi 已启动")
                return True
        elif tool == "gtkwave":
            if self.check_tool("gtkwave"):
                subprocess.Popen(["gtkwave", str(wave_file)])
                print("✓ GTKWave 已启动")
                return True
        
        print(f"❌ {tool} 未安装")
        return False
    
    def generate_report(self):
        """生成测试报告"""
        print("\n生成测试报告...")
        print("-" * 60)
        
        report_files = [
            self.sim_dir / "detailed_report.txt",
            self.sim_dir / "post_syn_report.txt"
        ]
        
        for report_file in report_files:
            if report_file.exists():
                print(f"\n报告内容 ({report_file.name}):")
                print("=" * 60)
                with open(report_file, 'r', encoding='utf-8') as f:
                    print(f.read())
                return True
        
        print("❌ 未找到测试报告")
        return False
    
    def run_full_flow(self, simulator="vcs", testbench="advanced"):
        """运行完整流程"""
        self.print_header()
        
        # 检查网表
        if not self.check_netlist():
            return False
        
        # 运行仿真
        success = False
        if simulator == "vcs":
            success = self.run_vcs_simulation(testbench)
        elif simulator == "verilator":
            success = self.run_verilator_simulation()
        else:
            print(f"❌ 不支持的仿真器: {simulator}")
            return False
        
        if not success:
            return False
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("完整流程完成")
        print("=" * 60)
        print("\n下一步:")
        print("  查看波形: python run_post_syn_sim.py --wave")
        print("  查看报告: python run_post_syn_sim.py --report")
        print()
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="逻辑综合后网表仿真自动化脚本"
    )
    
    parser.add_argument(
        "--simulator",
        choices=["vcs", "verilator"],
        default="vcs",
        help="选择仿真器 (默认: vcs)"
    )
    
    parser.add_argument(
        "--testbench",
        choices=["basic", "advanced"],
        default="advanced",
        help="选择测试平台 (默认: advanced)"
    )
    
    parser.add_argument(
        "--wave",
        action="store_true",
        help="查看波形"
    )
    
    parser.add_argument(
        "--wave-tool",
        choices=["verdi", "gtkwave"],
        default="verdi",
        help="波形查看工具 (默认: verdi)"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成测试报告"
    )
    
    parser.add_argument(
        "--design",
        default="SimpleEdgeAiSoC",
        help="设计名称 (默认: SimpleEdgeAiSoC)"
    )
    
    args = parser.parse_args()
    
    sim = PostSynthesisSimulator(args.design)
    
    if args.wave:
        sim.view_waveform(args.wave_tool)
    elif args.report:
        sim.generate_report()
    else:
        sim.run_full_flow(args.simulator, args.testbench)

if __name__ == "__main__":
    main()
