#!/usr/bin/env python3
"""
ECOS 项目仿真自动化脚本
支持 VCS 和 Icarus Verilog 仿真器
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

class EcosSimulator:
    def __init__(self):
        self.design = "soc_tb"
        self.root_dir = Path(__file__).parent
        self.filelist_dir = self.root_dir.parent / "filelist"
        self.tb_dir = self.root_dir.parent / "tb"
        self.netlist_dir = self.root_dir.parent.parent / "netlist"
        
        # 仿真模式
        self.sim_mode = "rtl"  # rtl 或 netlist
        
        # 网表路径, title):
        self.netlist_dir = self.synthesis_root / "netlist"
        
        # PDK 路径
        self.pdk_root = self.synthesis_root / "pdk" / "icsprout55-pdk"
        self.pdk_verilog = self.pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"
        
        # Chisel 生成的 RTL
        self.chisel_rtl = self.synthesis_root.parent / "generated" / "simple_edgeaisoc" / "SimpleEdgeAiSoC.sv"
        
    def print_header(self, mode="rtl"):
    def check_tool(self, tool_name):
        print(具是否可用"""
        try:
            subprocess.run([tool_name, "--version"], 
        print(f"设计: {self.desure_output=True, 
                         check=True,
                         timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_file_list(self, filelist_name):
        """读取文件列表"""
        filelist_path = self.filelist_dir / filelist_name
        if not filelist_path.exists():
            return []
        
        files = []
        with open(filelist_path, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if line and not line.startswith('#') and not line.startswith('//'):
                    # 处理相对路径
                    if line.startswith('$'):
                        # 跳过环境变量
                        continue
                    files.append(line)
        return files
    
    def run_vcs_simulation(self, ip_sel=None, prog=None, wave=False):
        """使用 VCS 运行仿真"""
        self.print_header("VCS 仿真")
                # 跳过注释和空行
                if line and not line.startswith('#') and not line.startswith('//'):
                    # 处理环境变量
                    if '$RTL_PATH' in line:
                        # 替换为相对路径
                        line = line.replace('$RTL_PATH', str(self.project_root))
                    files.append(line)
        
            "-full64",
            "+v2k",
            "-sverilog",
        """获取 RTL 文件列表"""
        files = []
        for list_file in ['asic_top.f', 'ip.f', 'lib.f', 'soc.f']:
            files.extend(self.get_file_list(list_file))
        return filessh+all",
            "+lint=TFIPC-L",
            "+define+no_warning",
            "+define+S50",
            "+define+SVA_OFF",
            "-work", "DEFAULT",
            "+define+RANDOMIZE_REG_INIT",
            "+define+PDK_BEHAV",
            "+notimingcheck",
            "+nospecify"
        
        if not self.check_tool("vcs"):
            print("❌ VCS 未安装或不在 PATH 中")
            return False
        
        # 获取文件列表
        rtl_"../tb",
            "../tb/include"
        ]
        
        for inc_dir in include_dirs:
            vcs_opts.append(f"+incdir+{inc_dir}")
        
        # 添加文件列表
        filelists = ["asic_top.f", "ip.f", "lib.f", "soc.f"]
        for flist in filelists:
            flist_path = self.filelist_dir / flist
            if flist_path.exists():
                vcs_opts.extend(["-f", str(flist_path)])
        
        # 添加测试平台
        tb_flist = self.filelist_dir / "asic_tblist.f"
        if tb_flist.exists():
            vcs_opts.extend(["-f", str(tb_flist)])
        
        vcs_opts.extend(["-top", self.design])
        vcs_opts.extend(["-l", "compile.log
            "-work", "DEFAULT",
            "+define+RANDOMIZE_REG_INIT",
        print("1. 编译设计...")
        print(f"   命令: {' '.join(vcs_opts[:5])} ...")
        try:
            result = subprocess.run(vcs_opts, cwd=self.root_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print("编译输出:")
                print(result.stdout)
                print(result.stderr)
                print("❌ 编译失败")
                return False
            print("✓ 编译成功")
        except Exception as e:
            print(f"❌ 编译失败: {e}")
            return False
        
        # 运行仿真
        print("\n2. 运行仿真...")
        sim_opts = ["./simv", "-l", "sim.log"]
        
        # 添加测试选项
        if ip_sel:
            sim_opts.append(f"+ip_sel{ip_sel:02d}")
        if prog:
            sim_opts.append(f"+{prog}")
        if wave:
            sim_opts.append("+dump_all")
        
        try:
            result = subprocess.run(sim_opts, cwd=self.root_dir, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("警告/错误:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("✓ 仿真成功")
                return True
            else:
                print("❌ 仿真失败")
                return False
        except Exception as e:
            print(f"❌ 仿真失败: {e}")
            return False
    
    def run_iverilog_simulation(self, netlist=False):
        """使用 Icarus Verilog 运行仿真"""
        self.sim_mode = "netlist" if netlist else "rtl"
        self.print_header("Icarus Verilog 仿真")
        
        # 检查 iverilog
        iverilog_path = "/opt/tools/oss-cad/oss-cad-suite/bin/iverilog"
        vvp_path = "/opt/tools/oss-cad/oss-cad-suite/bin/vvp"
        
        if not Path(iverilog_path).exists():
            if not self.check_tool("iverilog"):
                print("❌ Icarus Verilog 未安装")
                return False
            iverilog_path = "iverilog"
            vvp_path = "vvp"
        
        print(f"✓ 使用 Icarus Verilog: {iverilog_path}")
        
        # 使用 Makefile.iverilog
        makefile = self.ro               print("❌ 编译失败")
                return False
            print(f"❌ 未找到 Makefile: {makefile}")
            return False
        
        # 运行 make
        print("\n1. 编译和仿真...")
        sim_mode = "netlist" if netlist else "rtl"
        
        try:
            result = subprocess.run(
                ["make", "-f", "Makefile.iverilog", "sim", f"SIM_MODE={sim_mode}"],
                cwd=self.root_dir,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode == 0:
                print("\n✓ 仿真成功")
                
                # 检查波形文件
                wave_file = self.root_dir / f"soc_tb_{sim_mode}.vcd"
              print("✓ 编译成功")
        except Exception as e:
                    print(f"✓ 波形文件: {wave_file} ({size_mb:.1f} MB)")
                
                return True
            print(f"❌ 编译失败: {e}")
            return False
        
        # 构建仿真命令xception as e:
            print(f"❌ 仿真失败: {e}")
            return False
    
    def view_waveform(self, tool="gtkwave"):
        """查看波形"""
        print(f"\n使用 {tool} 查看波形...")
        
        # 查找波形文件
        wave_files = list(self.root_dir.glob("*.vcd")) + \
            
        sim_cmd = ["./simv", "-l", "sim.log"]
        
        # 添加测试参数
        if ip_sel:
            sim_cmd.append(f"+ip_sel{ip_sel:02d}")
        wave_file = wave_files[0]
        print(f"波形文件: {wave_file}")
        
        if tool == "verdi":
            if self.check_tool("verdi"):
                try:
                    subprocess.Popen(["verdi", "-ssf", str(wave_file), "-nologo"])
                    print("✓ Verdi 已启动")
                    return True
                except Exception as e:
                    print(f"❌ 启动 Verdi 失败: {e}")
            else:
                print("❌ Verdi 未安装")
        elif tool == "gtkwave":
            if not os.environ.get('DISPLAY'):
                print("❌ 无法启动 GTKWave: 未检测到图形显示环境")
                print(f"   波形文件位置: {wave_file}")
                return False
            
            if self.check_tool("gtkwave"):
                try:
                    subprocess.Popen(["gtkwave", str(wave_file)])
                    print("✓ GTKWave 已启动")
                    return True
                except Exception as e:
                    print(f"❌ 启动 GTKWave 失败: {e}")
            else:
                print("❌ GTKWave 未安装")
        
        return False
    
    def clean(self):
        """清理生成的文件"""
        print("清理仿真文件...")
        
        patterns = [
            "*.vvp", "*.vcd", "*.log", "*.fsdb", "*.vpd",
            "csrc", "*simv.daidir*", "*simv*", "ucli.key",
            "vcdplus.vpd", "DVEfiles", "INCA_libs", "novas.*",
            "verdiLog"
        ]
        
        import glob
        import shutil
        
        for pattern in patterns:
            for item in glob.glob(str(self.root_dir / pattern)):
                try:
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
                    print(f"  删除: {item}")
                except Exception as e:
                    print(f"  警告: 无法删除 {item}: {e}")
        
        print("✓ 清理完成")

def main():
    parser = argparse.ArgumentParser(
        description="ECOS 项目仿真自动化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # VCS RTL 仿真
  python run_sim.py --simulator vcs --ip-sel 1 --prog hello-mem
  
  # Icarus Verilog RTL 仿真
  python run_sim.py --simulator iverilog
  
  # Icarus Verilog 网表仿真
  python run_sim.py --simulator iverilog --netlist
  
  # 查看波形
  python run_sim.py --wave --wave-tool gtkwave
  
  # 清理
  python run_sim.py --clean
        """
    )
    
    parser.add_argument(
        "--simulator",
        choices=["vcs", "iverilog"],
        default="iverilog",
        help="选择仿真器 (默认: iverilog)"
    )
    
    parser.add_argument(
        "--ip-sel",
        type=int,
        choices=ran     if prog:
            sim_cmd.append(f"+{prog}")
        if wave:
            sim_cmd.append("+dump_all")
        
        # 仿真
        print("\n2. 运行仿真...")
        if ip_sel:
            print(f"   IP 选择: ip_sel{ip_sel:02d}")m"],
        help="程序选择"
    )
    
    parser.add_argument(
        "--wave",
        action="store_true",
        help="查看波形"
    )
    
    parser.add_argument(
        "--wave-tool",
        choices=["verdi", "gtkwave"],
        default="gtkwave",
        help="波形查看工具 (默认: gtkwave)"
    )
    
    parser.add_argument(
        "--netlist",
        action="store_true",
        help="使用网表仿真 (仅 iverilog)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理生成的文件"
    )
    
    args = parser.parse_args()
    
    sim = EcosSimulator()
    
    if args.clean:
        sim.clean()
    elif args.wave:
        sim.view_waveform(args.wave_tool)
    else:
        if args.simulator == "vcs":
            sim.run_vcs_simulation(args.ip_sel, args.prog, args.wave)
        elif args.simulator == "iverilog":
            sim.run_iverilog_simulation(ar
        if prog:
     wave:
            print(f"   波形:       try:
            result = subprocess.run(sim_cmd, cwd=self.root_dir, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("警告/错误:")
                print(result.stderr)
            
            if result.returncode == 0:
                print("✓ 仿真成功            print("❌int(f"❌ 仿真失败: {e}")
            return False
    
    def run_iverilog_rtl_simulation(self, wave=False):
        """使用 Icarus Verilog 运行 RTL 仿真"\n使用 Icarus Verilog 进行 RTL 仿真...")
        print("-" * 70)
        
        # 检查 iverilog
        iverilog_path = "/opt/tools/oss-cad/oss-
        vvp_path = "/opt/tools/oss-cad/oss-cad-suite/bin/vvp"
        
        if not Path(iverilog_path).exists():
            if not self.check_tool("iverilog"):
                print("❌ Icarus Verilog 未安装")
                return False
            iverilog_path = "iverilog"
            vvp_path = "vvp"
        else:
            print(f"✓ 找到 Icarus Verilog: {iverilog_path}")
        
        # 获取文件列表
        rtl_files = self.get_rtl_files()
        tb_files = self.get_tb_files()
        
        print(f"RTL 文件: {len(rtl_files)} 个"     print(f"测试平台文件: {len(tb_files)} 个")
        
        # 构建编译命令
        compile_cmd = [
            iverilog_path,
            "-g2005-sv",
  LOG",
            "-DRANDOMIZE_REG_INIT",
            "-DPDK_BEHAV",
        ]
        
        # 添加 include 目录
        include_dirs = [
            "../top",
            "../utils",
            "../tb",
            "../tb/include",
            "../../../generated/simple_edgeaisoc        for inc_dir in include_dirs:
            compile_cmd.append(f"-I{inc_dir}")
        
        # 输出文件
        vvp_file = "soc_tb_rtl.vvp"
        compile_cmd.extend(["-o", vvp_file])
        
        # 顶层模块
        compile_cmd.extend(["-s", self.design])
        
        # 添加文件
        compile_cmd.extend(rtl_files)
        compile_cmd.extend(tb_files)
        
    t = subprocess.run(compile_cmd, cwd=self.root_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print("编译输出:")
                print(result.stdout)
                print(result.stderr)
                print("❌ 编译失败")
                
                # 保存日志
                with open(self.root_dir / "comperr)
                print(f"详细日志已保存到: compile_rtl.log")
                return False
            print("✓ 编译成功")
        except Exception as e:
            print(f"❌ 编译失败: {e}")
            return False
        
        # 仿真
        print("\n2. 运行仿真...")
        sim_cmd = [vvp_path, vvp_file]
        
        try:
            result = subprocess.run(sim_cmd, cwd=self.root_dir, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("警告/错误:")
                print(result.stderr)
            
            # 保存日志
            with open(self.root_dir / "sim_rtl.log", 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            if result.returncode == 0:
                print("✓ 仿真成功")
                
                # 检查波形文件
                vcd_file = self.root_dir / "soc_tb_rtl.vcd"
                if vcd_file.exists():
                    size_mb = vcd_file.stat().st_size / (1024 * 1024)
                    print(f"✓ 波形文件: {vcd_file} ({size_mb:.1f} MB)")
                
                return True
            else:
                print("❌ 仿真失败")
                return False
        except Exception as e:
            print(f"❌ 仿真失败: {e}")
            return False
    
    def run_iverilog_netlist_simulation(self):
        """使用 Icarus Verilog 运行网表仿真"""
        print("\n使用 Icarus Verilog 进行网表仿真...")
        print("-" * 70)
        
        # 检查 iverilog
        iverilog_path = "/opt/tools/oss-cad/oss-cad-suite/bin/iverilog"
        vvp_path = "/opt/tools/oss-cad/oss-cad-suite/bin/vvp"
        
            p     if not Path(iverilog_pt netlist_file.exists():
   rin      pt运行(f"❌ 网表文件不存在: {netlist_file}")
      rint("请先综合生成网表")
        pri: {netlin           return False
 t(f    
 "✓ 网表文件st_file}")
           # 检查 elf.pdkPDKxists():
 log
        if not s_verilog.e           print(f"❌ PDK Veri 文件不存在: {self.pdk_verilog}")
            return False
        print(f"✓ PDK 文件: {self.pdk_verilog}")
        
        # 测试平台
        tb_file = 测试平台: {tb_file}")
        # 构建编译命令
        compile_cmd = [
            iveril_BEHAV",
  og_path, include 目录
            "../ub",
        ]se",
            "DVEfiles",
            "verdiLog"
        ]
        
        count = 0
        for pattern in patterns:
            for item in self.root_dir.glob(pattern):
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
                    count += 1
        
        print(f"✓ 清理完成，删除了 {count} 个文件/目录")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="ECOS 项目仿真工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # RTL 仿真 (Icarus Verilog)
  python run_sim.py --mode rtl --simulator iverilog
  
  # RTL 仿真 (VCS)
  python run_sim.py --mode rtl --simulator vcs --ip-sel 1 --prog hello-mem
  
  # 网表仿真
  python run_sim.py --mode netlist
  
  # 查看波形
  python run_sim.py --wave --mode rtl
  
  # 清理
  python run_sim.py --clean
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["rtl", "netlist"],
        default="rtl",
        help="仿真模式: rtl (默认) 或 netlist"
    )
    
    parser.add_argument(
        "--simulator",
        choices=["vcs", "iverilog"],
        default="iverilog",
        help="仿真器: vcs 或 iverilog (默认)"
    )
    
    parser.add_argument(
        "--ip-sel",
        type=int,
        choices=range(6),
        help="IP 选择 (0-5)，仅用于 VCS"
    )
    
    parser.add_argument(
        "--prog",
        help="程序名称，如: hello-mem, hello-flash 等"
    )
    
    parser.add_argument(
        "--wave",
        action="store_true",
        help="查看波形"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理生成的文件"
    )
    
    args = parser.parse_args()
    
    sim = EcosSimulator()
    
    if args.clean:
        sim.clean()
        return
    
    if args.wave:
        sim.view_waveform(args.mode)
        return
    
    # 运行仿真
    sim.print_header(args.mode)
    
    success = False
    if args.mode == "rtl":
        if args.simulator == "vcs":
            success = sim.run_vcs_rtl_simulation(
                ip_sel=args.ip_sel,
                prog=args.prog,
                wave=True
            )
        else:  # iverilog
            success = sim.run_iverilog_rtl_simulation(wave=True)
    else:  # netlist
        success = sim.run_iverilog_netlist_simulation()
    
    if success:
        print("\n" + "=" * 70)
        print("仿真完成")
        print("=" * 70)
        print("\n下一步:")
        print(f"  查看波形: python run_sim.py --wave --mode {args.mode}")
        print(f"  清理文件: python run_sim.py --clean")
        print()
    else:
        print("\n" + "=" * 70)
        print("仿真失败")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
novas.*",
            "          "INCA_libs",
  "ucli.key",
              "csrc",
          
            "*.daidir",
            "simv*            "*.vpd",
 ns = [",
   
            "*.fsdb",.log",
            "*
            "*.vcd  "*.vvp",
          
        patter   
         print("-" * 70)
      d   print("\n清理生成的文件...")
  ef c""清理生成的文件"""
     lean(self):
        "
        wave"):Falnot os.environ.get('DISPLAY'):
        return   print("请使用波形查看工具打开")
      rint(f"\n波形文件位置: {wave_file}")
        p      
                  print("❌ 无法启动 GTKWave: 未检测到图形显示环境")
                print(f"\n波形文件位置: {wave_file}")
                return False
            try:
                subprocess.Popen(["gtkwave", str(wave_file)])
                print("✓ GTKWave 已启动")
                return True
            except Exception as e:
            if 
        if self.check_tool("gtk  # 尝试 GTKWave
           
   for inc_dir_cmd.append(f"-I{inc_dir}")
             urint(f"文件大小: {size_mb:.1f} MB")
   n False}")e / (1024 * 1024)
        pr
        size_mb = wave_file.stat().st_siz
        print(f"波形文件: {wave_file    
    
        # 输出文件形文件")
            ret
        i   print("❌ 未找到波f not wave_file:
               vvp_file = "soc_tb_netlist.vvp"
          com     break
        
pile_cmd.extend(["-o", vvp_file])
           in includes:
                   wave_file = files[0]
      _dirs:f.root_dir.glob(pattern))
            if file
        for pattern in t(selwave_patterns:
            files = lis  wave_file = None
            compile     "../tb/include"
           
        print("-" * 70)
        #f mode == "terns = ["soc_rtl":
            wave_pat 查找波形文件
        i        
    return pt 仿e
    s   elfprint(f"\n查看波形 ({mode} 模式)...")
     , mode="r"
        tl"):
        """查看波形""
    def view_waveform(真失败: {e}")
            return Falsion as e:
            print(f"❌False
        except Excecompile_c("❌ 仿真超时 (5分钟)")
        md.append(str(red:
            print
         xcept subprocess.TimeoutExpi       return False
        enetlist_file))
        compile print("❌ 仿真失败")_cmd.append(str(se_file))cwd=self.root_dir, capture_output=True, text=True)
                    else:
                  return True
             i   
    f result.returncode != 0:
               exceprinrn False / "soists():ize_mb:.1f} MB)")
      
                    print(f"✓ 波形文件: {vcd_file} ({s
                    size_mb = vcd_file.stat().st_size / (1024 * 1024)c_tb_netlist.vcd"
                if vcd_file.ex
        300)esult.stfile = self.root_dirdout)
                vcd_      # 检查波形文件
                  
         print("警告/错成功")
               误:")r)
            w') f.writ"✓ 仿真e(result.stderr)
                print(
            urncode == 0:
            if result.retas f:ut)
                
                f.write(result.stdo
            with open(self.root_dir / "sim_netlist.log", '        # 保存日志
    
                print(result.stder      if result.stderr:
      
            print(r
        print("\n2. 运行仿真...")_file]utput=True, text=True, timeout=
        ir, capture_o
        try: subprocess.run(sim_cmd, cwd=self.root_d
            result =
        sim_cmd = [vvp_path, vvp    # 仿真
    t(f"❌ 编译失败: {e}")
            retut Exception as e:
            p        printn F功")
        alse
            print("✓ 编译成("编译输出:"")e_netmpile_netlist.log")
                returlist.log", 'w') as f:
                print(f"详细日志已保存到: co            f.write(result.stderr)
               f.write(result.stdout)
             
                # 保存日志elf.root_dir / "compil
                with open(s        
        )
                print("❌ 编译失败int(result.stderr)
                prt(result.stdout)
                prin
        print(f"   网表: {ne     # 编译
        print("\n1. 编译...")
     
      lf.pdk_verilog))
        compile_cmd.append(str(tb   tils",
            "../t../top",
            "        include_dirs = [

            "-DPDK            "-g2005-sv"E_REG_INIT",
,
            "-DRANDOMIZ         "-DIVERILOG",
         "-Wno-timescale",
           "-Wall",
            
    self.e
        print(f"✓tbists():
            return Fals      print(f"❌ 测试平台不存在: {tb_file}")
      _dir / "netlist_tb.sv"
        if not tb_file.ex