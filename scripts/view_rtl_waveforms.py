#!/usr/bin/env python3
"""
RTL波形查看器 - 查看RTL仿真生成的波形文件
"""

import os
import sys
import time
import subprocess
from pathlib import Path

class RTLWaveformViewer:
    """RTL波形查看器"""
    
    def __init__(self):
        self.rtl_path = Path("verification/simple_rtl")
        self.vcd_files = []
        self._find_vcd_files()
    
    def _find_vcd_files(self):
        """查找VCD波形文件"""
        self.vcd_files = list(self.rtl_path.glob("*.vcd"))
        print(f"📊 找到 {len(self.vcd_files)} 个波形文件:")
        for vcd_file in self.vcd_files:
            file_size = vcd_file.stat().st_size
            print(f"  📈 {vcd_file.name} ({file_size} bytes)")
    
    def check_viewers(self):
        """检查可用的波形查看器"""
        available_viewers = []
        
        print("\n🔍 检查可用的波形查看器:")
        
        # 检查GTKWave (多种可能的路径)
        gtkwave_paths = [
            "gtkwave",
            "/opt/homebrew/bin/gtkwave",
            "/usr/local/bin/gtkwave",
            "/Applications/gtkwave.app/Contents/Resources/bin/gtkwave"
        ]
        
        gtkwave_found = False
        for path in gtkwave_paths:
            try:
                result = subprocess.run([path, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_viewers.append("gtkwave")
                    print(f"  ✅ GTKWave: {path}")
                    gtkwave_found = True
                    break
            except:
                continue
        
        if not gtkwave_found:
            # 尝试简单的which命令
            try:
                result = subprocess.run(["which", "gtkwave"], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    available_viewers.append("gtkwave")
                    print(f"  ✅ GTKWave: {result.stdout.strip()}")
                    gtkwave_found = True
            except:
                pass
        
        if not gtkwave_found:
            print("  ❌ GTKWave: 未安装或不可用")
        
        # macOS总是有open命令
        if sys.platform == "darwin":
            available_viewers.append("open")
            print("  ✅ macOS默认应用: 可用")
        
        # 检查其他可能的查看器
        try:
            result = subprocess.run(["which", "code"], capture_output=True, text=True)
            if result.returncode == 0:
                available_viewers.append("vscode")
                print(f"  ✅ VS Code: {result.stdout.strip()}")
        except:
            pass
        
        return available_viewers
    
    def view_with_gtkwave(self, vcd_file):
        """使用GTKWave查看波形"""
        try:
            print(f"🚀 使用GTKWave打开: {vcd_file}")
            
            # 尝试启动GTKWave
            process = subprocess.Popen(["gtkwave", str(vcd_file)], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            
            # 等待一小段时间检查是否启动成功
            import time
            time.sleep(1)
            
            if process.poll() is None:
                print("✅ GTKWave已启动")
                return True
            else:
                # 获取错误信息
                stdout, stderr = process.communicate()
                if stderr:
                    print(f"❌ GTKWave启动失败: {stderr.decode()}")
                    print("💡 尝试修复GTKWave Perl依赖问题...")
                    self._fix_gtkwave_perl_issue()
                else:
                    print("❌ GTKWave启动失败")
                return False
                
        except Exception as e:
            print(f"❌ GTKWave启动失败: {e}")
            return False
    
    def view_with_macos_open(self, vcd_file):
        """使用macOS默认应用打开"""
        try:
            print(f"🚀 使用macOS默认应用打开: {vcd_file}")
            subprocess.Popen(["open", str(vcd_file)])
            print("✅ 已使用默认应用打开")
            return True
        except Exception as e:
            print(f"❌ 默认应用打开失败: {e}")
            return False
    
    def _fix_gtkwave_perl_issue(self):
        """修复GTKWave的Perl依赖问题"""
        print("\n� GTKWave 装Perl依赖修复:")
        print("=" * 40)
        print("检测到GTKWave缺少Perl Switch模块")
        print("")
        print("🛠️ 修复方法:")
        print("1. 安装Perl Switch模块:")
        print("   sudo cpan Switch")
        print("")
        print("2. 或者重新安装GTKWave:")
        print("   brew uninstall --cask gtkwave")
        print("   brew install --cask gtkwave")
        print("")
        print("3. 或者使用替代方案:")
        print("   - 使用macOS默认应用打开VCD文件")
        print("   - 使用在线波形查看器")
        print("   - 使用文本编辑器查看VCD内容")
    
    def install_gtkwave_instructions(self):
        """显示GTKWave安装说明"""
        print("\n📦 GTKWave安装说明:")
        print("=" * 40)
        
        if sys.platform == "darwin":
            print("🍺 macOS (推荐使用Homebrew):")
            print("  brew install --cask gtkwave")
            print("")
            print("🔧 如果遇到Perl依赖问题:")
            print("  sudo cpan Switch")
            print("")
            print("🔗 或者从官网下载:")
            print("  https://gtkwave.sourceforge.net/")
        
        elif sys.platform.startswith("linux"):
            print("🐧 Linux:")
            print("  # Ubuntu/Debian:")
            print("  sudo apt-get install gtkwave")
            print("")
            print("  # CentOS/RHEL:")
            print("  sudo yum install gtkwave")
        
        print("\n✨ GTKWave是查看VCD波形文件的最佳工具!")
        print("💡 如果GTKWave有问题，可以使用替代方案查看波形")
    
    def view_vcd_as_text(self, vcd_file):
        """以文本形式查看VCD文件内容"""
        try:
            print(f"\n� 文本查看V件CD文件: {vcd_file.name}")
            print("=" * 50)
            
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            print("📋 VCD文件头部信息:")
            header_lines = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('$dumpvars') or header_lines > 50:
                    break
                print(f"  {line.rstrip()}")
                header_lines += 1
            
            print(f"\n📊 信号变化数据 (最后20行):")
            for line in lines[-20:]:
                line = line.rstrip()
                if line:
                    print(f"  {line}")
            
            print(f"\n💡 提示: 完整内容请使用 'cat {vcd_file}' 查看")
            
        except Exception as e:
            print(f"❌ 查看VCD文件失败: {e}")
    
    def analyze_vcd_content(self, vcd_file):
        """分析VCD文件内容"""
        try:
            print(f"\n🔍 分析波形文件: {vcd_file.name}")
            print("=" * 50)
            
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            # 统计信息
            total_lines = len(lines)
            print(f"📊 文件统计:")
            print(f"  总行数: {total_lines}")
            print(f"  文件大小: {vcd_file.stat().st_size} bytes")
            
            # 查找信号定义
            signals = []
            modules = []
            
            for line in lines[:100]:  # 只检查前100行
                line = line.strip()
                if line.startswith('$var'):
                    # $var wire 1 ! clk $end
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_name = parts[4]
                        signals.append(signal_name)
                elif line.startswith('$scope'):
                    # $scope module test_simple_tpu_mac $end
                    parts = line.split()
                    if len(parts) >= 3:
                        module_name = parts[2]
                        modules.append(module_name)
            
            print(f"\n📡 发现的模块:")
            for module in set(modules):
                print(f"  🔧 {module}")
            
            print(f"\n📈 发现的信号 (前10个):")
            for signal in signals[:10]:
                print(f"  📊 {signal}")
            
            if len(signals) > 10:
                print(f"  ... 还有 {len(signals) - 10} 个信号")
            
            # 查找时间范围
            time_values = []
            for line in lines[-50:]:  # 检查最后50行
                if line.startswith('#'):
                    try:
                        time_val = int(line[1:])
                        time_values.append(time_val)
                    except:
                        pass
            
            if time_values:
                print(f"\n⏰ 仿真时间范围:")
                print(f"  开始时间: 0")
                print(f"  结束时间: {max(time_values)}")
                print(f"  时间单位: 1ps (根据timescale)")
            
        except Exception as e:
            print(f"❌ 分析VCD文件失败: {e}")
    
    def create_simple_waveform_html(self, vcd_file):
        """创建简单的HTML波形查看器"""
        try:
            print(f"\n🌐 创建HTML波形查看器: {vcd_file.name}")
            
            html_file = vcd_file.parent / f"{vcd_file.stem}_waveform.html"
            
            # 解析VCD文件
            signals = {}
            time_values = []
            
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            # 解析信号定义
            for line in lines:
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_name = parts[4]
                        signal_id = parts[3]
                        signals[signal_id] = signal_name
                elif line.startswith('#'):
                    try:
                        time_val = int(line[1:])
                        time_values.append(time_val)
                    except:
                        pass
            
            # 创建HTML内容
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RTL波形查看器 - {vcd_file.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .signal {{ margin: 5px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }}
        .time-info {{ background: #e8f4fd; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #2196F3; }}
        .vcd-content {{ background: #2d3748; color: #e2e8f0; padding: 15px; font-family: 'Courier New', monospace; 
                       white-space: pre-wrap; max-height: 400px; overflow-y: auto; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .section h3 {{ color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat-box {{ background: #f7fafc; padding: 15px; border-radius: 5px; border: 1px solid #e2e8f0; flex: 1; min-width: 200px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 RTL波形查看器</h1>
            <p><strong>文件:</strong> {vcd_file.name}</p>
            <p><strong>路径:</strong> {vcd_file}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h4>📊 文件统计</h4>
                <p><strong>大小:</strong> {vcd_file.stat().st_size:,} bytes</p>
                <p><strong>行数:</strong> {len(lines):,}</p>
                <p><strong>生成时间:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="stat-box">
                <h4>⏰ 仿真信息</h4>
                <p><strong>时间范围:</strong> 0 - {max(time_values) if time_values else 0:,} ps</p>
                <p><strong>时间点数:</strong> {len(time_values):,}</p>
                <p><strong>信号数量:</strong> {len(signals)}</p>
            </div>
        </div>
        
        <div class="section">
            <h3>📡 信号列表</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px;">
                {chr(10).join(f'<div class="signal">📊 <strong>{name}</strong> <small>(ID: {sid})</small></div>' for sid, name in list(signals.items())[:20])}
                {f'<div class="signal">... 还有 {len(signals) - 20} 个信号</div>' if len(signals) > 20 else ''}
            </div>
        </div>
        
        <div class="section">
            <h3>📄 VCD文件内容预览</h3>
            <div class="vcd-content">{''.join(lines[:100])}</div>
            <p><em>显示前100行，完整内容请查看原始VCD文件</em></p>
        </div>
        
        <div class="time-info">
            <h3>💡 使用说明</h3>
            <ul>
                <li>这是一个简单的VCD文件查看器，用于快速预览波形信息</li>
                <li>推荐使用专业工具如GTKWave查看完整的时序波形图</li>
                <li>安装GTKWave: <code>brew install --cask gtkwave</code></li>
                <li>或者使用在线波形查看器: <a href="https://wavedrom.com/" target="_blank">WaveDrom</a></li>
            </ul>
        </div>
    </div>
</body>
</html>"""
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ HTML文件已创建: {html_file}")
            
            # 尝试在浏览器中打开
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(html_file)])
                print("🌐 已在浏览器中打开")
            
            return html_file
            
        except Exception as e:
            print(f"❌ 创建HTML查看器失败: {e}")
            return None
    
    def run_rtl_and_view(self):
        """运行RTL仿真并查看波形"""
        print("🔧 运行RTL仿真生成新的波形...")
        
        # 运行RTL后端生成波形
        try:
            from rtl_hardware_backend import RTLHardwareBackend
            backend = RTLHardwareBackend()
            backend.test_rtl_connection()
            print("✅ RTL仿真完成")
        except Exception as e:
            print(f"⚠️ RTL仿真失败: {e}")
        
        # 重新查找VCD文件
        self._find_vcd_files()
    
    def interactive_viewer(self):
        """交互式波形查看器"""
        print("🌊 RTL波形查看器")
        print("=" * 40)
        
        if not self.vcd_files:
            print("❌ 未找到VCD波形文件")
            print("🔧 尝试运行RTL仿真生成波形...")
            self.run_rtl_and_view()
            
            if not self.vcd_files:
                print("❌ 仍然没有波形文件，请检查RTL仿真")
                return
        
        # 检查可用的查看器
        available_viewers = self.check_viewers()
        
        while True:
            print(f"\n📊 可用的波形文件:")
            for i, vcd_file in enumerate(self.vcd_files):
                print(f"  {i+1}. {vcd_file.name}")
            
            print(f"\n🛠️ 操作选项:")
            print("  v) 查看波形文件 (GTKWave)")
            print("  o) 使用默认应用打开")
            print("  h) 创建HTML波形查看器")
            print("  t) 文本查看VCD内容")
            print("  a) 分析波形内容")
            print("  r) 重新运行RTL仿真")
            print("  i) 显示GTKWave安装说明")
            print("  q) 退出")
            
            try:
                choice = input("\n请选择操作: ").strip().lower()
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出程序")
                break
            except EOFError:
                print("\n\n👋 输入结束，退出程序")
                break
            
            if choice == 'q':
                break
            elif choice == 'v':
                if not self.vcd_files:
                    print("❌ 没有可用的波形文件")
                    continue
                
                # 选择文件
                try:
                    file_idx = int(input(f"选择文件 (1-{len(self.vcd_files)}): ")) - 1
                    if 0 <= file_idx < len(self.vcd_files):
                        vcd_file = self.vcd_files[file_idx]
                        
                        # 尝试使用GTKWave
                        if "gtkwave" in available_viewers:
                            success = self.view_with_gtkwave(vcd_file)
                            if not success:
                                print("💡 GTKWave启动失败，尝试使用默认应用...")
                                if "open" in available_viewers:
                                    self.view_with_macos_open(vcd_file)
                        else:
                            print("❌ GTKWave不可用")
                            self.install_gtkwave_instructions()
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n操作被中断")
            
            elif choice == 'o':
                if not self.vcd_files:
                    print("❌ 没有可用的波形文件")
                    continue
                
                try:
                    file_idx = int(input(f"选择文件 (1-{len(self.vcd_files)}): ")) - 1
                    if 0 <= file_idx < len(self.vcd_files):
                        vcd_file = self.vcd_files[file_idx]
                        if "open" in available_viewers:
                            self.view_with_macos_open(vcd_file)
                        else:
                            print("❌ 默认应用不可用")
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n操作被中断")
            
            elif choice == 'h':
                if not self.vcd_files:
                    print("❌ 没有可用的波形文件")
                    continue
                
                try:
                    file_idx = int(input(f"选择文件 (1-{len(self.vcd_files)}): ")) - 1
                    if 0 <= file_idx < len(self.vcd_files):
                        vcd_file = self.vcd_files[file_idx]
                        self.create_simple_waveform_html(vcd_file)
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n操作被中断")
            
            elif choice == 't':
                if not self.vcd_files:
                    print("❌ 没有可用的波形文件")
                    continue
                
                try:
                    file_idx = int(input(f"选择要查看的文件 (1-{len(self.vcd_files)}): ")) - 1
                    if 0 <= file_idx < len(self.vcd_files):
                        self.view_vcd_as_text(self.vcd_files[file_idx])
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
                except KeyboardInterrupt:
                    print("\n操作被中断")
            
            elif choice == 'a':
                if not self.vcd_files:
                    print("❌ 没有可用的波形文件")
                    continue
                
                try:
                    file_idx = int(input(f"选择要分析的文件 (1-{len(self.vcd_files)}): ")) - 1
                    if 0 <= file_idx < len(self.vcd_files):
                        self.analyze_vcd_content(self.vcd_files[file_idx])
                    else:
                        print("❌ 无效的文件编号")
                except ValueError:
                    print("❌ 请输入有效的数字")
            
            elif choice == 'r':
                self.run_rtl_and_view()
            
            elif choice == 'i':
                self.install_gtkwave_instructions()
            
            else:
                print("❌ 无效选择")

def main():
    """主程序"""
    viewer = RTLWaveformViewer()
    
    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == "--analyze":
            for vcd_file in viewer.vcd_files:
                viewer.analyze_vcd_content(vcd_file)
        elif sys.argv[1] == "--install":
            viewer.install_gtkwave_instructions()
        elif sys.argv[1] == "--run":
            viewer.run_rtl_and_view()
        elif sys.argv[1] == "--view":
            # 快速查看模式
            if viewer.vcd_files:
                main_vcd = viewer.vcd_files[0]  # 使用第一个VCD文件
                print(f"🚀 快速查看: {main_vcd.name}")
                
                available_viewers = viewer.check_viewers()
                if "gtkwave" in available_viewers:
                    success = viewer.view_with_gtkwave(main_vcd)
                    if not success and "open" in available_viewers:
                        viewer.view_with_macos_open(main_vcd)
                elif "open" in available_viewers:
                    viewer.view_with_macos_open(main_vcd)
                else:
                    print("❌ 没有可用的波形查看器")
                    viewer.install_gtkwave_instructions()
            else:
                print("❌ 没有找到VCD文件")
        elif sys.argv[1] == "--text":
            # 文本查看模式
            for vcd_file in viewer.vcd_files:
                viewer.view_vcd_as_text(vcd_file)
        elif sys.argv[1] == "--help":
            print("RTL波形查看器使用说明:")
            print("=" * 40)
            print("  python3 view_rtl_waveforms.py              # 交互模式")
            print("  python3 view_rtl_waveforms.py --view       # 快速查看")
            print("  python3 view_rtl_waveforms.py --analyze    # 分析波形")
            print("  python3 view_rtl_waveforms.py --text       # 文本查看")
            print("  python3 view_rtl_waveforms.py --run        # 运行RTL仿真")
            print("  python3 view_rtl_waveforms.py --install    # 安装说明")
            print("  python3 view_rtl_waveforms.py --help       # 显示帮助")
        else:
            print("❌ 未知参数，使用 --help 查看帮助")
    else:
        # 交互模式
        try:
            viewer.interactive_viewer()
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见!")
        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()