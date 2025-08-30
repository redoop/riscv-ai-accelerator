#!/usr/bin/env python3
"""
RISC-V AI加速器硬件检查脚本
检查硬件连接状态、驱动程序和系统兼容性
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path

def run_command(cmd, capture_output=True, shell=True):
    """安全地运行系统命令"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=capture_output, 
                              text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return False, "", str(e)

def check_system_info():
    """检查系统基本信息"""
    print("=== 系统信息 ===")
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "kernel": platform.release() if platform.system() == "Linux" else "N/A"
    }
    
    for key, value in system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return system_info

def check_riscv_ai_compatibility():
    """检查RISC-V AI加速器兼容性"""
    print("\n=== RISC-V AI加速器兼容性 ===")
    
    system = platform.system()
    
    if system == "Linux":
        print("✓ Linux系统 - 支持RISC-V AI加速器")
        return True
    elif system == "Darwin":
        print("⚠ macOS系统 - 不支持RISC-V AI加速器硬件")
        print("  可以运行CPU基准测试和软件仿真")
        return False
    elif system == "Windows":
        print("⚠ Windows系统 - 不支持RISC-V AI加速器")
        print("  建议使用Linux系统")
        return False
    else:
        print(f"⚠ 未知系统: {system} - 兼容性未知")
        return False

def check_device_files():
    """检查设备文件"""
    print("\n=== 设备文件检查 ===")
    
    device_files = [
        "/dev/ai_accel",
        "/dev/ai_accel0",
        "/dev/ai_accel1",
        "/dev/riscv_ai",
        "/dev/tpu0",
        "/dev/tpu1"
    ]
    
    found_devices = []
    
    for device in device_files:
        if os.path.exists(device):
            try:
                stat = os.stat(device)
                print(f"✓ 找到设备: {device}")
                print(f"  权限: {oct(stat.st_mode)[-3:]}")
                print(f"  所有者: {stat.st_uid}:{stat.st_gid}")
                found_devices.append(device)
            except OSError as e:
                print(f"⚠ 设备文件存在但无法访问: {device} ({e})")
        else:
            print(f"✗ 未找到设备: {device}")
    
    if not found_devices:
        print("\n⚠ 未找到任何AI加速器设备文件")
        print("可能的原因:")
        print("  1. 硬件未连接")
        print("  2. 驱动程序未加载")
        print("  3. 设备文件路径不同")
    
    return found_devices

def check_kernel_modules():
    """检查内核模块"""
    print("\n=== 内核模块检查 ===")
    
    if platform.system() != "Linux":
        print("⚠ 非Linux系统，跳过内核模块检查")
        return []
    
    # 检查lsmod命令是否可用
    success, _, _ = run_command("which lsmod")
    if not success:
        print("⚠ lsmod命令不可用")
        return []
    
    # 获取已加载的模块
    success, output, error = run_command("lsmod")
    if not success:
        print(f"⚠ 无法获取内核模块列表: {error}")
        return []
    
    # 查找AI相关模块
    ai_modules = []
    target_modules = ["ai_accel", "riscv_ai", "tpu_driver", "vpu_driver"]
    
    lines = output.split('\n')
    for line in lines:
        for module in target_modules:
            if module in line.lower():
                ai_modules.append(line.split()[0])
                print(f"✓ 找到模块: {line}")
    
    if not ai_modules:
        print("⚠ 未找到AI加速器相关内核模块")
        print("预期的模块名称:")
        for module in target_modules:
            print(f"  - {module}")
    
    return ai_modules

def check_pcie_devices():
    """检查PCIe设备"""
    print("\n=== PCIe设备检查 ===")
    
    if platform.system() != "Linux":
        print("⚠ 非Linux系统，跳过PCIe设备检查")
        return []
    
    # 检查lspci命令是否可用
    success, _, _ = run_command("which lspci")
    if not success:
        print("⚠ lspci命令不可用，请安装pciutils")
        return []
    
    # 获取PCIe设备列表
    success, output, error = run_command("lspci")
    if not success:
        print(f"⚠ 无法获取PCIe设备列表: {error}")
        return []
    
    # 查找AI相关设备
    ai_devices = []
    keywords = ["ai", "accelerator", "riscv", "tpu", "neural", "tensor"]
    
    lines = output.split('\n')
    for line in lines:
        line_lower = line.lower()
        for keyword in keywords:
            if keyword in line_lower:
                ai_devices.append(line)
                print(f"✓ 可能的AI设备: {line}")
                break
    
    if not ai_devices:
        print("⚠ 未找到明显的AI加速器PCIe设备")
        print("所有PCIe设备:")
        for line in lines[:10]:  # 只显示前10个设备
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... 还有 {len(lines) - 10} 个设备")
    
    return ai_devices

def check_software_dependencies():
    """检查软件依赖"""
    print("\n=== 软件依赖检查 ===")
    
    dependencies = {
        "torch": "PyTorch深度学习框架",
        "numpy": "数值计算库",
        "pybind11": "Python C++绑定库"
    }
    
    available_deps = {}
    
    for dep, desc in dependencies.items():
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep} ({desc}): {version}")
            available_deps[dep] = version
        except ImportError:
            print(f"✗ {dep} ({desc}): 未安装")
            available_deps[dep] = None
    
    return available_deps

def check_pytorch_backend():
    """检查PyTorch后端"""
    print("\n=== PyTorch后端检查 ===")
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA设备数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
        
        # 尝试导入RISC-V AI后端
        try:
            sys.path.insert(0, str(Path(__file__).parent / "software" / "frameworks" / "pytorch"))
            sys.path.insert(0, str(Path(__file__).parent))
            import riscv_ai_backend
            print("✓ RISC-V AI后端: 可用")
            
            # 检查后端功能
            if hasattr(riscv_ai_backend, 'is_available'):
                available = riscv_ai_backend.is_available()
                print(f"✓ 后端状态: {'可用' if available else '不可用'}")
            
            if hasattr(riscv_ai_backend, 'device_count'):
                device_count = riscv_ai_backend.device_count()
                print(f"✓ AI设备数: {device_count}")
            
        except ImportError as e:
            print(f"⚠ RISC-V AI后端: 不可用 ({e})")
            print("  这是正常的，如果硬件不可用或后端未构建")
        
    except ImportError:
        print("✗ PyTorch未安装")

def generate_hardware_report():
    """生成硬件检查报告"""
    print("\n" + "="*60)
    print("RISC-V AI加速器硬件检查报告")
    print("="*60)
    
    # 收集所有检查结果
    system_info = check_system_info()
    compatible = check_riscv_ai_compatibility()
    devices = check_device_files()
    modules = check_kernel_modules()
    pcie_devices = check_pcie_devices()
    dependencies = check_software_dependencies()
    check_pytorch_backend()
    
    # 生成总结
    print("\n=== 检查总结 ===")
    
    if compatible and devices and modules:
        print("✅ 硬件状态: RISC-V AI加速器已正确安装和配置")
        print("✅ 建议: 可以运行完整的AI加速器测试")
    elif compatible and not devices:
        print("⚠️ 硬件状态: 系统兼容但未检测到硬件")
        print("⚠️ 建议: 检查硬件连接和驱动安装")
    elif not compatible:
        print("❌ 硬件状态: 系统不兼容RISC-V AI加速器")
        print("❌ 建议: 使用Linux系统或运行CPU基准测试")
    else:
        print("⚠️ 硬件状态: 部分功能可用")
        print("⚠️ 建议: 检查配置或联系技术支持")
    
    # 下一步建议
    print("\n=== 下一步建议 ===")
    
    if system_info["os"] == "Darwin":
        print("1. 运行CPU基准测试:")
        print("   make -f Makefile.pytorch_test test-simple")
        print("2. 在Linux系统上测试完整功能")
        print("3. 使用软件仿真进行开发")
    elif system_info["os"] == "Linux":
        if devices:
            print("1. 运行完整测试:")
            print("   make -f Makefile.pytorch_test test-comprehensive")
            print("2. 检查性能基准:")
            print("   make -f Makefile.pytorch_test benchmark")
        else:
            print("1. 检查硬件连接")
            print("2. 安装或加载驱动程序:")
            print("   sudo modprobe ai_accel")
            print("3. 检查设备权限:")
            print("   sudo chmod 666 /dev/ai_accel")
            print("4. 运行CPU基准测试:")
            print("   make -f Makefile.pytorch_test test-simple")
    
    # 保存报告
    report_data = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "system_info": system_info,
        "compatible": compatible,
        "devices_found": devices,
        "kernel_modules": modules,
        "pcie_devices": pcie_devices,
        "dependencies": dependencies
    }
    
    try:
        os.makedirs("test_results", exist_ok=True)
        with open("test_results/hardware_check_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 详细报告已保存到: test_results/hardware_check_report.json")
    except Exception as e:
        print(f"\n⚠ 无法保存报告: {e}")

def main():
    """主函数"""
    print("RISC-V AI加速器硬件检查工具")
    print("检查硬件连接、驱动程序和系统兼容性")
    print("="*60)
    
    try:
        generate_hardware_report()
    except KeyboardInterrupt:
        print("\n\n检查被用户中断")
    except Exception as e:
        print(f"\n\n检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()