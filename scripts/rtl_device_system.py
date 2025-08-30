#!/usr/bin/env python3
"""
RTL设备系统 - 完整的设备管理和监控系统
将RTL代码作为系统设备进行管理
"""

import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime
from simple_rtl_device import SimpleRTLDevice
import numpy as np

class RTLDeviceSystem:
    """RTL设备系统管理器"""
    
    def __init__(self):
        self.devices = {}
        self.device_count = 0
        self.system_log = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 系统配置
        self.config = {
            "max_devices": 8,
            "log_level": "INFO",
            "monitor_interval": 5.0,  # 秒
            "device_timeout": 30.0,   # 秒
        }
        
        self.log("系统初始化", "RTL设备系统已启动")
    
    def log(self, event, message, level="INFO"):
        """系统日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "event": event,
            "message": message
        }
        self.system_log.append(log_entry)
        
        # 控制台输出
        if level in ["INFO", "WARN", "ERROR"]:
            icon = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌"}[level]
            print(f"{icon} [{timestamp}] {event}: {message}")
    
    def create_device(self, device_name=None, device_type="standard"):
        """创建新设备"""
        if len(self.devices) >= self.config["max_devices"]:
            self.log("设备创建", f"已达到最大设备数限制 ({self.config['max_devices']})", "WARN")
            return None
        
        if device_name is None:
            device_name = f"rtl_chip_{self.device_count}"
        
        if device_name in self.devices:
            self.log("设备创建", f"设备 {device_name} 已存在", "WARN")
            return self.devices[device_name]
        
        try:
            device = SimpleRTLDevice(device_name)
            device_info = {
                "device": device,
                "created_at": datetime.now(),
                "type": device_type,
                "status": "active",
                "last_activity": datetime.now()
            }
            
            self.devices[device_name] = device_info
            self.device_count += 1
            
            self.log("设备创建", f"设备 {device_name} 创建成功 (类型: {device_type})")
            return device
            
        except Exception as e:
            self.log("设备创建", f"设备 {device_name} 创建失败: {e}", "ERROR")
            return None
    
    def remove_device(self, device_name):
        """移除设备"""
        if device_name not in self.devices:
            self.log("设备移除", f"设备 {device_name} 不存在", "WARN")
            return False
        
        try:
            del self.devices[device_name]
            self.log("设备移除", f"设备 {device_name} 已移除")
            return True
        except Exception as e:
            self.log("设备移除", f"移除设备 {device_name} 失败: {e}", "ERROR")
            return False
    
    def get_device(self, device_name):
        """获取设备"""
        device_info = self.devices.get(device_name)
        if device_info:
            device_info["last_activity"] = datetime.now()
            return device_info["device"]
        return None
    
    def list_devices(self):
        """列出所有设备"""
        device_list = []
        for name, info in self.devices.items():
            device_list.append({
                "name": name,
                "type": info["type"],
                "status": info["status"],
                "created_at": info["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                "last_activity": info["last_activity"].strftime("%Y-%m-%d %H:%M:%S"),
                "operations": info["device"].operation_count,
                "compute_time": f"{info['device'].total_compute_time:.4f}s"
            })
        return device_list
    
    def get_system_status(self):
        """获取系统状态"""
        total_operations = sum(info["device"].operation_count for info in self.devices.values())
        total_compute_time = sum(info["device"].total_compute_time for info in self.devices.values())
        
        status = {
            "system_info": {
                "total_devices": len(self.devices),
                "active_devices": sum(1 for info in self.devices.values() if info["status"] == "active"),
                "total_operations": total_operations,
                "total_compute_time": f"{total_compute_time:.4f}s",
                "uptime": self._get_uptime(),
                "log_entries": len(self.system_log)
            },
            "devices": self.list_devices(),
            "recent_logs": self.system_log[-10:]  # 最近10条日志
        }
        
        return status
    
    def _get_uptime(self):
        """获取系统运行时间"""
        if self.system_log:
            start_time = datetime.strptime(self.system_log[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
            uptime = datetime.now() - start_time
            return str(uptime).split('.')[0]  # 去掉微秒
        return "0:00:00"
    
    def start_monitoring(self):
        """启动系统监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.log("系统监控", "监控服务已启动")
    
    def stop_monitoring(self):
        """停止系统监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.log("系统监控", "监控服务已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检查设备状态
                current_time = datetime.now()
                for name, info in list(self.devices.items()):
                    time_since_activity = (current_time - info["last_activity"]).total_seconds()
                    
                    if time_since_activity > self.config["device_timeout"]:
                        info["status"] = "inactive"
                        self.log("设备监控", f"设备 {name} 超时未活动", "WARN")
                    elif info["status"] == "inactive" and time_since_activity < self.config["device_timeout"]:
                        info["status"] = "active"
                        self.log("设备监控", f"设备 {name} 恢复活动")
                
                # 系统资源检查
                self._check_system_resources()
                
                time.sleep(self.config["monitor_interval"])
                
            except Exception as e:
                self.log("系统监控", f"监控异常: {e}", "ERROR")
                time.sleep(1.0)
    
    def _check_system_resources(self):
        """检查系统资源"""
        try:
            # 检查内存使用
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.log("资源监控", f"内存使用率过高: {memory.percent:.1f}%", "WARN")
            
            # 检查CPU使用
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.log("资源监控", f"CPU使用率过高: {cpu_percent:.1f}%", "WARN")
                
        except ImportError:
            # psutil不可用时跳过资源检查
            pass
        except Exception as e:
            self.log("资源监控", f"资源检查失败: {e}", "ERROR")
    
    def run_system_benchmark(self):
        """运行系统基准测试"""
        self.log("系统测试", "开始系统基准测试")
        
        # 创建测试设备
        test_device = self.create_device("benchmark_device", "benchmark")
        if not test_device:
            self.log("系统测试", "无法创建测试设备", "ERROR")
            return None
        
        try:
            # 运行基准测试
            results = test_device.benchmark_test([32, 64, 128])
            
            # 运行神经网络测试
            nn_results = test_device.neural_network_demo()
            
            benchmark_summary = {
                "matrix_multiply": results,
                "neural_network": {
                    "input_shape": nn_results["input_shape"],
                    "output_shape": nn_results["output_shape"],
                    "batch_size": nn_results["input_shape"][0]
                },
                "device_info": test_device.get_device_info()
            }
            
            self.log("系统测试", "系统基准测试完成")
            
            # 移除测试设备
            self.remove_device("benchmark_device")
            
            return benchmark_summary
            
        except Exception as e:
            self.log("系统测试", f"基准测试失败: {e}", "ERROR")
            self.remove_device("benchmark_device")
            return None
    
    def export_system_report(self, filename=None):
        """导出系统报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rtl_system_report_{timestamp}.json"
        
        try:
            report = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "system_version": "1.0.0",
                    "report_type": "RTL Device System Report"
                },
                "system_status": self.get_system_status(),
                "configuration": self.config,
                "full_log": self.system_log
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.log("报告导出", f"系统报告已导出: {filename}")
            return filename
            
        except Exception as e:
            self.log("报告导出", f"导出报告失败: {e}", "ERROR")
            return None

def interactive_demo():
    """交互式演示"""
    print("🔧 RTL设备系统交互式演示")
    print("=" * 50)
    
    # 创建系统
    system = RTLDeviceSystem()
    system.start_monitoring()
    
    try:
        while True:
            print("\n📋 可用命令:")
            print("  1. 创建设备")
            print("  2. 列出设备")
            print("  3. 设备操作")
            print("  4. 系统状态")
            print("  5. 运行基准测试")
            print("  6. 导出报告")
            print("  7. 退出")
            
            choice = input("\n请选择操作 (1-7): ").strip()
            
            if choice == "1":
                device_name = input("设备名称 (回车使用默认): ").strip()
                device_name = device_name if device_name else None
                device = system.create_device(device_name)
                if device:
                    print(f"✅ 设备创建成功: {device.device_name}")
            
            elif choice == "2":
                devices = system.list_devices()
                if devices:
                    print("\n📱 设备列表:")
                    for device in devices:
                        print(f"  {device['name']}: {device['status']} "
                              f"({device['operations']} ops, {device['compute_time']})")
                else:
                    print("📱 暂无设备")
            
            elif choice == "3":
                device_name = input("设备名称: ").strip()
                device = system.get_device(device_name)
                if device:
                    print("🧮 执行矩阵乘法测试...")
                    a = np.random.randn(32, 32).astype(np.float32)
                    b = np.random.randn(32, 32).astype(np.float32)
                    result = device.matrix_multiply(a, b)
                    print(f"✅ 计算完成，结果形状: {result.shape}")
                else:
                    print(f"❌ 设备 {device_name} 不存在")
            
            elif choice == "4":
                status = system.get_system_status()
                print("\n🖥️ 系统状态:")
                for key, value in status["system_info"].items():
                    print(f"  {key}: {value}")
            
            elif choice == "5":
                print("🚀 运行系统基准测试...")
                results = system.run_system_benchmark()
                if results:
                    print("✅ 基准测试完成")
                    for size, result in results["matrix_multiply"].items():
                        print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS")
            
            elif choice == "6":
                filename = system.export_system_report()
                if filename:
                    print(f"✅ 报告已导出: {filename}")
            
            elif choice == "7":
                break
            
            else:
                print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    
    finally:
        system.stop_monitoring()
        print("🔧 系统已关闭")

def main():
    """主程序"""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        # 自动演示
        print("🔧 RTL设备系统自动演示")
        print("=" * 40)
        
        # 创建系统
        system = RTLDeviceSystem()
        system.start_monitoring()
        
        try:
            # 创建多个设备
            print("\n1️⃣ 创建设备...")
            device1 = system.create_device("ai_chip_0", "compute")
            device2 = system.create_device("ai_chip_1", "inference")
            
            # 设备操作
            print("\n2️⃣ 设备操作...")
            if device1:
                a = np.random.randn(64, 64).astype(np.float32)
                b = np.random.randn(64, 64).astype(np.float32)
                result = device1.matrix_multiply(a, b)
                print(f"设备1计算结果形状: {result.shape}")
            
            if device2:
                x = np.random.randn(100).astype(np.float32)
                relu_result = device2.relu_activation(x)
                print(f"设备2 ReLU结果: {np.sum(relu_result > 0)} 个正值")
            
            # 系统状态
            print("\n3️⃣ 系统状态:")
            status = system.get_system_status()
            for key, value in status["system_info"].items():
                print(f"  {key}: {value}")
            
            # 基准测试
            print("\n4️⃣ 系统基准测试:")
            benchmark_results = system.run_system_benchmark()
            if benchmark_results:
                print("基准测试结果:")
                for size, result in benchmark_results["matrix_multiply"].items():
                    print(f"  {size}x{size}: {result['gflops']:.2f} GFLOPS")
            
            # 导出报告
            print("\n5️⃣ 导出系统报告:")
            report_file = system.export_system_report()
            if report_file:
                print(f"报告文件: {report_file}")
            
            print("\n🎉 RTL设备系统演示完成!")
            
        finally:
            system.stop_monitoring()

if __name__ == "__main__":
    main()