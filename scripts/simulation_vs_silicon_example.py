#!/usr/bin/env python3
"""
RTL仿真 vs 芯片测试对比示例
展示设计验证流程中的不同阶段
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TestVector:
    """测试向量"""
    input_a: int
    input_b: int
    input_c: int
    expected_output: int
    test_name: str

@dataclass
class SimulationResult:
    """仿真结果"""
    output: int
    timing: float  # 仿真时间 (ps)
    power: float   # 功耗估算 (mW)
    test_name: str

@dataclass
class SiliconResult:
    """芯片测试结果"""
    output: int
    timing: float  # 实际延迟 (ps)
    power: float   # 实际功耗 (mW)
    temperature: float  # 测试温度 (°C)
    voltage: float      # 测试电压 (V)
    test_name: str

class RTLSimulator:
    """RTL仿真器 (模拟我们的RTL代码)"""
    
    def __init__(self):
        self.name = "RTL Simulator"
        self.ideal_conditions = True
        
    def simulate_mac_operation(self, test_vector: TestVector) -> SimulationResult:
        """模拟MAC操作 (A * B + C)"""
        # 理想的RTL仿真 - 精确计算
        result = test_vector.input_a * test_vector.input_b + test_vector.input_c
        
        # 理想时序 (基于RTL设计)
        timing = 1000.0  # 1ns (理想情况)
        
        # 理想功耗估算
        power = 10.0  # 10mW (估算值)
        
        return SimulationResult(
            output=result,
            timing=timing,
            power=power,
            test_name=test_vector.test_name
        )

class SiliconChip:
    """物理芯片 (模拟真实芯片的行为)"""
    
    def __init__(self, process_corner="typical", temperature=25.0, voltage=1.0):
        self.name = "Physical Silicon Chip"
        self.process_corner = process_corner  # fast, typical, slow
        self.temperature = temperature
        self.voltage = voltage
        
        # 工艺偏差影响
        self.process_variation = self._get_process_variation()
        
    def _get_process_variation(self):
        """获取工艺偏差参数"""
        variations = {
            "fast": {"timing_factor": 0.8, "power_factor": 1.2},
            "typical": {"timing_factor": 1.0, "power_factor": 1.0},
            "slow": {"timing_factor": 1.3, "power_factor": 0.9}
        }
        return variations.get(self.process_corner, variations["typical"])
    
    def test_mac_operation(self, test_vector: TestVector) -> SiliconResult:
        """测试物理芯片的MAC操作"""
        # 功能应该相同 (如果设计正确)
        result = test_vector.input_a * test_vector.input_b + test_vector.input_c
        
        # 真实时序 - 受工艺、电压、温度影响
        base_timing = 1000.0  # 基准时序
        
        # PVT (Process, Voltage, Temperature) 影响
        timing_factor = self.process_variation["timing_factor"]
        
        # 温度影响 (温度越高，速度越慢)
        temp_factor = 1.0 + (self.temperature - 25) * 0.002
        
        # 电压影响 (电压越低，速度越慢)
        voltage_factor = (self.voltage / 1.0) ** 2
        
        actual_timing = base_timing * timing_factor * temp_factor / voltage_factor
        
        # 真实功耗
        base_power = 10.0
        power_factor = self.process_variation["power_factor"]
        
        # 功耗与电压平方成正比，与频率成正比
        freq_factor = 1000.0 / actual_timing  # 频率
        actual_power = base_power * power_factor * (self.voltage ** 2) * freq_factor
        
        # 添加一些测量噪声
        noise = np.random.normal(0, 0.01)  # 1% 噪声
        actual_power *= (1 + noise)
        actual_timing *= (1 + noise * 0.5)
        
        return SiliconResult(
            output=result,
            timing=actual_timing,
            power=actual_power,
            temperature=self.temperature,
            voltage=self.voltage,
            test_name=test_vector.test_name
        )

class VerificationFramework:
    """验证框架 - 对比RTL仿真和芯片测试结果"""
    
    def __init__(self):
        self.rtl_sim = RTLSimulator()
        self.test_vectors = self._generate_test_vectors()
        
    def _generate_test_vectors(self) -> List[TestVector]:
        """生成测试向量"""
        vectors = [
            TestVector(10, 20, 5, 205, "基本MAC测试"),
            TestVector(7, 8, 100, 156, "中等数值测试"),
            TestVector(15, 4, 25, 85, "小数值测试"),
            TestVector(0, 999, 42, 42, "零乘法测试"),
            TestVector(255, 255, 0, 65025, "最大值测试"),
            TestVector(-10, 5, 20, -30, "负数测试"),
            TestVector(100, 100, -500, 9500, "大数值测试"),
        ]
        return vectors
    
    def run_rtl_verification(self) -> List[SimulationResult]:
        """运行RTL验证"""
        print("🔬 运行RTL仿真验证...")
        print("=" * 50)
        
        results = []
        for vector in self.test_vectors:
            result = self.rtl_sim.simulate_mac_operation(vector)
            results.append(result)
            
            print(f"✅ {vector.test_name}:")
            print(f"   输入: {vector.input_a} * {vector.input_b} + {vector.input_c}")
            print(f"   输出: {result.output}")
            print(f"   时序: {result.timing:.1f} ps")
            print(f"   功耗: {result.power:.1f} mW")
            print()
        
        return results
    
    def run_silicon_verification(self, conditions: List[Dict[str, Any]]) -> Dict[str, List[SiliconResult]]:
        """运行芯片验证 (多种条件)"""
        print("🔌 运行芯片测试验证...")
        print("=" * 50)
        
        all_results = {}
        
        for condition in conditions:
            print(f"📊 测试条件: {condition['name']}")
            
            chip = SiliconChip(
                process_corner=condition['process'],
                temperature=condition['temperature'],
                voltage=condition['voltage']
            )
            
            results = []
            for vector in self.test_vectors:
                result = chip.test_mac_operation(vector)
                results.append(result)
                
                print(f"   {vector.test_name}: "
                      f"输出={result.output}, "
                      f"时序={result.timing:.1f}ps, "
                      f"功耗={result.power:.1f}mW")
            
            all_results[condition['name']] = results
            print()
        
        return all_results
    
    def compare_results(self, rtl_results: List[SimulationResult], 
                       silicon_results: Dict[str, List[SiliconResult]]):
        """对比RTL仿真和芯片测试结果"""
        print("⚖️ RTL仿真 vs 芯片测试对比分析")
        print("=" * 60)
        
        # 功能正确性检查
        print("🎯 功能正确性验证:")
        for i, rtl_result in enumerate(rtl_results):
            test_name = rtl_result.test_name
            rtl_output = rtl_result.output
            
            print(f"\n📋 {test_name}:")
            print(f"   RTL仿真输出: {rtl_output}")
            
            all_match = True
            for condition_name, silicon_list in silicon_results.items():
                silicon_output = silicon_list[i].output
                match = (rtl_output == silicon_output)
                all_match &= match
                
                status = "✅ PASS" if match else "❌ FAIL"
                print(f"   {condition_name}: {silicon_output} {status}")
            
            overall_status = "✅ 功能正确" if all_match else "❌ 功能错误"
            print(f"   总体: {overall_status}")
        
        # 性能对比分析
        print(f"\n📊 性能对比分析:")
        self._analyze_performance(rtl_results, silicon_results)
        
        # 功耗对比分析
        print(f"\n⚡ 功耗对比分析:")
        self._analyze_power(rtl_results, silicon_results)
    
    def _analyze_performance(self, rtl_results: List[SimulationResult], 
                           silicon_results: Dict[str, List[SiliconResult]]):
        """性能分析"""
        rtl_avg_timing = np.mean([r.timing for r in rtl_results])
        
        print(f"RTL仿真平均时序: {rtl_avg_timing:.1f} ps")
        
        for condition_name, silicon_list in silicon_results.items():
            silicon_avg_timing = np.mean([r.timing for r in silicon_list])
            deviation = ((silicon_avg_timing - rtl_avg_timing) / rtl_avg_timing) * 100
            
            print(f"{condition_name}: {silicon_avg_timing:.1f} ps "
                  f"(偏差: {deviation:+.1f}%)")
    
    def _analyze_power(self, rtl_results: List[SimulationResult], 
                      silicon_results: Dict[str, List[SiliconResult]]):
        """功耗分析"""
        rtl_avg_power = np.mean([r.power for r in rtl_results])
        
        print(f"RTL仿真平均功耗: {rtl_avg_power:.1f} mW")
        
        for condition_name, silicon_list in silicon_results.items():
            silicon_avg_power = np.mean([r.power for r in silicon_list])
            deviation = ((silicon_avg_power - rtl_avg_power) / rtl_avg_power) * 100
            
            print(f"{condition_name}: {silicon_avg_power:.1f} mW "
                  f"(偏差: {deviation:+.1f}%)")
    
    def generate_correlation_report(self, rtl_results: List[SimulationResult], 
                                  silicon_results: Dict[str, List[SiliconResult]]):
        """生成相关性报告"""
        print(f"\n📈 RTL仿真与芯片测试相关性报告")
        print("=" * 60)
        
        # 功能相关性 (应该是100%)
        functional_correlation = self._calculate_functional_correlation(rtl_results, silicon_results)
        print(f"功能相关性: {functional_correlation:.1f}% (目标: 100%)")
        
        # 时序相关性
        timing_correlation = self._calculate_timing_correlation(rtl_results, silicon_results)
        print(f"时序相关性: {timing_correlation:.1f}% (典型: 80-95%)")
        
        # 功耗相关性
        power_correlation = self._calculate_power_correlation(rtl_results, silicon_results)
        print(f"功耗相关性: {power_correlation:.1f}% (典型: 70-90%)")
        
        print(f"\n💡 分析结论:")
        if functional_correlation == 100:
            print("✅ 功能验证通过 - RTL设计正确")
        else:
            print("❌ 功能验证失败 - 需要修复RTL设计")
        
        if timing_correlation > 80:
            print("✅ 时序预测准确 - 可以信赖RTL时序分析")
        else:
            print("⚠️ 时序预测偏差较大 - 需要改进时序模型")
        
        if power_correlation > 70:
            print("✅ 功耗预测合理 - 可以用于功耗优化")
        else:
            print("⚠️ 功耗预测偏差较大 - 需要改进功耗模型")
    
    def _calculate_functional_correlation(self, rtl_results, silicon_results):
        """计算功能相关性"""
        total_tests = len(rtl_results) * len(silicon_results)
        correct_tests = 0
        
        for i, rtl_result in enumerate(rtl_results):
            for silicon_list in silicon_results.values():
                if rtl_result.output == silicon_list[i].output:
                    correct_tests += 1
        
        return (correct_tests / total_tests) * 100
    
    def _calculate_timing_correlation(self, rtl_results, silicon_results):
        """计算时序相关性 (简化的相关系数)"""
        # 这里使用简化的计算方法
        rtl_timings = [r.timing for r in rtl_results]
        
        correlations = []
        for silicon_list in silicon_results.values():
            silicon_timings = [r.timing for r in silicon_list]
            # 计算相对变化的相关性
            corr = np.corrcoef(rtl_timings, silicon_timings)[0, 1]
            correlations.append(abs(corr) * 100)
        
        return np.mean(correlations)
    
    def _calculate_power_correlation(self, rtl_results, silicon_results):
        """计算功耗相关性"""
        rtl_powers = [r.power for r in rtl_results]
        
        correlations = []
        for silicon_list in silicon_results.values():
            silicon_powers = [r.power for r in silicon_list]
            corr = np.corrcoef(rtl_powers, silicon_powers)[0, 1]
            correlations.append(abs(corr) * 100)
        
        return np.mean(correlations)

def main():
    """主程序 - 演示RTL仿真与芯片测试的对比"""
    print("🔬 RTL仿真 vs 芯片测试对比演示")
    print("=" * 60)
    print("这个演示展示了RTL波形图在芯片设计验证中的作用")
    print("RTL仿真用于设计验证，芯片测试用于制造验证")
    print()
    
    # 创建验证框架
    framework = VerificationFramework()
    
    # 运行RTL验证
    rtl_results = framework.run_rtl_verification()
    
    # 定义芯片测试条件 (不同的PVT条件)
    test_conditions = [
        {
            "name": "典型条件 (TT, 25°C, 1.0V)",
            "process": "typical",
            "temperature": 25.0,
            "voltage": 1.0
        },
        {
            "name": "快速条件 (FF, 0°C, 1.1V)",
            "process": "fast", 
            "temperature": 0.0,
            "voltage": 1.1
        },
        {
            "name": "慢速条件 (SS, 85°C, 0.9V)",
            "process": "slow",
            "temperature": 85.0,
            "voltage": 0.9
        }
    ]
    
    # 运行芯片测试
    silicon_results = framework.run_silicon_verification(test_conditions)
    
    # 对比分析
    framework.compare_results(rtl_results, silicon_results)
    
    # 生成相关性报告
    framework.generate_correlation_report(rtl_results, silicon_results)
    
    print(f"\n🎯 总结:")
    print("1. RTL波形图用于设计阶段的功能和时序验证")
    print("2. 芯片测试验证RTL设计在真实硅片上的实现")
    print("3. 两者的对比帮助改进设计和验证方法")
    print("4. 高相关性说明RTL模型准确，可以信赖仿真结果")
    print()
    print("🌊 我们创建的JavaScript波形查看器就是RTL验证阶段的重要工具!")

if __name__ == "__main__":
    main()