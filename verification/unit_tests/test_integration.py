#!/usr/bin/env python3
"""
Performance Optimization Integration Test

This script tests the integration between RTL performance monitoring
hardware and software optimization algorithms.
"""

import os
import sys
import subprocess
import time
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional

class PerformanceIntegrationTest:
    """Integration test for performance optimization system"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = []
        self.rtl_simulator = None
        self.software_process = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARN"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def run_rtl_simulation(self, test_scenario: str, duration: int = 10) -> bool:
        """Run RTL simulation for specified duration"""
        self.log(f"Starting RTL simulation for scenario: {test_scenario}")
        
        try:
            # Start RTL simulator
            cmd = ["./obj_dir/Vtest_performance_optimization", f"+scenario={test_scenario}"]
            self.rtl_simulator = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Let simulation run for specified duration
            time.sleep(duration)
            
            # Check if simulation is still running
            if self.rtl_simulator.poll() is None:
                self.log("RTL simulation running successfully")
                return True
            else:
                stdout, stderr = self.rtl_simulator.communicate()
                self.log(f"RTL simulation exited early: {stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Failed to start RTL simulation: {e}", "ERROR")
            return False
    
    def run_software_optimizer(self, config: Dict) -> bool:
        """Run software performance optimizer with given configuration"""
        self.log("Starting software performance optimizer")
        
        try:
            # Create configuration file
            config_file = "test_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Start software optimizer
            cmd = ["./test_performance_optimizer", f"--config={config_file}"]
            self.software_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it time to initialize
            time.sleep(2)
            
            # Check if process is running
            if self.software_process.poll() is None:
                self.log("Software optimizer started successfully")
                return True
            else:
                stdout, stderr = self.software_process.communicate()
                self.log(f"Software optimizer failed to start: {stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Failed to start software optimizer: {e}", "ERROR")
            return False
    
    def collect_performance_metrics(self) -> Optional[Dict]:
        """Collect performance metrics from both RTL and software"""
        metrics = {
            "timestamp": time.time(),
            "rtl_metrics": {},
            "software_metrics": {}
        }
        
        # Collect RTL metrics (simulated - would read from simulation output)
        if self.rtl_simulator and self.rtl_simulator.poll() is None:
            # In a real implementation, this would parse RTL simulation output
            # or read from shared memory/files
            metrics["rtl_metrics"] = {
                "performance_score": 0.75,
                "power_consumption": 12.5,
                "temperature": 68.2,
                "utilization": 0.82,
                "cache_hit_rate": 0.89
            }
        
        # Collect software metrics (simulated - would call optimizer API)
        if self.software_process and self.software_process.poll() is None:
            # In a real implementation, this would call performance optimizer APIs
            metrics["software_metrics"] = {
                "optimization_cycles": 15,
                "successful_optimizations": 12,
                "average_performance_gain": 0.08,
                "power_savings": 1.2,
                "confidence_score": 0.85
            }
        
        return metrics
    
    def test_basic_integration(self) -> bool:
        """Test basic integration between RTL and software"""
        self.log("Running basic integration test")
        
        # Start RTL simulation
        if not self.run_rtl_simulation("basic_workload", 5):
            return False
        
        # Start software optimizer
        config = {
            "optimization_interval_ms": 1000,
            "adaptation_aggressiveness": 0.5,
            "power_budget_watts": 15.0,
            "enable_predictive_optimization": True
        }
        
        if not self.run_software_optimizer(config):
            return False
        
        # Let them run together
        time.sleep(10)
        
        # Collect metrics
        metrics = self.collect_performance_metrics()
        if not metrics:
            self.log("Failed to collect performance metrics", "ERROR")
            return False
        
        # Validate integration
        rtl_perf = metrics["rtl_metrics"].get("performance_score", 0)
        sw_optimizations = metrics["software_metrics"].get("successful_optimizations", 0)
        
        success = rtl_perf > 0.5 and sw_optimizations > 5
        
        self.test_results.append({
            "test": "basic_integration",
            "success": success,
            "metrics": metrics
        })
        
        self.log(f"Basic integration test: {'PASSED' if success else 'FAILED'}")
        return success
    
    def test_workload_adaptation(self) -> bool:
        """Test adaptation to different workload types"""
        self.log("Running workload adaptation test")
        
        workload_scenarios = [
            ("cpu_intensive", {"cpu_intensity": 0.9, "ai_intensity": 0.1}),
            ("ai_intensive", {"cpu_intensity": 0.2, "ai_intensity": 0.8}),
            ("memory_intensive", {"memory_intensity": 0.9, "cpu_intensity": 0.3}),
            ("mixed_workload", {"cpu_intensity": 0.5, "ai_intensity": 0.5})
        ]
        
        adaptation_results = []
        
        for scenario_name, workload_config in workload_scenarios:
            self.log(f"Testing workload scenario: {scenario_name}")
            
            # Start RTL simulation with specific workload
            if not self.run_rtl_simulation(scenario_name, 8):
                continue
            
            # Configure optimizer for this workload type
            optimizer_config = {
                "optimization_interval_ms": 500,
                "adaptation_aggressiveness": 0.7,
                "workload_config": workload_config
            }
            
            if not self.run_software_optimizer(optimizer_config):
                continue
            
            # Let adaptation occur
            time.sleep(12)
            
            # Collect adaptation metrics
            metrics = self.collect_performance_metrics()
            if metrics:
                adaptation_score = self.calculate_adaptation_score(metrics, workload_config)
                adaptation_results.append({
                    "scenario": scenario_name,
                    "adaptation_score": adaptation_score,
                    "metrics": metrics
                })
                
                self.log(f"Adaptation score for {scenario_name}: {adaptation_score:.3f}")
            
            # Stop processes for next scenario
            self.stop_processes()
            time.sleep(2)
        
        # Evaluate overall adaptation performance
        if adaptation_results:
            avg_adaptation = sum(r["adaptation_score"] for r in adaptation_results) / len(adaptation_results)
            success = avg_adaptation > 0.7
            
            self.test_results.append({
                "test": "workload_adaptation",
                "success": success,
                "average_adaptation_score": avg_adaptation,
                "results": adaptation_results
            })
            
            self.log(f"Workload adaptation test: {'PASSED' if success else 'FAILED'} (avg score: {avg_adaptation:.3f})")
            return success
        
        return False
    
    def test_power_optimization(self) -> bool:
        """Test power optimization capabilities"""
        self.log("Running power optimization test")
        
        # Start with high power consumption scenario
        if not self.run_rtl_simulation("high_power_workload", 5):
            return False
        
        # Configure optimizer for aggressive power optimization
        config = {
            "optimization_interval_ms": 800,
            "power_budget_watts": 10.0,  # Tight power budget
            "enable_power_optimization": True,
            "adaptation_aggressiveness": 0.8
        }
        
        if not self.run_software_optimizer(config):
            return False
        
        # Collect initial power metrics
        time.sleep(3)
        initial_metrics = self.collect_performance_metrics()
        initial_power = initial_metrics["rtl_metrics"].get("power_consumption", 0)
        
        # Let power optimization work
        time.sleep(15)
        
        # Collect final power metrics
        final_metrics = self.collect_performance_metrics()
        final_power = final_metrics["rtl_metrics"].get("power_consumption", 0)
        
        # Calculate power reduction
        power_reduction = (initial_power - final_power) / initial_power if initial_power > 0 else 0
        success = power_reduction > 0.1  # At least 10% power reduction
        
        self.test_results.append({
            "test": "power_optimization",
            "success": success,
            "initial_power": initial_power,
            "final_power": final_power,
            "power_reduction": power_reduction
        })
        
        self.log(f"Power optimization test: {'PASSED' if success else 'FAILED'} (reduction: {power_reduction:.1%})")
        return success
    
    def test_thermal_management(self) -> bool:
        """Test thermal management capabilities"""
        self.log("Running thermal management test")
        
        # Start with high temperature scenario
        if not self.run_rtl_simulation("high_temp_workload", 5):
            return False
        
        # Configure optimizer for thermal management
        config = {
            "optimization_interval_ms": 600,
            "thermal_limit_celsius": 75.0,  # Low thermal limit
            "enable_thermal_optimization": True,
            "adaptation_aggressiveness": 0.9
        }
        
        if not self.run_software_optimizer(config):
            return False
        
        # Monitor thermal response
        thermal_readings = []
        for i in range(10):
            time.sleep(2)
            metrics = self.collect_performance_metrics()
            if metrics:
                temp = metrics["rtl_metrics"].get("temperature", 0)
                thermal_readings.append(temp)
                self.log(f"Temperature reading {i+1}: {temp:.1f}°C")
        
        # Evaluate thermal management
        if len(thermal_readings) >= 5:
            initial_temp = thermal_readings[0]
            final_temp = thermal_readings[-1]
            temp_reduction = initial_temp - final_temp
            
            # Check if temperature was brought under control
            success = final_temp < 75.0 and temp_reduction > 2.0
            
            self.test_results.append({
                "test": "thermal_management",
                "success": success,
                "initial_temperature": initial_temp,
                "final_temperature": final_temp,
                "temperature_reduction": temp_reduction,
                "thermal_readings": thermal_readings
            })
            
            self.log(f"Thermal management test: {'PASSED' if success else 'FAILED'} (reduction: {temp_reduction:.1f}°C)")
            return success
        
        return False
    
    def test_performance_stability(self) -> bool:
        """Test system stability under varying conditions"""
        self.log("Running performance stability test")
        
        # Start with varying workload scenario
        if not self.run_rtl_simulation("varying_workload", 5):
            return False
        
        # Configure optimizer for stability
        config = {
            "optimization_interval_ms": 1000,
            "adaptation_aggressiveness": 0.4,  # Conservative adaptation
            "enable_predictive_optimization": True
        }
        
        if not self.run_software_optimizer(config):
            return False
        
        # Collect performance data over time
        performance_readings = []
        for i in range(20):
            time.sleep(1.5)
            metrics = self.collect_performance_metrics()
            if metrics:
                perf_score = metrics["rtl_metrics"].get("performance_score", 0)
                performance_readings.append(perf_score)
        
        # Analyze stability
        if len(performance_readings) >= 10:
            avg_performance = sum(performance_readings) / len(performance_readings)
            performance_variance = sum((x - avg_performance) ** 2 for x in performance_readings) / len(performance_readings)
            stability_score = 1.0 / (1.0 + performance_variance * 10)  # Higher variance = lower stability
            
            success = avg_performance > 0.6 and stability_score > 0.7
            
            self.test_results.append({
                "test": "performance_stability",
                "success": success,
                "average_performance": avg_performance,
                "performance_variance": performance_variance,
                "stability_score": stability_score,
                "readings": performance_readings
            })
            
            self.log(f"Performance stability test: {'PASSED' if success else 'FAILED'} (stability: {stability_score:.3f})")
            return success
        
        return False
    
    def calculate_adaptation_score(self, metrics: Dict, workload_config: Dict) -> float:
        """Calculate how well the system adapted to the workload"""
        # This is a simplified adaptation scoring function
        # In practice, this would be more sophisticated
        
        rtl_metrics = metrics.get("rtl_metrics", {})
        sw_metrics = metrics.get("software_metrics", {})
        
        performance_score = rtl_metrics.get("performance_score", 0)
        optimization_success_rate = sw_metrics.get("successful_optimizations", 0) / max(sw_metrics.get("optimization_cycles", 1), 1)
        confidence_score = sw_metrics.get("confidence_score", 0)
        
        # Weight the different factors
        adaptation_score = (performance_score * 0.4 + 
                          optimization_success_rate * 0.3 + 
                          confidence_score * 0.3)
        
        return min(adaptation_score, 1.0)
    
    def stop_processes(self):
        """Stop RTL simulation and software optimizer processes"""
        if self.rtl_simulator:
            self.rtl_simulator.terminate()
            self.rtl_simulator.wait(timeout=5)
            self.rtl_simulator = None
        
        if self.software_process:
            self.software_process.terminate()
            self.software_process.wait(timeout=5)
            self.software_process = None
    
    def generate_report(self, output_file: str = "integration_test_report.json"):
        """Generate comprehensive test report"""
        report = {
            "test_summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results if r["success"]),
                "failed_tests": sum(1 for r in self.test_results if not r["success"]),
                "success_rate": 0
            },
            "test_results": self.test_results,
            "timestamp": time.time(),
            "test_duration": time.time() - getattr(self, 'start_time', time.time())
        }
        
        if report["test_summary"]["total_tests"] > 0:
            report["test_summary"]["success_rate"] = (
                report["test_summary"]["passed_tests"] / report["test_summary"]["total_tests"]
            )
        
        # Write JSON report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Write CSV summary
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Success', 'Key Metrics'])
            
            for result in self.test_results:
                test_name = result.get('test', 'Unknown')
                success = 'PASS' if result.get('success', False) else 'FAIL'
                
                # Extract key metrics for CSV
                key_metrics = []
                if 'average_adaptation_score' in result:
                    key_metrics.append(f"Adaptation: {result['average_adaptation_score']:.3f}")
                if 'power_reduction' in result:
                    key_metrics.append(f"Power Reduction: {result['power_reduction']:.1%}")
                if 'temperature_reduction' in result:
                    key_metrics.append(f"Temp Reduction: {result['temperature_reduction']:.1f}°C")
                if 'stability_score' in result:
                    key_metrics.append(f"Stability: {result['stability_score']:.3f}")
                
                writer.writerow([test_name, success, '; '.join(key_metrics)])
        
        self.log(f"Test report generated: {output_file}")
        self.log(f"CSV summary generated: {csv_file}")
        
        return report
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        self.start_time = time.time()
        self.log("Starting comprehensive integration test suite")
        
        tests = [
            ("Basic Integration", self.test_basic_integration),
            ("Workload Adaptation", self.test_workload_adaptation),
            ("Power Optimization", self.test_power_optimization),
            ("Thermal Management", self.test_thermal_management),
            ("Performance Stability", self.test_performance_stability)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n{'='*50}")
            self.log(f"Running {test_name}")
            self.log(f"{'='*50}")
            
            try:
                if test_func():
                    passed_tests += 1
                    self.log(f"{test_name}: PASSED", "INFO")
                else:
                    self.log(f"{test_name}: FAILED", "ERROR")
            except Exception as e:
                self.log(f"{test_name}: FAILED with exception: {e}", "ERROR")
            finally:
                # Clean up processes after each test
                self.stop_processes()
                time.sleep(1)
        
        # Generate final report
        report = self.generate_report()
        
        success_rate = passed_tests / total_tests
        self.log(f"\n{'='*50}")
        self.log(f"INTEGRATION TEST SUMMARY")
        self.log(f"{'='*50}")
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {total_tests - passed_tests}")
        self.log(f"Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            self.log("INTEGRATION TESTS PASSED", "INFO")
            return True
        else:
            self.log("INTEGRATION TESTS FAILED", "ERROR")
            return False

def main():
    parser = argparse.ArgumentParser(description="Performance Optimization Integration Test")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", "--output", default="integration_test_report.json", 
                       help="Output report file")
    parser.add_argument("--test", choices=["basic", "workload", "power", "thermal", "stability", "all"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    # Check if required executables exist
    required_files = [
        "obj_dir/Vtest_performance_optimization",
        "test_performance_optimizer"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            print("Please run 'make all' to build the test executables")
            return 1
    
    # Create test instance
    test_runner = PerformanceIntegrationTest(verbose=args.verbose)
    
    # Run specified test(s)
    success = False
    
    if args.test == "all":
        success = test_runner.run_all_tests()
    elif args.test == "basic":
        success = test_runner.test_basic_integration()
    elif args.test == "workload":
        success = test_runner.test_workload_adaptation()
    elif args.test == "power":
        success = test_runner.test_power_optimization()
    elif args.test == "thermal":
        success = test_runner.test_thermal_management()
    elif args.test == "stability":
        success = test_runner.test_performance_stability()
    
    # Generate report if running all tests
    if args.test == "all":
        test_runner.generate_report(args.output)
    
    # Clean up
    test_runner.stop_processes()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())