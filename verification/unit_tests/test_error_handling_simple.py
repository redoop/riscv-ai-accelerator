#!/usr/bin/env python3
"""
Simple test script to verify error handling implementation
Tests the basic functionality of ECC, error detection, and recovery mechanisms
"""

import os
import sys

def test_ecc_functionality():
    """Test ECC controller basic functionality"""
    print("Testing ECC Controller...")
    
    # Check if ECC controller file exists and has required modules
    ecc_file = "../../rtl/memory/ecc_controller.sv"
    if not os.path.exists(ecc_file):
        print("FAIL: ECC controller file not found")
        return False
    
    with open(ecc_file, 'r') as f:
        content = f.read()
        
    # Check for required modules and functionality
    required_components = [
        "module ecc_controller",
        "module ecc_encoder", 
        "module ecc_decoder",
        "single_error",
        "double_error",
        "error_inject"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: ECC Controller has all required components")
    return True

def test_error_detector_functionality():
    """Test error detector basic functionality"""
    print("Testing Error Detector...")
    
    # Check if error detector file exists and has required functionality
    detector_file = "../../rtl/memory/error_detector.sv"
    if not os.path.exists(detector_file):
        print("FAIL: Error detector file not found")
        return False
    
    with open(detector_file, 'r') as f:
        content = f.read()
        
    # Check for required functionality
    required_components = [
        "module error_detector",
        "error_interrupt",
        "error_status",
        "error_severity",
        "error_log_valid",
        "error_timestamp"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: Error Detector has all required components")
    return True

def test_error_injector_functionality():
    """Test error injector basic functionality"""
    print("Testing Error Injector...")
    
    # Check if error injector file exists and has required functionality
    injector_file = "../../rtl/memory/error_injector.sv"
    if not os.path.exists(injector_file):
        print("FAIL: Error injector file not found")
        return False
    
    with open(injector_file, 'r') as f:
        content = f.read()
        
    # Check for required functionality
    required_components = [
        "module error_injector",
        "inject_enable",
        "inject_mode",
        "single_error_inject",
        "double_error_inject",
        "injection_active"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: Error Injector has all required components")
    return True

def test_checkpoint_controller_functionality():
    """Test checkpoint controller basic functionality"""
    print("Testing Checkpoint Controller...")
    
    # Check if checkpoint controller file exists and has required functionality
    checkpoint_file = "../../rtl/memory/checkpoint_controller.sv"
    if not os.path.exists(checkpoint_file):
        print("FAIL: Checkpoint controller file not found")
        return False
    
    with open(checkpoint_file, 'r') as f:
        content = f.read()
        
    # Check for required functionality
    required_components = [
        "module checkpoint_controller",
        "checkpoint_enable",
        "checkpoint_trigger",
        "recovery_trigger",
        "checkpoint_complete",
        "recovery_complete"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: Checkpoint Controller has all required components")
    return True

def test_recovery_controller_functionality():
    """Test recovery controller basic functionality"""
    print("Testing Recovery Controller...")
    
    # Check if recovery controller file exists and has required functionality
    recovery_file = "../../rtl/memory/recovery_controller.sv"
    if not os.path.exists(recovery_file):
        print("FAIL: Recovery controller file not found")
        return False
    
    with open(recovery_file, 'r') as f:
        content = f.read()
        
    # Check for required functionality
    required_components = [
        "module recovery_controller",
        "error_detected",
        "recovery_enable",
        "recovery_active",
        "recovery_success",
        "retry_count"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: Recovery Controller has all required components")
    return True

def test_fault_isolation_functionality():
    """Test fault isolation basic functionality"""
    print("Testing Fault Isolation...")
    
    # Check if fault isolation file exists and has required functionality
    isolation_file = "../../rtl/memory/fault_isolation.sv"
    if not os.path.exists(isolation_file):
        print("FAIL: Fault isolation file not found")
        return False
    
    with open(isolation_file, 'r') as f:
        content = f.read()
        
    # Check for required functionality
    required_components = [
        "module fault_isolation",
        "core_isolated",
        "tpu_isolated",
        "vpu_isolated",
        "system_health",
        "performance_level",
        "emergency_mode"
    ]
    
    for component in required_components:
        if component not in content:
            print(f"FAIL: Missing component: {component}")
            return False
    
    print("PASS: Fault Isolation has all required components")
    return True

def main():
    """Run all error handling tests"""
    print("=== Error Handling Implementation Test ===")
    print()
    
    tests = [
        test_ecc_functionality,
        test_error_detector_functionality,
        test_error_injector_functionality,
        test_checkpoint_controller_functionality,
        test_recovery_controller_functionality,
        test_fault_isolation_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())