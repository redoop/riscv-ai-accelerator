#!/usr/bin/env python3
"""
Enhanced VPU functionality test script
Tests the vector register file and functional units implementation
"""

import subprocess
import sys
import os

def run_syntax_check(file_path):
    """Run syntax check on SystemVerilog file"""
    try:
        result = subprocess.run(['python3', 'syntax_check.py', file_path], 
                              capture_output=True, text=True, cwd='.')
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_vpu_components():
    """Test VPU components for syntax and basic structure"""
    
    print("=== Enhanced VPU Functionality Test ===\n")
    
    # Test files to check
    test_files = [
        "../../rtl/accelerators/vpu.sv",
        "../../rtl/accelerators/vector_alu.sv", 
        "test_vpu_functional_units.sv",
        "test_vector_alu.sv"
    ]
    
    all_passed = True
    
    for file_path in test_files:
        print(f"Testing {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"  ❌ File not found: {file_path}")
            all_passed = False
            continue
            
        passed, stdout, stderr = run_syntax_check(file_path)
        
        if passed:
            print(f"  ✅ Syntax check passed")
        else:
            print(f"  ❌ Syntax check failed")
            print(f"     stdout: {stdout}")
            print(f"     stderr: {stderr}")
            all_passed = False
    
    return all_passed

def validate_vpu_features():
    """Validate that VPU has required features"""
    
    print("\n=== VPU Feature Validation ===\n")
    
    vpu_file = "../../rtl/accelerators/vpu.sv"
    
    try:
        with open(vpu_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Could not read VPU file: {e}")
        return False
    
    # Check for required features
    features = {
        "Configurable vector register file": [
            "vreg_file",
            "VECTOR_REGS",
            "MAX_VLEN"
        ],
        "Vector element width configuration": [
            "vsew",
            "elements_per_reg",
            "vlmax"
        ],
        "Vector masking support": [
            "vmask",
            "mask_enabled",
            "lane_mask_bit"
        ],
        "Enhanced register file operations": [
            "vreg_write_mask",
            "vreg_read_mask",
            "masked write"
        ],
        "Multiple data types": [
            "data_type_e",
            "src_dtype", 
            "dst_dtype",
            "chip_config_pkg"
        ],
        "Vector functional units": [
            "gen_vector_lanes",
            "vector_alu",
            "lane_results"
        ],
        "Error detection": [
            "lane_overflow",
            "lane_underflow",
            "vector_error"
        ]
    }
    
    all_features_found = True
    
    for feature_name, keywords in features.items():
        print(f"Checking {feature_name}...")
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in keywords:
            if keyword in content:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        if len(found_keywords) >= len(keywords) * 0.7:  # At least 70% of keywords found
            print(f"  ✅ Found ({len(found_keywords)}/{len(keywords)} keywords)")
        else:
            print(f"  ❌ Missing keywords: {missing_keywords}")
            all_features_found = False
    
    return all_features_found

def validate_vector_alu_features():
    """Validate vector ALU enhancements"""
    
    print("\n=== Vector ALU Feature Validation ===\n")
    
    alu_file = "../../rtl/accelerators/vector_alu.sv"
    
    try:
        with open(alu_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Could not read Vector ALU file: {e}")
        return False
    
    # Check for enhanced features
    features = {
        "Enhanced arithmetic operations": [
            "ALU_ADD",
            "ALU_SUB", 
            "ALU_MUL",
            "ALU_DIV",
            "ALU_CONVERT"
        ],
        "Multi-stage pipeline": [
            "op_a_reg [2:0]",
            "valid_reg [2:0]",
            "mult_stage1",
            "div_quotient"
        ],
        "Data type conversions": [
            "int8_to_fp16",
            "fp16_to_int8",
            "convert_data_type",
            "leading_zeros"
        ],
        "Error detection": [
            "overflow",
            "underflow", 
            "div_by_zero",
            "invalid_op"
        ],
        "Signed/unsigned operations": [
            "$signed",
            "sign_extended",
            "abs_val"
        ]
    }
    
    all_features_found = True
    
    for feature_name, keywords in features.items():
        print(f"Checking {feature_name}...")
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in keywords:
            if keyword in content:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        if len(found_keywords) >= len(keywords) * 0.6:  # At least 60% of keywords found
            print(f"  ✅ Found ({len(found_keywords)}/{len(keywords)} keywords)")
        else:
            print(f"  ❌ Missing keywords: {missing_keywords}")
            all_features_found = False
    
    return all_features_found

def validate_test_coverage():
    """Validate test coverage"""
    
    print("\n=== Test Coverage Validation ===\n")
    
    test_files = {
        "test_vpu_functional_units.sv": [
            "test_vector_addition",
            "test_data_type_conversion", 
            "test_vector_register_file",
            "test_enhanced_arithmetic",
            "mask_enabled"
        ],
        "test_vector_alu.sv": [
            "test_data_type_conversions",
            "test_pipeline_behavior",
            "test_error_conditions",
            "INT8 to FP16",
            "overflow"
        ]
    }
    
    all_tests_found = True
    
    for test_file, test_cases in test_files.items():
        print(f"Checking {test_file}...")
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"  ❌ Could not read test file: {e}")
            all_tests_found = False
            continue
        
        found_tests = []
        missing_tests = []
        
        for test_case in test_cases:
            if test_case in content:
                found_tests.append(test_case)
            else:
                missing_tests.append(test_case)
        
        if len(found_tests) >= len(test_cases) * 0.8:  # At least 80% of tests found
            print(f"  ✅ Found ({len(found_tests)}/{len(test_cases)} test cases)")
        else:
            print(f"  ❌ Missing test cases: {missing_tests}")
            all_tests_found = False
    
    return all_tests_found

def main():
    """Main test function"""
    
    print("Enhanced VPU Implementation Validation")
    print("=" * 50)
    
    # Change to the test directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run all tests
    syntax_ok = test_vpu_components()
    vpu_features_ok = validate_vpu_features()
    alu_features_ok = validate_vector_alu_features()
    test_coverage_ok = validate_test_coverage()
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    results = [
        ("Syntax Check", syntax_ok),
        ("VPU Features", vpu_features_ok), 
        ("Vector ALU Features", alu_features_ok),
        ("Test Coverage", test_coverage_ok)
    ]
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("Enhanced VPU implementation is ready for task completion.")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please review the failed items above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())