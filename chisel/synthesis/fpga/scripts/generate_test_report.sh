#!/bin/bash
# Generate comprehensive test report

set -e

REPORT_FILE="test_results/test_report_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p test_results

echo "=== Generating Test Report ===" | tee $REPORT_FILE
echo "" | tee -a $REPORT_FILE

# Header
cat >> $REPORT_FILE << EOF
╔════════════════════════════════════════════════════════════╗
║           RISC-V AI Accelerator FPGA Test Report          ║
╚════════════════════════════════════════════════════════════╝

Date: $(date)
AFI ID: $(cat build/afi_info.txt 2>/dev/null | grep "AFI ID" | awk '{print $3}' || echo "N/A")
Instance: $(ec2-metadata --instance-type 2>/dev/null | awk '{print $2}' || echo "N/A")

EOF

# System Information
echo "System Information:" | tee -a $REPORT_FILE
echo "  OS: $(uname -s)" | tee -a $REPORT_FILE
echo "  Kernel: $(uname -r)" | tee -a $REPORT_FILE
echo "  FPGA Status: $(sudo fpga-describe-local-image -S 0 -H 2>/dev/null | grep Status || echo "Not loaded")" | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

# Run all tests
echo "Running Tests..." | tee -a $REPORT_FILE
echo "" | tee -a $REPORT_FILE

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Processor Boot
echo "Test 1: Processor Boot" | tee -a $REPORT_FILE
if ./test_processor_boot.sh >> $REPORT_FILE 2>&1; then
    echo "  ✓ PASS" | tee -a $REPORT_FILE
    ((TESTS_PASSED++))
else
    echo "  ✗ FAIL" | tee -a $REPORT_FILE
    ((TESTS_FAILED++))
fi
echo "" | tee -a $REPORT_FILE

# Test 2: UART
echo "Test 2: UART Communication" | tee -a $REPORT_FILE
if ./test_uart.sh >> $REPORT_FILE 2>&1; then
    echo "  ✓ PASS" | tee -a $REPORT_FILE
    ((TESTS_PASSED++))
else
    echo "  ✗ FAIL" | tee -a $REPORT_FILE
    ((TESTS_FAILED++))
fi
echo "" | tee -a $REPORT_FILE

# Test 3: GPIO
echo "Test 3: GPIO" | tee -a $REPORT_FILE
if ./test_gpio.sh >> $REPORT_FILE 2>&1; then
    echo "  ✓ PASS" | tee -a $REPORT_FILE
    ((TESTS_PASSED++))
else
    echo "  ✗ FAIL" | tee -a $REPORT_FILE
    ((TESTS_FAILED++))
fi
echo "" | tee -a $REPORT_FILE

# Test 4: CompactAccel
echo "Test 4: CompactAccel" | tee -a $REPORT_FILE
if ./test_compact_accel.sh >> $REPORT_FILE 2>&1; then
    echo "  ✓ PASS" | tee -a $REPORT_FILE
    ((TESTS_PASSED++))
else
    echo "  ✗ FAIL" | tee -a $REPORT_FILE
    ((TESTS_FAILED++))
fi
echo "" | tee -a $REPORT_FILE

# Test 5: Performance
echo "Test 5: Performance Benchmark" | tee -a $REPORT_FILE
if ./benchmark_gops.sh >> $REPORT_FILE 2>&1; then
    echo "  ✓ PASS" | tee -a $REPORT_FILE
    ((TESTS_PASSED++))
else
    echo "  ✗ FAIL" | tee -a $REPORT_FILE
    ((TESTS_FAILED++))
fi
echo "" | tee -a $REPORT_FILE

# Summary
TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
cat >> $REPORT_FILE << EOF

╔════════════════════════════════════════════════════════════╗
║                      Test Summary                          ║
╚════════════════════════════════════════════════════════════╝

Total Tests: $TOTAL_TESTS
Passed: $TESTS_PASSED
Failed: $TESTS_FAILED
Success Rate: $(echo "scale=1; $TESTS_PASSED * 100 / $TOTAL_TESTS" | bc)%

EOF

if [ $TESTS_FAILED -eq 0 ]; then
    echo "Overall Result: ✓ ALL TESTS PASSED" | tee -a $REPORT_FILE
else
    echo "Overall Result: ✗ SOME TESTS FAILED" | tee -a $REPORT_FILE
fi

echo "" | tee -a $REPORT_FILE
echo "Report saved to: $REPORT_FILE"

# Exit with appropriate code
exit $TESTS_FAILED
