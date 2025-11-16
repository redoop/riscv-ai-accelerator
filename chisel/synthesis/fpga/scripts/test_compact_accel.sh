#!/bin/bash
# Test CompactAccel matrix multiplication

set -e

echo "=== CompactAccel Test ==="

# Base address for CompactAccel
ACCEL_BASE=0x3000
CTRL_REG=$((ACCEL_BASE + 0x00))
STATUS_REG=$((ACCEL_BASE + 0x04))
MATRIX_A_BASE=$((ACCEL_BASE + 0x100))
MATRIX_B_BASE=$((ACCEL_BASE + 0x200))
RESULT_BASE=$((ACCEL_BASE + 0x300))

# Test 1: 2x2 matrix multiplication
# A = [[1, 2], [3, 4]]
# B = [[5, 6], [7, 8]]
# Expected: [[19, 22], [43, 50]]

echo "Test 1: 2x2 matrix multiplication"

# Write matrix A
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_A_BASE + 0))) -d 0x00000001
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_A_BASE + 4))) -d 0x00000002
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_A_BASE + 8))) -d 0x00000003
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_A_BASE + 12))) -d 0x00000004

# Write matrix B
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_B_BASE + 0))) -d 0x00000005
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_B_BASE + 4))) -d 0x00000006
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_B_BASE + 8))) -d 0x00000007
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_B_BASE + 12))) -d 0x00000008

# Start computation (write size=2 and start bit)
sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $CTRL_REG) -d 0x00000003  # size=2, start=1

# Wait for completion
echo "  Computing..."
for i in {1..100}; do
    STATUS=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $STATUS_REG))
    if [ "$((STATUS & 0x1))" == "1" ]; then
        echo "  Completed in $i iterations"
        break
    fi
    usleep 1000
done

# Read results
R00=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $((RESULT_BASE + 0))))
R01=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $((RESULT_BASE + 4))))
R10=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $((RESULT_BASE + 8))))
R11=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $((RESULT_BASE + 12))))

echo "  Result: [[$R00, $R01], [$R10, $R11]]"
echo "  Expected: [[19, 22], [43, 50]]"

# Verify results
if [ "$R00" == "0x00000013" ] && [ "$R01" == "0x00000016" ] && \
   [ "$R10" == "0x0000002b" ] && [ "$R11" == "0x00000032" ]; then
    echo "  ✓ PASS"
else
    echo "  ✗ FAIL: Result mismatch"
    exit 1
fi

echo ""
echo "PASS: CompactAccel test completed"
