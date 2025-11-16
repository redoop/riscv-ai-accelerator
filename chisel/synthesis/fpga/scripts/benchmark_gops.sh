#!/bin/bash
# Benchmark GOPS performance

set -e

echo "=== GOPS Performance Benchmark ==="

ACCEL_BASE=0x3000
CTRL_REG=$((ACCEL_BASE + 0x00))
STATUS_REG=$((ACCEL_BASE + 0x04))
MATRIX_A_BASE=$((ACCEL_BASE + 0x100))
MATRIX_B_BASE=$((ACCEL_BASE + 0x200))

# Test parameters
MATRIX_SIZE=8
ITERATIONS=1000

echo "Configuration:"
echo "  Matrix size: ${MATRIX_SIZE}x${MATRIX_SIZE}"
echo "  Iterations: $ITERATIONS"
echo ""

# Prepare test matrices (random values)
echo "Preparing test data..."
for i in $(seq 0 $((MATRIX_SIZE * MATRIX_SIZE - 1))); do
    VALUE=$((RANDOM % 256))
    sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_A_BASE + i*4))) -d $(printf '0x%08x' $VALUE) > /dev/null
    VALUE=$((RANDOM % 256))
    sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $((MATRIX_B_BASE + i*4))) -d $(printf '0x%08x' $VALUE) > /dev/null
done

echo "Running benchmark..."
START_TIME=$(date +%s%N)

for iter in $(seq 1 $ITERATIONS); do
    # Start computation
    sudo fpga-pci-write -s 0 -b 0 -a $(printf '0x%04x' $CTRL_REG) -d $(printf '0x%08x' $((MATRIX_SIZE | 0x100))) > /dev/null
    
    # Wait for completion
    while true; do
        STATUS=$(sudo fpga-pci-read -s 0 -b 0 -a $(printf '0x%04x' $STATUS_REG))
        if [ "$((STATUS & 0x1))" == "1" ]; then
            break
        fi
    done
    
    # Progress indicator
    if [ $((iter % 100)) == 0 ]; then
        echo "  Progress: $iter/$ITERATIONS"
    fi
done

END_TIME=$(date +%s%N)
ELAPSED_NS=$((END_TIME - START_TIME))
ELAPSED_MS=$((ELAPSED_NS / 1000000))

# Calculate GOPS
# Operations per matrix multiply: 2 * N^3 (N^3 multiplications + N^3 additions)
OPS_PER_ITER=$((2 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE))
TOTAL_OPS=$((OPS_PER_ITER * ITERATIONS))
GOPS=$(echo "scale=2; $TOTAL_OPS / $ELAPSED_NS" | bc)

echo ""
echo "Results:"
echo "  Total time: ${ELAPSED_MS} ms"
echo "  Operations per iteration: $OPS_PER_ITER"
echo "  Total operations: $TOTAL_OPS"
echo "  Performance: ${GOPS} GOPS"
echo "  Target: 6.4 GOPS"
echo ""

# Check if target is met
TARGET_GOPS=6.0  # Allow 6% margin
if (( $(echo "$GOPS >= $TARGET_GOPS" | bc -l) )); then
    echo "✓ PASS: Performance target met"
else
    echo "✗ FAIL: Performance below target"
    exit 1
fi
