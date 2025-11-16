#!/bin/bash
# Test UART communication

set -e

echo "=== UART Communication Test ==="

# Test string
TEST_STRING="Hello FPGA"

echo "Sending: $TEST_STRING"

# Send characters via UART TX register
for (( i=0; i<${#TEST_STRING}; i++ )); do
    CHAR="${TEST_STRING:$i:1}"
    ASCII=$(printf "%d" "'$CHAR")
    HEX=$(printf "0x%08x" $ASCII)
    
    # Write to UART TX register (address 0x1000)
    sudo fpga-pci-write -s 0 -b 0 -a 0x1000 -d $HEX
    
    # Wait for transmission (115200 baud = ~87us per byte)
    usleep 100
done

echo "Waiting for response..."
sleep 0.5

# Read received characters from UART RX buffer
RECEIVED=""
for (( i=0; i<${#TEST_STRING}; i++ )); do
    # Read from UART RX register (address 0x1004)
    DATA=$(sudo fpga-pci-read -s 0 -b 0 -a 0x1004)
    ASCII=$((DATA & 0xFF))
    CHAR=$(printf "\\$(printf '%03o' $ASCII)")
    RECEIVED="${RECEIVED}${CHAR}"
done

echo "Received: $RECEIVED"

if [ "$RECEIVED" == "$TEST_STRING" ]; then
    echo "✓ UART loopback successful"
    echo "PASS: UART communication test"
else
    echo "ERROR: Mismatch"
    echo "  Expected: $TEST_STRING"
    echo "  Received: $RECEIVED"
    exit 1
fi
