#!/bin/bash
# Test processor boot sequence

set -e

echo "=== Processor Boot Test ==="

# Check if FPGA is loaded
if ! sudo fpga-describe-local-image -S 0 -H | grep -q "loaded"; then
    echo "ERROR: FPGA not loaded"
    exit 1
fi

echo "Testing processor boot..."

# Assert reset
echo "Asserting reset..."
sudo fpga-pci-write -s 0 -b 0 -a 0x0000 -d 0x00000001
sleep 0.1

# Release reset
echo "Releasing reset..."
sudo fpga-pci-write -s 0 -b 0 -a 0x0000 -d 0x00000000
sleep 0.5

# Check processor status
echo "Checking processor status..."
STATUS=$(sudo fpga-pci-read -s 0 -b 0 -a 0x0004)

if [ "$STATUS" != "0x00000001" ]; then
    echo "ERROR: Processor not running (status: $STATUS)"
    exit 1
fi

echo "✓ Processor started successfully"
echo "PASS: Processor boot test"
